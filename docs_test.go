package main

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// collapseMarkdown lowercases s and collapses every whitespace run to one space,
// matching normalizeMarkdown for text already read from disk.
func collapseMarkdown(s string) string {
	return strings.ToLower(strings.Join(strings.Fields(s), " "))
}

// near reports whether a and b both occur in text within window bytes of each
// other. Proximity is what makes a documentation claim a claim: "driverkit" and
// "source of truth" anywhere in one file only says the file contains two phrases.
func near(text, a, b string, window int) bool {
	for i := 0; ; {
		j := strings.Index(text[i:], a)
		if j < 0 {
			return false
		}
		i += j
		if strings.Index(text[max(i-window, 0):min(i+len(b)+window, len(text))], b) >= 0 {
			return true
		}
		i++
	}
}

// glossarySection returns one CONTEXT.md glossary entry: the text between the
// "**Term**:" marker and the next bolded term, so a claim pins to the entry that
// defines the word rather than to anywhere in the file. Pass a lowercased term —
// the document is normalized first.
func glossarySection(norm, term string) string {
	marker := "**" + term + "**:"
	i := strings.Index(norm, marker)
	if i < 0 {
		return ""
	}
	rest := norm[i+len(marker):]
	if end := strings.Index(rest, "**"); end >= 0 {
		return rest[:end]
	}
	return rest
}

// satisfiesOrder reports whether some sentence of norm that mentions needle also
// lists needles in order. An ordering claim belongs to the sentence that makes
// it: elsewhere in the same document these words appear in another order, and a
// whole-document scan would be satisfied by accident.
func satisfiesOrder(norm, needle string, needles ...string) bool {
	for _, s := range sentencesContaining(norm, needle) {
		if indexOrder(s, needles...) {
			return true
		}
	}
	return false
}

// countAny returns how many of needles appear in hay.
func countAny(hay string, needles ...string) int {
	n := 0
	for _, s := range needles {
		if strings.Contains(hay, s) {
			n++
		}
	}
	return n
}

// TestContextDefinesEventAndPassLog pins the two glossary entries issue #19 adds.
// This vocabulary is the contract's user-facing surface: a host author who calls an
// event a "log line", or the stream a "transcript", reaches for text scraping
// instead of decoding, so each entry's _Avoid_ line has to retire the pre-event
// words it replaces.
func TestContextDefinesEventAndPassLog(t *testing.T) {
	const file = "CONTEXT.md"
	if _, err := os.Stat(file); err != nil {
		t.Fatalf("%s is missing and must define the event and pass log vocabulary: %v", file, err)
	}
	norm := normalizeMarkdown(t, file)

	event := glossarySection(norm, "event")
	if event == "" {
		t.Fatal("CONTEXT.md has no **Event** glossary entry")
	}
	for _, want := range []string{"driverkit", "json", "kind", "epoch millisecond", "stdout", "line"} {
		if !strings.Contains(event, want) {
			t.Errorf("the **Event** entry does not state that an event is one driverkit fact on stdout as one JSON object on one line, named by kind and timestamped in epoch milliseconds (missing %q)", want)
		}
	}
	if !strings.Contains(event, "_avoid_:") {
		t.Error("the **Event** entry has no _Avoid_ line")
	}
	if n := countAny(event, "log line", "record", "frame"); n < 2 {
		t.Errorf("the **Event** entry retires %d of the old words (log line, record, frame); want at least 2", n)
	}

	passLog := glossarySection(norm, "pass log")
	if passLog == "" {
		t.Fatal("CONTEXT.md has no **Pass log** glossary entry")
	}
	for _, want := range []string{"jsonl", "stdout", "stderr", "line by line"} {
		if !strings.Contains(passLog, want) {
			t.Errorf("the **Pass log** entry does not state that it is the whole JSONL stream one run writes to stdout, with its stderr interleaved and the result decoded line by line (missing %q)", want)
		}
	}
	if !strings.Contains(passLog, "_avoid_:") {
		t.Error("the **Pass log** entry has no _Avoid_ line")
	}
	if n := countAny(passLog, "transcript", "log file", "stdout dump"); n < 2 {
		t.Errorf("the **Pass log** entry retires %d of the old words (transcript, log file, stdout dump); want at least 2", n)
	}
}

// TestContextDefinesDriverConfig pins the glossary entry for the Driver Config.
// The file is the only channel that carries a host-side setting into a running
// Box, so an author who calls it a "project config" or a "dotfile" goes looking
// for it in the working directory — where a Target repo could steer its own
// repair — instead of under $HOME where only spindrift writes it.
func TestContextDefinesDriverConfig(t *testing.T) {
	const file = "CONTEXT.md"
	if _, err := os.Stat(file); err != nil {
		t.Fatalf("%s is missing and must define the driver config vocabulary: %v", file, err)
	}
	norm := normalizeMarkdown(t, file)

	entry := glossarySection(norm, "driver config")
	if entry == "" {
		t.Fatal("CONTEXT.md has no **Driver Config** glossary entry")
	}
	for _, want := range []string{"json", "base url", "api key", "idle timeout", "home", "spindrift_config", "envfiles", "target repo"} {
		if !strings.Contains(entry, want) {
			t.Errorf("the **Driver Config** entry does not state that it is the JSON file of base_url, api_key and the timeout idle bound at $HOME/.config/spindrift-driver/config.json, overridable by SPINDRIFT_CONFIG, written through envFiles and never read out of the Target repo (missing %q)", want)
		}
	}
	if !strings.Contains(entry, "_avoid_:") {
		t.Error("the **Driver Config** entry has no _Avoid_ line")
	}
	if n := countAny(entry, "dotfile", "project config", "settings file", "argv"); n < 2 {
		t.Errorf("the **Driver Config** entry retires %d of the old words (dotfile, project config, settings file, argv); want at least 2", n)
	}
	// The ladder belongs inside the definition: three places can set one field,
	// and a reader who cannot see which wins cannot predict what a run does.
	if !near(entry, "below", "above", 200) {
		t.Error("the **Driver Config** entry does not rank the file in one breath — what it sits below (the flag, the environment variable) and what it sits above (the built-in default)")
	}
}

// TestContextDefinesAgentFile pins the glossary entry for the subagent roster's
// unit file. The Agent File is where a child's model id and system prompt come
// from, so a reader who calls it a "profile" or a "persona" expects a document
// richer than the one the Driver parses — one whose tools and spawns lists bind
// something — and writes a roster whose restrictions bind nothing.
func TestContextDefinesAgentFile(t *testing.T) {
	const file = "CONTEXT.md"
	if _, err := os.Stat(file); err != nil {
		t.Fatalf("%s is missing and must define the agent file vocabulary: %v", file, err)
	}
	norm := normalizeMarkdown(t, file)

	entry := glossarySection(norm, "agent file")
	if entry == "" {
		t.Fatal("CONTEXT.md has no **Agent File** glossary entry")
	}
	for _, want := range []string{"frontmatter", "name", "description", "model", "tools", "spawns", "system prompt", "directory", "home", ".config/spindrift-driver/agents", "agents_dir"} {
		if !strings.Contains(entry, want) {
			t.Errorf("the **Agent File** entry does not state that it is one roster entry as a file — frontmatter of name, description, model, tools and spawns over a system-prompt body, in a directory defaulting to $HOME/.config/spindrift-driver/agents and redirected by the Driver Config's agents_dir (missing %q)", want)
		}
	}
	// The id is a router's key, not something the Driver composes: an author who
	// expects an effort suffix to be split off writes "model: x:high" and hands the
	// endpoint a model nobody serves.
	if !strings.Contains(entry, "verbatim") {
		t.Error("the **Agent File** entry does not say the model id is used verbatim, with no effort parsed out of it")
	}
	if !strings.Contains(entry, "_avoid_:") {
		t.Error("the **Agent File** entry has no _Avoid_ line")
	}
	if n := countAny(entry, "profile", "persona", "yaml config", "subagent definition"); n < 2 {
		t.Errorf("the **Agent File** entry retires %d of the old words (profile, persona, yaml config, subagent definition); want at least 2", n)
	}
}

// TestContextDefinesSubagentTool pins the glossary entry for the Tool that
// delegates to a roster entry. The entry is the one place the delegation's whole
// shape is stated: that the child is this same binary re-executed rather than a
// bespoke call the parent builds, that a child's events are the parent's events
// with attribution, that a failing child is a Tool Result rather than a failed
// run, and that a child cannot spawn in turn.
func TestContextDefinesSubagentTool(t *testing.T) {
	const file = "CONTEXT.md"
	if _, err := os.Stat(file); err != nil {
		t.Fatalf("%s is missing and must define the subagent tool vocabulary: %v", file, err)
	}
	norm := normalizeMarkdown(t, file)

	entry := glossarySection(norm, "subagent tool")
	if entry == "" {
		t.Fatal("CONTEXT.md has no **Subagent Tool** glossary entry")
	}
	for _, want := range []string{"delegate", "named", "re-execut", "role", "in_subagent", "usage", "result", "subagent", "one at a time", "hook manifest"} {
		if !strings.Contains(entry, want) {
			t.Errorf("the **Subagent Tool** entry does not state that it delegates one task to a named roster entry by re-executing this same binary, relaying the child's events with role and in_subagent, rolling its usage into the parent's totals, returning its answer as the Tool Result that names the agent in subagent, spawning one at a time, and inheriting the Hook Manifest (missing %q)", want)
		}
	}
	// A child that dies must not take the run down with it, or the Model has no
	// turn left in which to pick another approach.
	if !near(entry, "error", "parent", 240) {
		t.Error("the **Subagent Tool** entry does not state that a child failure is an error Tool Result the parent goes on with")
	}
	// Recursion is bounded by the offer, not by the roster's spawns field.
	if !near(entry, "child", "never", 240) {
		t.Error("the **Subagent Tool** entry does not state that a child is never offered the Tool itself")
	}
	// Inheritance is what keeps the guards holding across a delegation: a child
	// that resolved its own manifest would run the command its parent was refused.
	if !near(entry, "inherit", "hook manifest", 240) {
		t.Error("the **Subagent Tool** entry does not state that a child inherits the Hook Manifest alongside the Driver Config")
	}
	if !strings.Contains(entry, "_avoid_:") {
		t.Error("the **Subagent Tool** entry has no _Avoid_ line")
	}
	if n := countAny(entry, "subprocess", "fork", "nested agent", "mcp"); n < 2 {
		t.Errorf("the **Subagent Tool** entry retires %d of the old words (subprocess, fork, nested agent, MCP); want at least 2", n)
	}
}

// TestContextDefinesGuardHookAndHookManifest pins the two glossary entries the
// pre-call Guard Hook chain adds. Both name things a host bakes and the Driver
// only reads: an author who calls the script "middleware" or the manifest a
// "policy file" goes looking for the policy in the Driver's source, where there is
// none, and writes a manifest whose verdicts the run cannot honour.
func TestContextDefinesGuardHookAndHookManifest(t *testing.T) {
	const file = "CONTEXT.md"
	if _, err := os.Stat(file); err != nil {
		t.Fatalf("%s is missing and must define the guard hook vocabulary: %v", file, err)
	}
	norm := normalizeMarkdown(t, file)

	hook := glossarySection(norm, "guard hook")
	if hook == "" {
		t.Fatal("CONTEXT.md has no **Guard Hook** glossary entry")
	}
	// The wire shape first: a hook author writes against these field names, and
	// nothing else in the repo tells them what to read on stdin or answer with.
	for _, want := range []string{"script", "stdin", "tool_name", "tool_input", "hookspecificoutput", "permissiondecision", "permissiondecisionreason"} {
		if !strings.Contains(hook, want) {
			t.Errorf("the **Guard Hook** entry does not state that the Driver runs a host-baked script, hands it tool_name and tool_input on stdin, and reads permissionDecision and permissionDecisionReason out of hookSpecificOutput (missing %q)", want)
		}
	}
	// A deny blocks, and it blocks as an error Tool Result — the only form of
	// refusal the Model has a turn in which to act on.
	if !near(hook, "deny", "error tool result", 240) {
		t.Error("the **Guard Hook** entry does not state that a deny blocks the call with an error Tool Result naming the script")
	}
	// The fail-closed half, case by case: the Driver cannot tell a broken hook from
	// an approving one, so every one of these refuses.
	for _, want := range []string{"cannot be run", "overrun", "non-zero", "nothing readable"} {
		if !strings.Contains(hook, want) {
			t.Errorf("the **Guard Hook** entry does not fail closed on one of its cases (missing %q)", want)
		}
	}
	// The rewrite half: a hook can hand back the command the run goes on with, and
	// one the manifest declares a rewriter owes that answer — withholding it is the
	// same doubt as a hook that stopped answering altogether.
	if !near(hook, "updatedinput", "replaces", 200) {
		t.Error("the **Guard Hook** entry does not state that updatedInput.command replaces the command the run goes on with")
	}
	if !near(hook, "rewriter", "refus", 200) {
		t.Error("the **Guard Hook** entry does not state that a hook the manifest declares a rewriter, answering with no rewrite, is refused by name like any other misbehaviour")
	}
	// jq belongs to the guarded run rather than to one hook: the baked scripts read
	// their own request with it, so without it every hook would answer having
	// inspected nothing and silence would read as a pass.
	if !strings.Contains(hook, "jq") {
		t.Error("the **Guard Hook** entry does not state that jq has to be runnable for a guarded run")
	}
	if !near(hook, "jq", "refus", 240) {
		t.Error("the **Guard Hook** entry does not tie a missing jq to refusing the guarded calls")
	}
	// Silence passing is what separates a guard from a gate: a hook with no opinion
	// must let the call through, or every command an entry does not name is refused.
	if !near(hook, "silence", "no objection", 160) {
		t.Error("the **Guard Hook** entry does not state that silence on a clean exit raises no objection")
	}
	// The post-call variant, and the contrast that is the point of naming it: the
	// same hook mechanics, the opposite default, because there the Driver guards a
	// command that has not run and here it holds a result already earned.
	if !strings.Contains(hook, "tool_response") {
		t.Error("the **Guard Hook** entry does not state that the summary hook is handed the finished Tool Result as tool_response")
	}
	if !near(hook, "updatedtooloutput", "replaces", 200) {
		t.Error("the **Guard Hook** entry does not state that updatedToolOutput replaces the finished Tool Result")
	}
	if !near(hook, "summary hook", "fail", 200) || !near(hook, "summary hook", "open", 200) {
		t.Error("the **Guard Hook** entry does not state that the post-call summary hook fails open")
	}
	if !near(hook, "unchanged", "open", 320) {
		t.Error("the **Guard Hook** entry does not state that a summary which cannot run leaves the Tool Result unchanged")
	}
	if !near(hook, "pre-call", "closed", 240) {
		t.Error("the **Guard Hook** entry does not contrast the pre-call chain, which fails closed")
	}
	// What it is for, and that none of the policy lives in the Driver.
	for _, want := range []string{"credential", "background"} {
		if !strings.Contains(hook, want) {
			t.Errorf("the **Guard Hook** entry does not name what a Guard Hook is for (missing %q)", want)
		}
	}
	if !near(hook, "hardcode", "hook name", 160) {
		t.Error("the **Guard Hook** entry does not state that the Driver hardcodes no hook names")
	}
	if !strings.Contains(hook, "_avoid_:") {
		t.Error("the **Guard Hook** entry has no _Avoid_ line")
	}
	if n := countAny(hook, "middleware", "interceptor", "permission prompt", "guardrail", "filter"); n < 2 {
		t.Errorf("the **Guard Hook** entry retires %d of the wrong words (middleware, interceptor, permission prompt, guardrail, filter); want at least 2", n)
	}

	manifest := glossarySection(norm, "hook manifest")
	if manifest == "" {
		t.Fatal("CONTEXT.md has no **Hook Manifest** glossary entry")
	}
	for _, want := range []string{"json", "run order", "path", "tool name", "rewrite", "hook_manifest", "no default path"} {
		if !strings.Contains(manifest, want) {
			t.Errorf("the **Hook Manifest** entry does not state that it is the JSON file listing the pre-call Guard Hooks in run order — script path, hook-protocol tool names, whether it rewrites — named only by the Driver Config's hook_manifest with no default path (missing %q)", want)
		}
	}
	// The post-call side of the file: one summary hook rather than a second chain,
	// and an entry with no rewrites flag to carry, the input being long gone.
	if !strings.Contains(manifest, "post") {
		t.Error("the **Hook Manifest** entry does not name the post-call summary hook's post key")
	}
	if !near(manifest, "post", "one", 160) {
		t.Error("the **Hook Manifest** entry does not state that post is one entry rather than a list")
	}
	if !near(manifest, "post", "pre", 200) {
		t.Error("the **Hook Manifest** entry does not place the post summary hook alongside the pre list")
	}
	if !near(manifest, "post", "rewrites", 240) {
		t.Error("the **Hook Manifest** entry does not state that a post entry names the script and the tool names only, there being no rewrites to declare after the call")
	}
	// The two states a reader has to be able to tell apart: nothing configured, so
	// nothing is guarded, versus something configured and unreadable.
	if !near(manifest, "unconfigured", "unguarded", 200) {
		t.Error("the **Hook Manifest** entry does not state that an unconfigured manifest means nothing is guarded")
	}
	if !near(manifest, "cannot be read", "startup", 240) {
		t.Error("the **Hook Manifest** entry does not state that a configured manifest which cannot be read or parsed stops the run at startup")
	}
	// Inheritance, or delegation is an escape hatch out of every guard.
	if !near(manifest, "child", "inherit", 200) {
		t.Error("the **Hook Manifest** entry does not state that a spawned child inherits the manifest the way the Driver Config is inherited")
	}
	if !strings.Contains(manifest, "_avoid_:") {
		t.Error("the **Hook Manifest** entry has no _Avoid_ line")
	}
	if n := countAny(manifest, "hook config", "plugin list", "policy file", "pre-flight"); n < 2 {
		t.Errorf("the **Hook Manifest** entry retires %d of the wrong words (hook config, plugin list, policy file, pre-flight); want at least 2", n)
	}
}

// TestContextRetiresBashToolAsOnlyTool pins the retirement of the pre-subagent
// line: #25 adds a second Tool, so an entry still calling the Bash Tool the one in
// scope is a definition of the offering that no longer matches the schema sent to
// the Model.
func TestContextRetiresBashToolAsOnlyTool(t *testing.T) {
	const file = "CONTEXT.md"
	norm := normalizeMarkdown(t, file)

	entry := glossarySection(norm, "bash tool")
	if entry == "" {
		t.Fatal("CONTEXT.md has no **Bash Tool** glossary entry")
	}
	if n := countAny(entry, "one tool in scope", "only tool", "single tool", "only one tool", "the one tool"); n > 0 {
		t.Errorf("the **Bash Tool** entry still claims exclusivity (%d matches): the Subagent Tool is offered whenever Agent Files are present", n)
	}
	for _, want := range []string{"shell", "output"} {
		if !strings.Contains(entry, want) {
			t.Errorf("the **Bash Tool** entry does not say what the Tool does — run a shell command and return its output (missing %q)", want)
		}
	}
}

// readADRs loads every ADR from disk keyed by file name, mapping to its raw text:
// which decision supersedes which is a property of the record set, not of a file
// name somebody chose, so the directory is read rather than its names assumed.
func readADRs(t *testing.T) map[string]string {
	t.Helper()
	const dir = "docs/adr"
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read %s: %v", dir, err)
	}
	ads := map[string]string{}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatalf("read %s: %v", e.Name(), err)
		}
		ads[e.Name()] = string(raw)
	}
	if len(ads) == 0 {
		t.Fatalf("%s holds no ADRs", dir)
	}
	return ads
}

// findADR returns the name of the first ADR (file-name order, so the pick is
// stable) whose text makes the claim: both phrases present, within window of each
// other.
func findADR(t *testing.T, ads map[string]string, a, b string) string {
	t.Helper()
	names := make([]string, 0, len(ads))
	for name := range ads {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if near(collapseMarkdown(ads[name]), a, b, 160) {
			return name
		}
	}
	return ""
}

// TestADRRecordsSpindriftFirstSupersedesStandaloneFirst pins the lineage shift:
// ADR-0002 deferred integration and kept a standalone CLI shape, and that deferral
// is now reversed in the record. It also pins the consequence — this repo carries
// no copy of driverkit, so the committed golden Pass log is the hand-off artifact
// between the two repositories.
func TestADRRecordsSpindriftFirstSupersedesStandaloneFirst(t *testing.T) {
	ads := readADRs(t)

	name := findADR(t, ads, "supersedes", "standalone")
	if name == "" {
		t.Fatal("no ADR in docs/adr supersedes the standalone-first decision of ADR-0002")
	}
	text := collapseMarkdown(ads[name])
	if !strings.Contains(text, "spindrift-first") && !strings.Contains(text, "spindrift first") {
		t.Errorf("%s supersedes ADR-0002 but never names spindrift-first as what replaces it", name)
	}
	// The point of the shift is that the consumer defines the interface.
	if !strings.Contains(text, "only if spindrift uses it") && !strings.Contains(text, "primary consumer") {
		t.Errorf("%s must state that an interface exists only if spindrift uses it", name)
	}
	// Standalone keeps working, but as a consequence rather than a goal.
	if !strings.Contains(text, "side effect") && !strings.Contains(text, "still works") {
		t.Errorf("%s must state that standalone runs keep working as a side effect of the spindrift interface", name)
	}
	// The hand-off: no copy of driverkit here, so the golden log is the artifact.
	if !near(text, "driverkit", "no copy", 160) && !near(text, "driverkit", "carries no", 160) {
		t.Errorf("%s must state that this repo carries no copy of driverkit", name)
	}
	if !strings.Contains(text, "testdata/pass.jsonl") && !strings.Contains(text, "golden pass log") {
		t.Errorf("%s must name the committed golden Pass log as the hand-off artifact", name)
	}
	if !strings.Contains(text, "-update") {
		t.Errorf("%s must name the explicit -update flag as the only way the golden Pass log is regenerated", name)
	}
}

// TestADRRecordsEventsOnlyStdoutContract pins the decisions that make stdout a
// single machine-readable contract: events are the whole of it, driverkit is the
// grammar's source of truth, stderr and exit codes keep their old jobs, and the
// committed golden log is what the host-side Strategy decodes against.
func TestADRRecordsEventsOnlyStdoutContract(t *testing.T) {
	ads := readADRs(t)

	name := findADR(t, ads, "driverkit", "source of truth")
	if name == "" {
		t.Fatal("no ADR in docs/adr records driverkit as the events grammar's source of truth")
	}
	text := collapseMarkdown(ads[name])
	if !strings.Contains(text, "cmd/launcher/internal/driver/driverkit") {
		t.Errorf("%s must name driverkit by its path in spindrift, not just by package name", name)
	}
	if !strings.Contains(text, "spindrift") {
		t.Errorf("%s must name spindrift as the repository that owns the grammar", name)
	}
	for _, want := range []string{"snake_case", "epoch millisecond"} {
		if !strings.Contains(text, want) {
			t.Errorf("%s does not pin the wire shape it inherits from driverkit (%s)", name, want)
		}
	}
	// A flag would make stdout two contracts, and the default one is the one the
	// host cannot parse — so the absence of a flag is decided explicitly.
	if !strings.Contains(text, "no output-mode flag") && !strings.Contains(text, "no output mode flag") {
		t.Errorf("%s must state that there is no output-mode flag", name)
	}
	if !strings.Contains(text, "text-only") && !strings.Contains(text, "text only") {
		t.Errorf("%s must record that the earlier text-only stdout contract is retired", name)
	}
	if !near(text, "final answer", "result", 120) {
		t.Errorf("%s must state that the final answer is the result event's text", name)
	}
	// What does not move: progress on stderr, exit codes authoritative.
	if !near(text, "stderr", "progress", 160) {
		t.Errorf("%s must state that progress stays on stderr", name)
	}
	if !near(text, "exit code", "authoritative", 160) {
		t.Errorf("%s must state that exit codes stay the authoritative signal", name)
	}
	// The golden log exists so the host can build a decoder against something real.
	if !near(text, "golden", "strategy", 240) {
		t.Errorf("%s must state that the committed golden Pass log is what the host-side Strategy consumes", name)
	}
	if !strings.Contains(text, "plain json") {
		t.Errorf("%s must state that this repo's tests read the emitted lines as plain JSON rather than through driverkit types", name)
	}
}

// TestADRMarkSuperseded0002 pins the visible pointer in the superseded ADR: a
// reader who opens ADR-0002 first must be told there, before the old argument,
// that ADR-0003 replaced it. The repo uses no `Status:` headers, so the claim is
// pinned as one line near the title rather than as a header of a given form.
func TestADRMarkSuperseded0002(t *testing.T) {
	ads := readADRs(t)

	const name = "0002-standalone-first-driver-contract-shape.md"
	raw, ok := ads[name]
	if !ok {
		t.Fatalf("docs/adr/%s is missing", name)
	}
	lines := strings.Split(raw, "\n")
	var marked, markedAt int = 0, -1
	for i, line := range lines {
		if strings.Contains(strings.ToLower(line), "superseded") {
			marked++
			if markedAt < 0 {
				markedAt = i
			}
		}
	}
	if marked == 0 {
		t.Fatalf("%s is superseded by ADR-0003 but does not say so", name)
	}
	if marked > 1 {
		t.Errorf("%s mentions being superseded in %d places; the pointer is one line under the title, not a running annotation", name, marked)
	}
	// Position matters: a pointer below the argument reads as part of the argument.
	nonEmpty := 0
	for _, line := range lines[:markedAt+1] {
		if strings.TrimSpace(line) != "" {
			nonEmpty++
		}
	}
	if nonEmpty > 3 {
		t.Errorf("%s: the superseded pointer sits at non-empty line %d; it belongs directly under the title", name, nonEmpty)
	}
	pointer := strings.ToLower(lines[markedAt])
	if !strings.Contains(pointer, "0003") && !strings.Contains(pointer, "spindrift-first") {
		t.Errorf("%s is marked superseded without naming ADR-0003 as its successor: %q", name, lines[markedAt])
	}
}

// TestADRRecordsDriverConfig pins ADR-0005: the Driver Config is the only channel
// that carries host-side settings into a running Box, so the record has to name
// what it reverses (spec #1's "no config files"), what it leaves alone (ADR-0001),
// and every rule that keeps a mistyped file from quietly changing where a Box's
// traffic goes.
func TestADRRecordsDriverConfig(t *testing.T) {
	ads := readADRs(t)

	const name = "0005-driver-config.md"
	if found := findADR(t, ads, "driver config", "envfiles"); found != name {
		t.Fatalf("the driver config / envFiles claim is recorded in %q, want %s", found, name)
	}
	raw, ok := ads[name]
	if !ok {
		t.Fatalf("docs/adr/%s is missing", name)
	}
	text := collapseMarkdown(raw)

	// The reversal is stated as a reversal, and attributed to the spec it reverses.
	if !strings.Contains(text, "no config files") {
		t.Errorf("%s never names the rule it reverses (spec #1's \"no config files\")", name)
	}
	if !near(text, "no config files", "spec #1", 240) {
		t.Errorf("%s reverses \"no config files\" without saying, in the same breath, that the rule is spec #1's", name)
	}
	// The field set is closed, and the settings that stay outside it are named.
	for _, want := range []string{"base_url", "api_key", "timeout"} {
		if !strings.Contains(text, want) {
			t.Errorf("%s does not name the config field %s", name, want)
		}
	}
	if !near(text, "base_url", "timeout", 240) {
		t.Errorf("%s does not present base_url, api_key and timeout as one closed field set", name)
	}
	if !near(text, "model", "flag", 200) {
		t.Errorf("%s must state that the model name stays a flag and out of the file", name)
	}
	// Location plus its one override, and the default is home-relative.
	if !near(text, "home", "spindrift_config", 240) {
		t.Errorf("%s does not give the home-relative default and the SPINDRIFT_CONFIG override together", name)
	}
	if !strings.Contains(text, ".config/spindrift-driver/config.json") {
		t.Errorf("%s does not name the default path .config/spindrift-driver/config.json", name)
	}
	// One ladder, stated in the order it is applied — pinned inside the sentence
	// that states it, because the title and the earlier prose use these same words
	// out of that order.
	if !satisfiesOrder(text, "ladder", "flag", "environment", "driver config", "default") {
		t.Errorf("%s has no one-sentence statement of precedence in order: flag, then environment, then Driver Config, then built-in default", name)
	}
	// Strictness: a bad file stops the run instead of being quietly half-read.
	if !near(text, "unknown field", "startup error", 240) {
		t.Errorf("%s does not make an unknown field a startup error naming the file and the problem", name)
	}
	if !near(text, "fall back", "typo", 240) {
		t.Errorf("%s must state that a typo never falls back to a default in silence", name)
	}
	// The key never travels by a route that makes it readable in ps output.
	if !near(text, "api key", "process listing", 240) {
		t.Errorf("%s does not say the api key never appears in a process listing", name)
	}
	// The repository under repair gets no vote in the run that repairs it.
	if !near(text, "target repo", "working directory", 240) {
		t.Errorf("%s does not say the Driver reads no config from the Target repo's working directory", name)
	}
	if !strings.Contains(raw, "## Considered Options") {
		t.Errorf("%s has no ## Considered Options section", name)
	}
	if n := strings.Count(raw, "— rejected:"); n < 4 {
		t.Errorf("%s rejects %d options in the `- **Option** — rejected: reason` form; want at least 4", name, n)
	}
}
