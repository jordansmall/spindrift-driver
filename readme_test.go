package main

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// normalizeMarkdown lowercases the README and collapses every whitespace run
// to one space, so the assertions below match the prose regardless of line
// wrapping, heading depth or code fences.
func normalizeMarkdown(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return strings.ToLower(strings.Join(strings.Fields(string(raw)), " "))
}

// markdownSectionAround returns the text between the heading above needle and
// the next heading after it, so a claim can be pinned to the place that makes
// it ("goimports is offered by the dev-shell section", not "somewhere").
func markdownSectionAround(norm, needle string) string {
	i := strings.Index(norm, needle)
	if i < 0 {
		return ""
	}
	start := strings.LastIndex(norm[:i], "# ")
	end := strings.Index(norm[i+1:], "# ")
	if end < 0 {
		return norm[start:]
	}
	return norm[start : i+1+end]
}

// TestReadmeDocumentsNixWorkflowAndSmokeTest pins the documentation half of
// the flake issue: the README is the only place the dev-shell entry and the
// live-server smoke test are described, and the smoke test is manual — it
// never runs in CI or under `nix flake check` — so an unstated command is a
// workflow nobody can perform.
func TestReadmeDocumentsNixWorkflowAndSmokeTest(t *testing.T) {
	const file = "README.md"
	if _, err := os.Stat(file); err != nil {
		t.Fatalf("%s is missing and must document the nix workflow and the live smoke test: %v", file, err)
	}
	norm := normalizeMarkdown(t, file)

	for _, want := range []string{
		"nix build",
		"./result/bin/spindrift-driver",
		"nix develop",
		"nix flake check",
		"--model",
		"--base-url",
		"$spindrift_base_url",
		"$spindrift_api_key",
		"llama.cpp",
	} {
		if !strings.Contains(norm, want) {
			t.Errorf("%s does not mention %q", file, want)
		}
	}

	// The dev shell exists to hand out the Go toolchain, so its own section
	// must name all three tools the shell installs.
	devShell := markdownSectionAround(norm, "nix develop")
	if devShell == "" {
		t.Fatal("no section documents `nix develop`")
	}
	for _, tool := range []string{"go", "gopls", "goimports"} {
		if !regexp.MustCompile(`\b` + tool + `\b`).MatchString(devShell) {
			t.Errorf("the `nix develop` section does not state that the shell provides %s", tool)
		}
	}

	// `nix flake check` is the gate that runs the suite; documenting it without
	// saying what it runs would not distinguish it from a build.
	checks := markdownSectionAround(norm, "nix flake check")
	if checks == "" {
		t.Fatal("no section documents `nix flake check`")
	}
	if !strings.Contains(checks, "go test") {
		t.Error("the `nix flake check` section must say the check runs the go test suite")
	}

	// The live smoke test is the only end-to-end verification the project has,
	// and it needs all three knobs from main.go to point at a remote server.
	smoke := markdownSectionAround(norm, "smoke test")
	if smoke == "" {
		t.Fatal("no section documents the manual live smoke test")
	}
	for _, want := range []string{"--model", "--base-url", "$spindrift_base_url", "$spindrift_api_key", "manual"} {
		if !strings.Contains(smoke, want) {
			t.Errorf("the smoke-test section does not document %s", want)
		}
	}
	// The key is read from the environment only; a flag would put the secret in
	// argv, so the README must never show one.
	if strings.Contains(norm, "--api-key") {
		t.Error("README documents an --api-key flag; SPINDRIFT_API_KEY is env-only by design")
	}

	// llama.cpp is excluded on purpose; the exclusion has to travel with its
	// reason or a reader tries to add it back.
	llama := markdownSectionAround(norm, "llama.cpp")
	if llama == "" {
		t.Fatal("no section states that llama.cpp is excluded")
	}
	if !strings.Contains(llama, "remote") {
		t.Error("the llama.cpp exclusion must state that the Model server is always remote")
	}
	if !regexp.MustCompile(`not packaged|is not included|excluded|never packaged|not bundled`).MatchString(llama) {
		t.Error("the llama.cpp note must say it is deliberately not packaged")
	}
}

// TestReadmeDocumentsChatTimeout pins the README to the --timeout flag: the
// default is read from chatAttemptTimeout instead of typed here, so raising it
// fails this test until the prose states the new value, and a timed-out attempt
// has to appear among the failure modes an operator can hit.
func TestReadmeDocumentsChatTimeout(t *testing.T) {
	const file = "README.md"
	norm := normalizeMarkdown(t, file)

	if !strings.Contains(norm, "--timeout") {
		t.Errorf("%s does not document the --timeout flag", file)
	}
	if want := chatAttemptTimeout.String(); !strings.Contains(norm, want) {
		t.Errorf("%s does not state the default timeout %s", file, want)
	}

	usageBlock := markdownSectionAround(norm, "usage: spindrift-driver")
	if usageBlock == "" {
		t.Fatal("no usage block in " + file)
	}
	// The fenced usage line is what a reader copies, so the flag has to appear in
	// its bracketed option form there, not merely in the prose below it.
	if !regexp.MustCompile(`usage: spindrift-driver.*\[--timeout <dur>\]`).MatchString(usageBlock) {
		t.Error("the usage line does not list [--timeout <dur>] next to [--max-turns <n>]")
	}

	smoke := markdownSectionAround(norm, "smoke test")
	if smoke == "" {
		t.Fatal("no section documents the manual live smoke test")
	}
	if !regexp.MustCompile(`timed[- ]out|times out|timeout`).MatchString(smoke) {
		t.Error("the smoke-test failure modes must name an attempt that times out alongside the unreachable server, rejected request and --max-turns cap")
	}

	// Responses stream, so what trips --timeout is silence, not a turn that took
	// too long overall. Prose that still describes a whole-attempt cap points an
	// operator at the wrong diagnosis when a live run stalls.
	if !regexp.MustCompile(`stall|silent|silence|goes silent`).MatchString(smoke) {
		t.Error("the smoke-test failure modes describe a stalled stream (silence past the bound), not an attempt that merely exceeds --timeout")
	}

	prose := markdownSectionAround(norm, "usage: spindrift-driver")
	if prose == "" {
		t.Fatal("no usage block in " + file)
	}
	for _, want := range []string{"idle", "gap between received bytes", "response headers"} {
		if !strings.Contains(prose, want) {
			t.Errorf("the README does not describe --timeout as an idle bound (%s)", want)
		}
	}
	// When to raise it, stated as the case that actually needs it.
	if !regexp.MustCompile(`raise it[^.]*first token`).MatchString(prose) {
		t.Error("the README does not say to raise --timeout when the server is slow to emit its first token (or a proxy buffers)")
	}
	if regexp.MustCompile(`raise it[^.]*generat`).MatchString(prose) {
		t.Error("the README still tells the operator to raise --timeout for slow generation; a streaming turn no longer needs a raise")
	}
}

// sentencesContaining returns the sentences of sec that mention needle, so a claim
// is pinned to one statement rather than to the file: two facts in two unrelated
// paragraphs of one section do not make the claim.
func sentencesContaining(sec, needle string) []string {
	var out []string
	for _, s := range strings.Split(sec, ". ") {
		if strings.Contains(s, needle) {
			out = append(out, s)
		}
	}
	return out
}

// indexOrder reports whether needles appear in hay in the given order.
func indexOrder(hay string, needles ...string) bool {
	at := -1
	for _, n := range needles {
		i := strings.Index(hay, n)
		if i <= at {
			return false
		}
		at = i
	}
	return true
}

// TestReadmeDocumentsEventStreamOnStdout pins the rewritten output contract. stdout
// is no longer prose, so the README is the only place a caller learns what to parse
// and how to pull the answer out of it; every token asserted here is one they would
// type or grep for.
func TestReadmeDocumentsEventStreamOnStdout(t *testing.T) {
	const file = "README.md"
	norm := normalizeMarkdown(t, file)

	for _, want := range []string{
		"jsonl", "kind", "epoch millisecond", "start", "reasoning", "text", "usage", "tool_call",
		"tool_result", "result", "error", "tool_id", "jq", "stderr", "testdata/pass.jsonl",
	} {
		if !strings.Contains(norm, want) {
			t.Errorf("%s does not document the stdout event stream (%s)", file, want)
		}
	}
	// The text-only contract is retired; a README that still promises the bare
	// answer on stdout sends a reader to parse something no longer emitted.
	if regexp.MustCompile(`final answer on stdout|answer is the stdout|stdout is the final answer`).MatchString(norm) {
		t.Error("README still promises the final answer itself on stdout; it is the result event's text")
	}

	out := markdownSectionAround(norm, "jsonl")
	if out == "" {
		t.Fatal("no section documents the JSONL event stream on stdout")
	}

	// One JSON object per line, named by kind, timestamped in epoch milliseconds —
	// the shape a decoder is written against.
	for _, want := range []string{"one json object per line", "kind", "epoch millisecond"} {
		if !strings.Contains(out, want) {
			t.Errorf("the event-stream section does not state the wire shape (%s)", want)
		}
	}
	// Emission order is part of the contract: a host stream-parsing the log needs
	// to know tool_result follows its tool_call and result closes a successful run.
	// reasoning rides ahead of the prose it produced, so a decoder that assumes
	// text is the first prose of a call mistakes thinking for the answer.
	if !indexOrder(out, "`start`", "`reasoning`", "`text`") {
		t.Error("the event-stream section does not list reasoning between start and text, the order it is emitted in")
	}
	if !indexOrder(out, "start", "text", "usage", "tool_call", "tool_result") && !indexOrder(out, "`start`", "`text`", "`usage`", "`tool_call`", "`tool_result`") {
		t.Error("the event-stream section does not give the order the kinds appear in (start, text, usage, tool_call, tool_result, then result or error)")
	}
	if !strings.Contains(out, "pair by") || !strings.Contains(out, "tool_id") {
		t.Error("the event-stream section does not say tool_call and tool_result pair by tool_id")
	}
	if !regexp.MustCompile(`exactly one .* per model call|one usage event per model call`).MatchString(out) {
		t.Error("the event-stream section does not state that usage is emitted exactly once per Model call")
	}
	if !strings.Contains(out, "num_turns") || !strings.Contains(out, "duration_ms") || !strings.Contains(out, "models") {
		t.Error("the result event's fields (num_turns, duration_ms, token totals, models) are not documented")
	}
	if !strings.Contains(out, "tokens") {
		t.Error("the result event's token totals are not documented")
	}
	// Every non-zero exit is preceded by an error event — the reason a host can
	// trust exit codes and stdout together.
	if !strings.Contains(out, "non-zero exit") || !strings.Contains(out, "raw") {
		t.Error("the event-stream section does not state that every non-zero exit is preceded by an error event carrying the raw error text")
	}
	// The answer and the failure text are both pulled with jq, the same way.
	if !strings.Contains(out, "jq") || !strings.Contains(out, ".text") {
		t.Error("the event-stream section shows no jq recipe reading .text out of an event")
	}
	for _, kind := range []string{"result", "error"} {
		if !regexp.MustCompile(`select\(\.kind\s*==\s*"` + kind + `"\)`).MatchString(out) {
			t.Errorf("the event-stream section shows no jq recipe for the %s event's text", kind)
		}
	}
	if !strings.Contains(out, "jq -r") {
		t.Error("the jq recipe is not given in the runnable `jq -r 'select(...)' | .text` form")
	}
	// Progress never moves to stdout, and -h is an exit path like any other.
	if !strings.Contains(out, "progress") || !strings.Contains(out, "stderr") {
		t.Error("the event-stream section does not state that progress stays on stderr")
	}
	var helpLines []string
	for _, s := range sentencesContaining(out, "-h") {
		if strings.Contains(s, "stderr") && strings.Contains(s, "usage") {
			helpLines = append(helpLines, s)
		}
	}
	if len(helpLines) == 0 {
		t.Error("no statement says -h/--help writes its usage to stderr")
	}

	// The golden artifact is the hand-off, and it only moves under an explicit flag.
	if !strings.Contains(norm, "go test -run testgoldenpasslog -update") {
		t.Error("README does not give the regeneration command `go test -run TestGoldenPassLog -update`")
	}
	if !near(norm, "testdata/pass.jsonl", "drift", 240) {
		t.Error("README does not state that the golden Pass log test fails on drift")
	}
}
