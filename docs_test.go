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
