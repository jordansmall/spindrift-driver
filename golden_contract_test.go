package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"testing"
)

// The tests in this file read testdata/pass.jsonl as the file a downstream host
// reads: os.ReadFile plus plain encoding/json, no driverkit import and no shared
// type (docs/adr/0004-events-are-the-only-stdout-contract.md, and the hand-off
// wording docs_test.go pins). They read the committed bytes rather than a stream
// produced in-process because they are the field-level promises the byte compare
// cannot state: TestGoldenPassLog says the artifact did not move, this file says
// what each field owes a decoder that never saw this repository's Go.

// goldenContractFail is the one shape every failure in this file takes: kind, the
// 1-based line index into the artifact, the field at fault, and how to rewrite it.
// A reader of a failure needs all four — without the line index they cannot find
// the offending event, and without the command they cannot fix it.
func goldenContractFail(t *testing.T, kind string, line int, field, format string, args ...any) {
	t.Helper()
	if kind == "" {
		kind = "(none)"
	}
	t.Errorf("kind %q at line %d, field %q: %s\nregenerate the artifact with: %s", kind, line, field, fmt.Sprintf(format, args...), goldenRegenerateCmd)
}

// goldenReadPass loads the committed artifact and decodes it with the suite's own
// JSONL reader, so these tests hold the same framing promise a run's stdout is
// held to: one JSON object per line, no blank lines, one trailing newline.
func goldenReadPass(t *testing.T) (string, []map[string]any) {
	t.Helper()
	raw, err := os.ReadFile(goldenPassPath)
	if err != nil {
		t.Fatalf("read %s: %v\nregenerate it with: %s", goldenPassPath, err, goldenRegenerateCmd)
	}
	return string(raw), stdoutEvents(t, string(raw))
}

// goldenKindOf reports a line's kind as the string the contract demands. An empty
// return means the line has no usable kind; the caller has already failed, and the
// empty string keeps the follow-up messages readable rather than repeating the
// type error on every field.
func goldenKindOf(t *testing.T, ev map[string]any, line int) string {
	t.Helper()
	raw, present := ev["kind"]
	kind, ok := raw.(string)
	if !ok || kind == "" {
		if !present {
			goldenContractFail(t, "", line, "kind", "missing: every event names its kind, and kind is one of the eight ADR-0004 fixes")
		} else {
			goldenContractFail(t, "", line, "kind", "= %s, want a non-empty string kind", goldenJSONValue(raw))
		}
		return ""
	}
	return kind
}

// The 13-digit window an epoch-millisecond instant falls in: the lower bound is
// 2001-09-09, so a value below it is a seconds-epoch timestamp (10 digits) or a
// counter that was never an instant, and the upper bound is year 2286. Pinning the
// unit rather than only "a number" is what catches a Driver that starts stamping
// seconds — the failure mode a host cannot distinguish from an old run.
const (
	goldenEpochMinMS int64 = 1_000_000_000_000
	goldenEpochMaxMS int64 = 9_999_999_999_999
)

// goldenFieldEpochMS reads a field that must name an instant in epoch
// milliseconds. encoding/json decodes every JSON number to float64, so the
// integral test is on the value rather than the token: a stamped fraction means a
// float clock reached the wire, and a quoted instant means a formatted time did.
func goldenFieldEpochMS(t *testing.T, ev map[string]any, kind, field string, line int) (int64, bool) {
	t.Helper()
	raw, present := ev[field]
	if !present {
		goldenContractFail(t, kind, line, field, "missing: a %s event states its instant as an epoch-millisecond number", kind)
		return 0, false
	}
	v, ok := raw.(float64)
	if !ok {
		goldenContractFail(t, kind, line, field, "= %s, want a bare JSON number, not a %T: an instant is epoch milliseconds, never a quoted or formatted time", goldenJSONValue(raw), raw)
		return 0, false
	}
	if v != math.Trunc(v) {
		goldenContractFail(t, kind, line, field, "= %v, want integral epoch milliseconds: a fractional stamp means a microsecond or float clock reached the wire", v)
		return 0, false
	}
	ms := int64(v)
	if ms < goldenEpochMinMS || ms > goldenEpochMaxMS {
		goldenContractFail(t, kind, line, field, "= %d, want a 13-digit epoch-millisecond instant between %d and %d: a 10-digit value is seconds, not milliseconds", ms, goldenEpochMinMS, goldenEpochMaxMS)
		return 0, false
	}
	return ms, true
}

// goldenJSONValue renders a field's value for a failure message, compactly and
// without the HTML escaping that would mangle a path or a quote in the report.
func goldenJSONValue(v any) string {
	var b strings.Builder
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return fmt.Sprintf("%v", v)
	}
	return strings.TrimSuffix(b.String(), "\n")
}

// TestGoldenPassLinesAreTypedEvents pins the envelope every line of the artifact
// shares: one JSON object (stdoutEvents fails the run otherwise), a non-empty
// string kind, and a timestamp that is an integral epoch-millisecond number.
func TestGoldenPassLinesAreTypedEvents(t *testing.T) {
	_, events := goldenReadPass(t)
	if len(events) == 0 {
		t.Fatalf("%s holds no events\nregenerate it with: %s", goldenPassPath, goldenRegenerateCmd)
	}
	for i, ev := range events {
		line := i + 1
		kind := goldenKindOf(t, ev, line)
		goldenFieldEpochMS(t, ev, kind, "timestamp", line)
	}
}

// goldenEmittedKinds is the closed set docs/adr/0004-events-are-the-only-stdout-contract.md:5
// fixes: "This Driver emits start, reasoning, text, usage, tool_call, tool_result,
// result and error, and nothing else". A kind outside it is a wire change; a kind
// missing from the artifact leaves the host's decoder that branch untested.
var goldenEmittedKinds = []string{"start", "reasoning", "text", "usage", "tool_call", "tool_result", "result", "error"}

// TestGoldenPassCoversAllEightKinds reads the artifact as the conformance corpus
// it exists to be: one file a host can point a decoder at to exercise every branch
// of its event switch, and no branch this Driver does not have.
func TestGoldenPassCoversAllEightKinds(t *testing.T) {
	_, events := goldenReadPass(t)

	seen := map[string]int{}
	for i, ev := range events {
		kind := goldenKindOf(t, ev, i+1)
		if kind == "" {
			continue
		}
		seen[kind]++
		if !slices.Contains(goldenEmittedKinds, kind) {
			goldenContractFail(t, kind, i+1, "kind", "is none of the eight kinds ADR-0004 fixes (%s): the Driver emits nothing else, so a host's decoder has no branch for this", strings.Join(goldenEmittedKinds, ", "))
		}
	}

	var missing []string
	for _, kind := range goldenEmittedKinds {
		if seen[kind] == 0 {
			missing = append(missing, kind)
		}
	}
	if len(missing) > 0 {
		t.Errorf("%s carries %d of the eight kinds but never kind %q: every kind the Driver writes needs at least one line for a host's decoder to parse\nregenerate the artifact with: %s",
			goldenPassPath, len(goldenEmittedKinds)-len(missing), strings.Join(missing, ", "), goldenRegenerateCmd)
	}
}

// goldenMessageIDRE is the shape main.go stamps: fmt.Sprintf("msg-%d", call), the
// 1-based Model call index of whichever process wrote the line — a relayed child
// restarts at msg-1 because it is a separate Pass.
var goldenMessageIDRE = regexp.MustCompile(`^msg-[0-9]+$`)

// goldenFieldMessageID asserts a kind carries the message_id that ties its lines to
// one Model call.
func goldenFieldMessageID(t *testing.T, ev map[string]any, kind string, line int) string {
	t.Helper()
	raw, present := ev["message_id"]
	if !present {
		goldenContractFail(t, kind, line, "message_id", "missing: every %s event states the Model call it belongs to", kind)
		return ""
	}
	id, ok := raw.(string)
	if !ok || id == "" {
		goldenContractFail(t, kind, line, "message_id", "= %s, want a non-empty string", goldenJSONValue(raw))
		return ""
	}
	return id
}

// goldenFieldText asserts a kind carries prose a host can read: a string with
// something in it.
func goldenFieldText(t *testing.T, ev map[string]any, kind string, line int) string {
	t.Helper()
	raw, present := ev["text"]
	if !present {
		goldenContractFail(t, kind, line, "text", "missing: a %s event has to state its content", kind)
		return ""
	}
	text, ok := raw.(string)
	if !ok {
		goldenContractFail(t, kind, line, "text", "= %s, want a string, not a %T", goldenJSONValue(raw), raw)
		return ""
	}
	if text == "" {
		goldenContractFail(t, kind, line, "text", "is empty: an empty %s line carries nothing for a decoder to surface", kind)
	}
	return text
}

// goldenTokenFields are the six totals a usage or result event states, spelled as
// tokenTotals declares them. Presence of all six is the whole point: a decoder has
// to be able to tell "zero" from "this server never reported it" only if the wire
// says zero rather than omitting the key.
var goldenTokenFields = []string{"input", "output", "cache_read", "cache_write_5m", "cache_write_1h", "cost_usd"}

// goldenFieldTokenTotals asserts tokens is an object stating exactly the six fields,
// each a JSON number.
func goldenFieldTokenTotals(t *testing.T, ev map[string]any, kind string, line int) {
	t.Helper()
	raw, present := ev["tokens"]
	if !present {
		goldenContractFail(t, kind, line, "tokens", "missing: a %s event reports its tally", kind)
		return
	}
	totals, ok := raw.(map[string]any)
	if !ok {
		goldenContractFail(t, kind, line, "tokens", "= %s, want a JSON object of token fields, not a %T", goldenJSONValue(raw), raw)
		return
	}
	for _, name := range goldenTokenFields {
		v, ok := totals[name]
		if !ok {
			goldenContractFail(t, kind, line, "tokens."+name, "missing: a %s event states all six token fields, so a missing key can never be read as an unreported one", kind)
			continue
		}
		if _, ok := v.(float64); !ok {
			goldenContractFail(t, kind, line, "tokens."+name, "= %s, want a JSON number, not a %T", goldenJSONValue(v), v)
		}
	}
	var extra []string
	for name := range totals {
		if !slices.Contains(goldenTokenFields, name) {
			extra = append(extra, name)
		}
	}
	if len(extra) > 0 {
		sort.Strings(extra)
		goldenContractFail(t, kind, line, "tokens", "states %s beyond the six fields %s", strings.Join(extra, ", "), strings.Join(goldenTokenFields, ", "))
	}
}

// goldenKindContract is one kind's field contract as one row of the table below, so
// the whole non-Tool contract reads at a glance in the order the kinds appear in a
// Pass: what a line of that kind must state, and what must stay off it.
type goldenKindContract struct {
	kind  string
	check func(t *testing.T, ev map[string]any, line int)
}

// goldenNonToolContracts is that table. The Tool kinds are absent on purpose: their
// contract is a tool_call/tool_result pair matching on tool_id across two lines
// rather than fields on one, so they are pinned as pairs (toolPair) instead.
var goldenNonToolContracts = []goldenKindContract{
	{"start", goldenContractStart},
	{"reasoning", goldenContractReasoning},
	{"text", goldenContractText},
	{"usage", goldenContractUsage},
	{"result", goldenContractResult},
	{"error", goldenContractError},
}

// goldenContractStart: a call opening states which call it is and nothing else.
func goldenContractStart(t *testing.T, ev map[string]any, line int) {
	const kind = "start"
	id := goldenFieldMessageID(t, ev, kind, line)
	if id != "" && !goldenMessageIDRE.MatchString(id) {
		goldenContractFail(t, kind, line, "message_id", "= %q, want the msg-<n> form main.go stamps", id)
	}
	// A start opens a Model call: nothing it says has arrived yet, so the fields
	// that carry an answer must be off the wire entirely rather than empty on it.
	if _, present := ev["text"]; present {
		goldenContractFail(t, kind, line, "text", "present as %s: a start announces a call, it does not answer one", goldenJSONValue(ev["text"]))
	}
	for field := range ev {
		if strings.HasPrefix(field, "tool_") {
			goldenContractFail(t, kind, line, field, "present: a start carries no Tool fields, omitempty keeps them off the wire")
		}
	}
}

// goldenContractReasoning: thinking names its Model call and carries thinking.
func goldenContractReasoning(t *testing.T, ev map[string]any, line int) {
	const kind = "reasoning"
	goldenFieldMessageID(t, ev, kind, line)
	goldenFieldText(t, ev, kind, line)
}

// goldenContractText: assistant prose names its Model call and carries prose.
func goldenContractText(t *testing.T, ev map[string]any, line int) {
	const kind = "text"
	goldenFieldMessageID(t, ev, kind, line)
	goldenFieldText(t, ev, kind, line)
}

// goldenContractUsage: a tally names the call, the model that earned it, and all
// six token fields (the set TestEventUsageStatesAllSixTokenFields pins).
func goldenContractUsage(t *testing.T, ev map[string]any, line int) {
	const kind = "usage"
	goldenFieldMessageID(t, ev, kind, line)
	model, ok := ev["model"].(string)
	if !ok || model == "" {
		goldenContractFail(t, kind, line, "model", "= %s, want the non-empty id of the model that answered this call", goldenJSONValue(ev["model"]))
	}
	goldenFieldTokenTotals(t, ev, kind, line)
}

// goldenContractResult: the closing summary states every field it can, including the
// zero ones — the one kind with no omitempty anywhere on its wire type.
func goldenContractResult(t *testing.T, ev map[string]any, line int) {
	const kind = "result"
	// resultPayload carries no omitempty at all: a duration of 0 is a measurement,
	// so every field below has to be on the wire even when it is empty or zero.
	for _, field := range []string{"text", "tokens", "duration_ms", "duration_api_ms", "num_turns", "models"} {
		if _, present := ev[field]; !present {
			goldenContractFail(t, kind, line, field, "missing: the result summary never leaves a field unreported, so omitempty belongs to no result field")
		}
	}
	if _, present := ev["text"]; present {
		goldenFieldText(t, ev, kind, line)
	}
	if _, present := ev["tokens"]; present {
		goldenFieldTokenTotals(t, ev, kind, line)
	}
	for _, field := range []string{"duration_ms", "duration_api_ms", "num_turns"} {
		raw, present := ev[field]
		if !present {
			continue
		}
		v, ok := raw.(float64)
		if !ok {
			goldenContractFail(t, kind, line, field, "= %s, want a JSON number, not a %T", goldenJSONValue(raw), raw)
			continue
		}
		if v != math.Trunc(v) {
			goldenContractFail(t, kind, line, field, "= %v, want integral milliseconds or turns", v)
			continue
		}
		if v < 0 {
			goldenContractFail(t, kind, line, field, "= %v, want a non-negative count", v)
		}
	}
	rawModels, present := ev["models"]
	if !present {
		return
	}
	models, ok := rawModels.([]any)
	if !ok {
		goldenContractFail(t, kind, line, "models", "= %s, want a JSON array of model ids, not a %T", goldenJSONValue(rawModels), rawModels)
		return
	}
	for i, m := range models {
		id, ok := m.(string)
		if !ok || id == "" {
			goldenContractFail(t, kind, line, fmt.Sprintf("models[%d]", i), "= %s, want a non-empty model id", goldenJSONValue(m))
		}
	}
}

// goldenContractError: a failure says what happened, and says when the server will
// listen again only if a reset was named.
func goldenContractError(t *testing.T, ev map[string]any, line int) {
	const kind = "error"
	goldenFieldText(t, ev, kind, line)
	// reset_at is the one optional instant: omitempty keeps it off every line whose
	// rejection named no reset, and where it exists it shares timestamp's unit.
	if _, present := ev["reset_at"]; present {
		goldenFieldEpochMS(t, ev, kind, "reset_at", line)
	}
}

// TestGoldenPassNonToolFieldContract walks the artifact and applies each non-Tool
// kind's contract to every line of that kind, relayed child lines included: the
// child writes the same kinds, so it owes the same fields.
func TestGoldenPassNonToolFieldContract(t *testing.T) {
	raw, events := goldenReadPass(t)
	contractOf := map[string]func(t *testing.T, ev map[string]any, line int){}
	for _, c := range goldenNonToolContracts {
		contractOf[c.kind] = c.check
	}
	checked := map[string]int{}
	for i, ev := range events {
		kind := goldenKindOf(t, ev, i+1)
		check, ok := contractOf[kind]
		if !ok {
			continue
		}
		checked[kind]++
		check(t, ev, i+1)
	}
	var missing []string
	for _, c := range goldenNonToolContracts {
		if checked[c.kind] == 0 {
			missing = append(missing, c.kind)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		t.Errorf("%s carries no line of kind %q, so that contract has nothing to check — field \"kind\" has to appear at least once\nregenerate the artifact with: %s",
			goldenPassPath, strings.Join(missing, ", "), goldenRegenerateCmd)
	}

	// The table above applies to the child's result line too, so the parent's own
	// closer is pinned separately: parentResult is the only safe reader on a relaying
	// log (the child's result shares the kind and in_subagent is what separates them)
	// and it fails unless exactly one parent result exists and it is the last line.
	var parentResults int
	for _, ev := range events {
		if ev["kind"] == "result" && ev["in_subagent"] != true {
			parentResults++
		}
	}
	if parentResults == 0 {
		goldenContractFail(t, "result", len(events), "kind", "the artifact never carries a parent result line: a Pass closes with its own summary, and a relayed child's does not stand in for it")
	} else {
		parentResult(t, raw)
	}
}

// TestGoldenPassRelayAttribution pins the two fields the relay adds and nothing
// else: in_subagent is a real boolean true, role names the roster entry, and no
// parent line carries either — a host routes on those keys, so a string "true" or a
// stray role on the parent's own line would misfile the whole stream.
func TestGoldenPassRelayAttribution(t *testing.T) {
	_, events := goldenReadPass(t)
	var relayed int
	for i, ev := range events {
		line := i + 1
		inSubagent, present := ev["in_subagent"]
		if !present {
			if _, hasRole := ev["role"]; hasRole {
				goldenContractFail(t, goldenKindOf(t, ev, line), line, "role", "= %s on a parent line: role is attribution for a relayed child and belongs to no line the parent wrote", goldenJSONValue(ev["role"]))
			}
			continue
		}
		relayed++
		if inSubagent != true {
			goldenContractFail(t, goldenKindOf(t, ev, line), line, "in_subagent", "= %s, want the JSON boolean true: a host branches on truthiness, and the string \"true\" is one", goldenJSONValue(inSubagent))
		}
		role, ok := ev["role"].(string)
		if !ok || role == "" {
			goldenContractFail(t, goldenKindOf(t, ev, line), line, "role", "= %s, want the non-empty roster entry name the child ran as", goldenJSONValue(ev["role"]))
		}
	}
	if relayed == 0 {
		t.Errorf("%s holds no relayed child line: a conformance log has to show in_subagent and role on a line a child produced\nregenerate the artifact with: %s", goldenPassPath, goldenRegenerateCmd)
	}
}

// goldenToolLine is one Tool line of the artifact held together with its place in
// the file. Tool promises span two lines, so the index travels with each half:
// goldenContractFail without it names a field a reader cannot go and look at.
type goldenToolLine struct {
	kind string
	line int
	ev   map[string]any
}

// goldenToolLines returns the artifact's tool_call and tool_result lines in file
// order, relayed child lines included: a child that calls a Tool writes these same
// two kinds onto the parent's stdout and owes the same pairing.
func goldenToolLines(t *testing.T, events []map[string]any) []goldenToolLine {
	t.Helper()
	var out []goldenToolLine
	for i, ev := range events {
		kind := goldenKindOf(t, ev, i+1)
		if kind == "tool_call" || kind == "tool_result" {
			out = append(out, goldenToolLine{kind: kind, line: i + 1, ev: ev})
		}
	}
	return out
}

// goldenFieldToolString reads one of the three string fields both Tool kinds have to
// state — tool_id, tool_name, tool_target — as the non-empty string the contract
// demands of it.
func goldenFieldToolString(t *testing.T, ev map[string]any, kind, field string, line int) string {
	t.Helper()
	raw, present := ev[field]
	if !present {
		goldenContractFail(t, kind, line, field, "missing: every %s event states its %s", kind, field)
		return ""
	}
	v, ok := raw.(string)
	if !ok || v == "" {
		goldenContractFail(t, kind, line, field, "= %s, want a non-empty string", goldenJSONValue(raw))
		return ""
	}
	return v
}

// goldenFieldToolInput reads a tool_call's arguments as the JSON object they have to
// reach the wire as. decodeToolArguments parses the Model's arguments before the
// event is emitted, so an object on the wire is the proof that parse happened: a
// string is raw JSON re-escaped, which a host would have to decode a second time to
// read at all.
func goldenFieldToolInput(t *testing.T, ev map[string]any, kind string, line int) map[string]any {
	t.Helper()
	raw, present := ev["tool_input"]
	if !present {
		goldenContractFail(t, kind, line, "tool_input", "missing: a %s event states the arguments it was called with", kind)
		return nil
	}
	if _, isString := raw.(string); isString {
		goldenContractFail(t, kind, line, "tool_input", "= %s: an escaped string of the arguments rather than the arguments. tool_input is a JSON value, so a decoder reads it straight off the wire without a second parse", goldenJSONValue(raw))
		return nil
	}
	obj, ok := raw.(map[string]any)
	if !ok {
		goldenContractFail(t, kind, line, "tool_input", "= %s, want a JSON object, not a %T", goldenJSONValue(raw), raw)
		return nil
	}
	return obj
}

// goldenToolInputField reads one string field of a call's arguments, for the
// per-Tool tool_target rules, each of which states the target in terms of the
// argument that Tool takes.
func goldenToolInputField(t *testing.T, input map[string]any, l goldenToolLine, tool, field string) string {
	t.Helper()
	if input == nil {
		return "" // goldenFieldToolInput already failed this line's tool_input
	}
	raw, present := input[field]
	if !present {
		goldenContractFail(t, l.kind, l.line, "tool_input."+field, "missing: the %s Tool states its tool_target in terms of it", tool)
		return ""
	}
	v, ok := raw.(string)
	if !ok || v == "" {
		goldenContractFail(t, l.kind, l.line, "tool_input."+field, "= %s, want the non-empty string the %s Tool reports as its target", goldenJSONValue(raw), tool)
		return ""
	}
	return v
}

// goldenFieldToolOutput reads a Tool Result's output, which the Driver carries
// verbatim: command output, one of the file Tools' sentences, a child's answer, or a
// refusal — and never a truncated or reformatted version of any of them.
func goldenFieldToolOutput(t *testing.T, ev map[string]any, kind string, line int) string {
	t.Helper()
	raw, present := ev["tool_output"]
	if !present {
		goldenContractFail(t, kind, line, "tool_output", "missing: a %s event reports what its Tool produced", kind)
		return ""
	}
	text, ok := raw.(string)
	if !ok {
		goldenContractFail(t, kind, line, "tool_output", "= %s, want a string, not a %T", goldenJSONValue(raw), raw)
		return ""
	}
	if text == "" {
		goldenContractFail(t, kind, line, "tool_output", "is empty: a Tool that ran reports something, and one that was refused reports the refusal")
	}
	return text
}

// goldenToolPair is one tool_call and the tool_result that answers it, joined on
// tool_id rather than on position: a relaying log puts a spawned child's whole stream
// between the parent's Subagent pair by design (subagent_test.go's spawnPair says the
// same of its own scan), so adjacency is not a property a pair may be read by.
type goldenToolPair struct {
	call   goldenToolLine
	result goldenToolLine
}

// goldenToolPairs pairs every Tool line of the artifact and stops the test on any
// line that cannot be paired. TestGoldenPassToolPairsClose asserts pairing itself;
// the tests below state promises about a pair — a target equal on both halves, a
// refusal that answers its own call — and take their pairs here rather than re-deriving
// them and reporting a field mismatch on a line that never had a partner.
func goldenToolPairs(t *testing.T, events []map[string]any) []goldenToolPair {
	t.Helper()
	lines := goldenToolLines(t, events)
	calls := map[string]goldenToolLine{}
	answers := map[string]goldenToolLine{}
	var order []string
	for _, l := range lines {
		id := goldenFieldToolString(t, l.ev, l.kind, "tool_id", l.line)
		if id == "" {
			continue
		}
		by := calls
		if l.kind == "tool_result" {
			by = answers
		}
		if prev, dup := by[id]; dup {
			t.Fatalf("%s at line %d repeats tool_id %q, first seen at line %d: one id is one Tool run\nregenerate the artifact with: %s", l.kind, l.line, id, prev.line, goldenRegenerateCmd)
		}
		by[id] = l
		if l.kind == "tool_call" {
			order = append(order, id)
		}
	}
	pairs := make([]goldenToolPair, 0, len(order))
	for _, id := range order {
		call := calls[id]
		result, ok := answers[id]
		if !ok {
			t.Fatalf("the tool_call at line %d is never answered: tool_id %q has no tool_result\nregenerate the artifact with: %s", call.line, id, goldenRegenerateCmd)
		}
		delete(answers, id)
		pairs = append(pairs, goldenToolPair{call: call, result: result})
	}
	if len(answers) > 0 {
		var orphans []string
		for id, result := range answers {
			orphans = append(orphans, fmt.Sprintf("line %d (tool_id %q)", result.line, id))
		}
		sort.Strings(orphans)
		t.Fatalf("%s carries tool_result line(s) %s no tool_call answers\nregenerate the artifact with: %s", goldenPassPath, strings.Join(orphans, ", "), goldenRegenerateCmd)
	}
	return pairs
}

// goldenExitCodeMarker is the line a command's result always closes with — the one
// main.go's toolResultIsError reads to decide is_error — and so the marker that tells
// a Tool that ran from one that never did.
const goldenExitCodeMarker = "exit code:"

// goldenBreakerSaid* are the two fragments of main.go's repeatedCallRefusal wording
// that no Tool's own output can produce: the count of identical sends and the demand
// for a different action.
const (
	goldenBreakerSaidRan    = "was already called"
	goldenBreakerSaidChange = "take a different action"
)

// goldenIsBreakerRefusal reports whether a tool_output is the Repeated-Call Breaker's
// synthetic result rather than anything a Tool emitted.
func goldenIsBreakerRefusal(output string) bool {
	return strings.Contains(output, goldenBreakerSaidRan) && strings.Contains(output, goldenBreakerSaidChange)
}

// TestGoldenPassToolPairsClose states what a closed pair owes beyond the pairing
// goldenToolPairs establishes: the result never lands ahead of the call that ran it,
// and the two halves agree on the Tool that ran and the Model call they belong to.
// Matching stays by tool_id and never by adjacency — on this artifact the reviewer's
// whole relayed block sits inside one parent pair, so a scan that expects the two lines
// to touch would call the artifact broken, and one that expects the file to alternate
// would fold two open pairs into one.
func TestGoldenPassToolPairsClose(t *testing.T) {
	_, events := goldenReadPass(t)
	pairs := goldenToolPairs(t, events)

	if len(pairs) == 0 {
		t.Errorf("%s holds no tool_call/tool_result pair at all: every other Tool promise is a promise about a pair\nregenerate the artifact with: %s", goldenPassPath, goldenRegenerateCmd)
	}

	for _, p := range pairs {
		// goldenToolPairs joins from maps, so it never compares which half came first.
		if p.call.line > p.result.line {
			goldenContractFail(t, p.result.kind, p.result.line, "tool_id", "answers the tool_call at line %d from ahead of it: a Tool's result cannot reach the wire before the call that ran it", p.call.line)
		}
		// Each side is read through the field reader, and a blank is not compared: a
		// missing tool_name or message_id is that line's failure once, not also a
		// mismatch here.
		callName := goldenFieldToolString(t, p.call.ev, p.call.kind, "tool_name", p.call.line)
		resultName := goldenFieldToolString(t, p.result.ev, p.result.kind, "tool_name", p.result.line)
		if callName != "" && resultName != "" && callName != resultName {
			goldenContractFail(t, p.result.kind, p.result.line, "tool_name", "= %q while the tool_call it answers at line %d names %q: the two halves report one Tool run, so they name one Tool", resultName, p.call.line, callName)
		}
		callMsg := goldenFieldMessageID(t, p.call.ev, p.call.kind, p.call.line)
		resultMsg := goldenFieldMessageID(t, p.result.ev, p.result.kind, p.result.line)
		if callMsg != "" && resultMsg != "" && callMsg != resultMsg {
			goldenContractFail(t, p.result.kind, p.result.line, "message_id", "= %q while the tool_call it answers at line %d belongs to %q: a call and its result are one turn of one Model call", resultMsg, p.call.line, callMsg)
		}
	}
}

// goldenToolNamesAreTheDriversTools are the four Tools main.go's planToolCall
// dispatches to — the names a tool_call's tool_name can carry and each one's
// tool_target means something different, so a log that never exercises one leaves
// that rule unpinned.
var goldenToolNamesAreTheDriversTools = []string{"bash", "edit", "write", "subagent"}

// TestGoldenPassToolTargetNamesEachTool pins tool_target per Tool: the command string
// verbatim for bash, an absolute path for the two file Tools, the roster entry name for
// the Subagent Tool — and the same value on both halves of a pair, because plan.target
// is decided when the call is planned and toolResultEvent copies that one value into
// the result. The file Tools' absoluteness is what the artifact's own normalisation
// depends on: a relative target would name a different file on every machine that
// regenerated the log.
func TestGoldenPassToolTargetNamesEachTool(t *testing.T) {
	_, events := goldenReadPass(t)
	pairs := goldenToolPairs(t, events)

	called, answered := map[string]int{}, map[string]int{}
	for _, p := range pairs {
		name := goldenFieldToolString(t, p.call.ev, p.call.kind, "tool_name", p.call.line)
		callTarget := goldenFieldToolString(t, p.call.ev, p.call.kind, "tool_target", p.call.line)
		resultTarget := goldenFieldToolString(t, p.result.ev, p.result.kind, "tool_target", p.result.line)
		called[name]++
		answered[name]++

		switch name {
		case "bash":
			command := goldenToolInputField(t, goldenFieldToolInput(t, p.call.ev, p.call.kind, p.call.line), p.call, name, "command")
			if command == "" {
				continue
			}
			if callTarget != command {
				goldenContractFail(t, p.call.kind, p.call.line, "tool_target", "= %q, want the command string verbatim (%q): the bash Tool has no path or entry to name, so its target is what it ran", callTarget, command)
			}
			if resultTarget != command {
				goldenContractFail(t, p.result.kind, p.result.line, "tool_target", "= %q, want the same command string its tool_call reported at line %d (%q)", resultTarget, p.call.line, command)
			}
		case "edit", "write":
			for _, half := range []goldenToolLine{p.call, p.result} {
				target := callTarget
				if half.kind == "tool_result" {
					target = resultTarget
				}
				if target == "" {
					continue
				}
				if !strings.HasPrefix(target, "/") || !filepath.IsAbs(target) {
					goldenContractFail(t, half.kind, half.line, "tool_target", "= %q: the %s Tool resolves its path through resolveFilePath, which answers filepath.Abs — a relative target means the Model's raw argument reached the wire, and then this artifact names a file that differs per run directory and per machine", target, name)
				}
			}
			if callTarget != "" && resultTarget != "" && callTarget != resultTarget {
				goldenContractFail(t, p.result.kind, p.result.line, "tool_target", "= %q while its tool_call at line %d reports %q: one pair touches one file, so one value names it on both halves", resultTarget, p.call.line, callTarget)
			}
		case "subagent":
			entry := goldenToolInputField(t, goldenFieldToolInput(t, p.call.ev, p.call.kind, p.call.line), p.call, name, "agent")
			if entry == "" {
				continue
			}
			if callTarget != entry {
				goldenContractFail(t, p.call.kind, p.call.line, "tool_target", "= %q, want the roster entry name the call delegated to (%q), which is what planSubagentCall puts there", callTarget, entry)
			}
			if resultTarget != entry {
				goldenContractFail(t, p.result.kind, p.result.line, "tool_target", "= %q, want the roster entry name its tool_call named at line %d (%q)", resultTarget, p.call.line, entry)
			}
		}
	}

	var missing []string
	for _, name := range goldenToolNamesAreTheDriversTools {
		if called[name] == 0 {
			missing = append(missing, name+" (no tool_call)")
		}
		if answered[name] == 0 {
			missing = append(missing, name+" (no tool_result)")
		}
	}
	if len(missing) > 0 {
		t.Errorf("%s never exercises %s: the Driver has four Tools — %s — each with its own tool_target rule, so a log that skips one leaves that rule for a host to guess\nregenerate the artifact with: %s",
			goldenPassPath, strings.Join(missing, ", "), strings.Join(goldenToolNamesAreTheDriversTools, ", "), goldenRegenerateCmd)
	}
}

// TestGoldenPassToolInputIsAJSONObject pins that a tool_call's arguments are a JSON
// value on the wire and never an escaped string of one, and that arguments belong to
// tool_call alone — omitempty keeps them off every kind that takes none.
func TestGoldenPassToolInputIsAJSONObject(t *testing.T) {
	_, events := goldenReadPass(t)
	var calls int
	for i, ev := range events {
		line := i + 1
		kind := goldenKindOf(t, ev, line)
		if kind != "tool_call" {
			if raw, present := ev["tool_input"]; present {
				goldenContractFail(t, kind, line, "tool_input", "present on a %s line as %s: arguments belong to the call that made them", kind, goldenJSONValue(raw))
			}
			continue
		}
		calls++
		if input := goldenFieldToolInput(t, ev, kind, line); input != nil && len(input) == 0 {
			goldenContractFail(t, kind, line, "tool_input", "is the empty object: every Tool this Driver has takes at least one argument, so {} means they were dropped between the Model and the wire")
		}
	}
	if calls == 0 {
		t.Errorf("%s carries no tool_call line, so this contract has nothing to check\nregenerate the artifact with: %s", goldenPassPath, goldenRegenerateCmd)
	}
}

// TestGoldenPassIsErrorIsAPromiseNotAMaybe pins is_error as a flag that only ever
// says true. events.go declares it as a Go bool with omitempty, so false cannot reach
// the wire at all: an explicit false, or a non-boolean a host would read by truthiness
// (the string "false" being truthy in the decoders this artifact exists for), means the
// field stopped meaning "this result failed".
func TestGoldenPassIsErrorIsAPromiseNotAMaybe(t *testing.T) {
	_, events := goldenReadPass(t)
	var failed, clean int
	for i, ev := range events {
		line := i + 1
		kind := goldenKindOf(t, ev, line)
		raw, present := ev["is_error"]
		if kind != "tool_result" {
			if present {
				goldenContractFail(t, kind, line, "is_error", "= %s on a %s line: only a Tool Result can fail, so nothing else may state failure", goldenJSONValue(raw), kind)
			}
			continue
		}
		if !present {
			clean++
			continue
		}
		failed++
		switch v := raw.(type) {
		case bool:
			if !v {
				goldenContractFail(t, kind, line, "is_error", "= false: omitempty keeps a false bool off the wire entirely, so an explicit false is a result claiming it failed while the flag says it did not — a clean result reports nothing here")
			}
		default:
			goldenContractFail(t, kind, line, "is_error", "= %s, want the JSON boolean true, not a %T: a host branches on truthiness, and %s is one", goldenJSONValue(raw), raw, goldenJSONValue(raw))
		}
	}
	if failed == 0 || clean == 0 {
		t.Errorf("%s carries %d is_error and %d clean tool_result line(s), want at least one of each: a log showing only one shape never exercises the branch that separates them\nregenerate the artifact with: %s", goldenPassPath, failed, clean, goldenRegenerateCmd)
	}
}

// TestGoldenPassSubagentFieldRidesOnlySubagentResults pins the field a host routes a
// delegation's answer on: present and naming the roster entry on the Subagent Tool's
// own results, absent from every other tool_result, from every tool_call and from every
// non-Tool kind — because its presence is what says "this line reports a delegation",
// and a stray one misfiles a plain command or a piece of prose as one.
func TestGoldenPassSubagentFieldRidesOnlySubagentResults(t *testing.T) {
	_, events := goldenReadPass(t)
	var entries []string
	for i, ev := range events {
		line := i + 1
		kind := goldenKindOf(t, ev, line)
		raw, present := ev["subagent"]
		if kind == "tool_result" && ev["tool_name"] == "subagent" {
			if !present {
				goldenContractFail(t, kind, line, "subagent", "missing: the Subagent Tool's result is the one line that names the roster entry the task ran as, and that name is how a host ties the answer back to the delegation")
				continue
			}
			entry, ok := raw.(string)
			if !ok || entry == "" {
				goldenContractFail(t, kind, line, "subagent", "= %s, want the non-empty roster entry name", goldenJSONValue(raw))
				continue
			}
			entries = append(entries, entry)
			// planSubagentCall gives plan.target and plan.subagent the same entry name,
			// so the two spellings of "who ran this" cannot disagree on one line.
			if target, _ := ev["tool_target"].(string); target != "" && target != entry {
				goldenContractFail(t, kind, line, "subagent", "= %q while tool_target on the same line is %q: both name the roster entry, so they cannot differ", entry, target)
			}
			continue
		}
		if present {
			goldenContractFail(t, kind, line, "subagent", "= %s on a line that delegates nothing: presence is what marks a delegation's answer, and this line is not one", goldenJSONValue(raw))
		}
	}
	if len(entries) == 0 {
		t.Errorf("%s carries no Subagent Tool Result naming its entry, so this contract has nothing to check\nregenerate the artifact with: %s", goldenPassPath, goldenRegenerateCmd)
	}
}

// TestGoldenPassSyntheticToolResultIsIdentifiable pins the one Tool Result no Tool
// produced: the Repeated-Call Breaker's refusal, handed out in place of running the
// command. driverkit's synthetic flag is not in this Driver's declared wire vocabulary
// — docs/adr/0004-events-are-the-only-stdout-contract.md:5 fixes the eight kinds and
// events.go declares no such field — so the shape below is what the log pins in its
// place: the breaker's own wording, no exit-code line where every command result
// carries one, and an ordinary pair around it.
func TestGoldenPassSyntheticToolResultIsIdentifiable(t *testing.T) {
	_, events := goldenReadPass(t)
	pairs := goldenToolPairs(t, events)

	var refusals []goldenToolPair
	for _, p := range pairs {
		if goldenIsBreakerRefusal(goldenFieldToolOutput(t, p.result.ev, p.result.kind, p.result.line)) {
			refusals = append(refusals, p)
		}
	}
	if len(refusals) != 1 {
		t.Errorf("%s holds %d tool_result line(s) reading as the Repeated-Call Breaker's refusal, want exactly one: it is the one result no Tool produced, so a decoder picks it out of the log by shape and needs that shape to be unique\nregenerate the artifact with: %s", goldenPassPath, len(refusals), goldenRegenerateCmd)
	}
	for _, p := range refusals {
		output := goldenFieldToolOutput(t, p.result.ev, p.result.kind, p.result.line)
		if _, present := p.result.ev["is_error"]; !present {
			goldenContractFail(t, p.result.kind, p.result.line, "is_error", "missing on the breaker's refusal: a call that was refused did not happen, and reporting its result as clean tells the Model the command ran")
		}
		if strings.Contains(output, goldenExitCodeMarker) {
			goldenContractFail(t, p.result.kind, p.result.line, "tool_output", "carries an %q line: the refused command never reached bash, so there is no exit status to report — its absence is half of what makes this result recognisably synthetic", goldenExitCodeMarker)
		}
		// The pair around the refusal is an ordinary one: the breaker answers the call
		// the Model made, under the id, Tool and Model call that made it, so a host
		// waiting on the pair resumes exactly as it would for a result a Tool produced.
		if callName, _ := p.call.ev["tool_name"].(string); callName != "" && p.result.ev["tool_name"] != callName {
			goldenContractFail(t, p.result.kind, p.result.line, "tool_name", "= %v while the call it refuses at line %d names %q", p.result.ev["tool_name"], p.call.line, callName)
		}
		if callMsg, _ := p.call.ev["message_id"].(string); callMsg != "" && p.result.ev["message_id"] != callMsg {
			goldenContractFail(t, p.result.kind, p.result.line, "message_id", "= %v while the call it refuses at line %d belongs to %q", p.result.ev["message_id"], p.call.line, callMsg)
		}
		if callTarget, _ := p.call.ev["tool_target"].(string); callTarget != "" && p.result.ev["tool_target"] != callTarget {
			goldenContractFail(t, p.result.kind, p.result.line, "tool_target", "= %v while the call it refuses at line %d targets %q: a refusal is still a result for that call", p.result.ev["tool_target"], p.call.line, callTarget)
		}
	}
}

// The three failure shapes this artifact keeps apart, and the one clean exit status.
// The two real failures are pinned to 3 rather than to "non-zero" so a run that
// started failing differently cannot quietly reclassify itself.
const (
	goldenExitCodeClean = "exit code: 0"
	goldenExitCode3     = "exit code: 3"
	// goldenFailedSpawnEntry is the roster entry whose spawn the scripted Pass drowns
	// in a 429; the marker below is the prefix subagent.go gives a child that died
	// before answering.
	goldenFailedSpawnEntry = "scout"
)

// TestGoldenPassFailureShapesStayDistinguishable keeps the three ways a Tool Result
// reports failure readable apart: the two commands that really ran and exited 3, the
// breaker's refusal of the third send, and the spawn whose child never answered. Only
// the first carries an exit-code line, only the last carries subagent, and the failure
// count closes on the three of them — so a fourth failure shape, or one of these three
// starting to look like another, cannot pass unnoticed.
func TestGoldenPassFailureShapesStayDistinguishable(t *testing.T) {
	_, events := goldenReadPass(t)
	lines := goldenToolLines(t, events)
	failedSpawnMarker := fmt.Sprintf("error: subagent %q failed:", goldenFailedSpawnEntry)

	var exitCodeRun, exitCode3, refused, spawnFailed, unclassified, failed, clean int
	for _, l := range lines {
		if l.kind != "tool_result" {
			continue
		}
		output := goldenFieldToolOutput(t, l.ev, l.kind, l.line)
		if _, isError := l.ev["is_error"]; !isError {
			clean++
			// A clean line that reads like one of the three failures means this
			// classification is no longer the whole story of what failed.
			if strings.Contains(output, goldenExitCodeMarker) && !strings.Contains(output, goldenExitCodeClean) {
				goldenContractFail(t, l.kind, l.line, "is_error", "missing while tool_output reports %q: a command that did not exit 0 is a failure, and a clean result claiming it succeeded is the one reading a host cannot recover from", goldenExitCodeMarker)
			}
			if goldenIsBreakerRefusal(output) || strings.HasPrefix(output, "error:") {
				goldenContractFail(t, l.kind, l.line, "is_error", "missing while tool_output = %s: refusal wording on a result that reports success contradicts itself", goldenJSONValue(output))
			}
			continue
		}
		failed++
		hasExit := strings.Contains(output, goldenExitCodeMarker)
		isRefusal := !hasExit && goldenIsBreakerRefusal(output)
		isSpawn := !hasExit && !isRefusal && strings.Contains(output, failedSpawnMarker)
		switch {
		case hasExit:
			exitCodeRun++
			if strings.Contains(output, goldenExitCode3) {
				exitCode3++
			}
			if name, _ := l.ev["tool_name"].(string); name != "bash" {
				goldenContractFail(t, l.kind, l.line, "tool_name", "= %q on a result carrying an exit code: only the bash Tool runs a command that has one", goldenJSONValue(l.ev["tool_name"]))
			}
		case isRefusal:
			refused++
			if _, present := l.ev["subagent"]; present {
				goldenContractFail(t, l.kind, l.line, "subagent", "= %s on the breaker's refusal: a refusal is not a delegation, and what separates it from the failed spawn in this log is exactly this key", goldenJSONValue(l.ev["subagent"]))
			}
		case isSpawn:
			spawnFailed++
			entry, ok := l.ev["subagent"].(string)
			if !ok || entry == "" {
				goldenContractFail(t, l.kind, l.line, "subagent", "= %s, want the roster entry whose spawn failed: the failed spawn is the failure shape that still names who it asked", goldenJSONValue(l.ev["subagent"]))
			}
		default:
			unclassified++
			goldenContractFail(t, l.kind, l.line, "tool_output", "= %s, an is_error result that reads as none of the three failure shapes this log pins — no exit code, no breaker wording, no failed-spawn prefix", goldenJSONValue(output))
		}
	}

	if exitCodeRun != 2 || exitCode3 != 2 {
		t.Errorf("%s holds %d is_error result(s) carrying an exit code and %d of them %q, want 2 and 2: the scripted Pass runs the failing command twice, and the exit-code failures are those two", goldenPassPath, exitCodeRun, exitCode3, goldenExitCode3)
	}
	if refused != 1 {
		t.Errorf("%s holds %d is_error result(s) reading as the breaker's refusal, want 1: is_error, the refusal wording, and no exit-code line", goldenPassPath, refused)
	}
	if spawnFailed != 1 {
		t.Errorf("%s holds %d is_error result(s) reading as the failed spawn (%s), want 1: the scout's failure and nothing else", goldenPassPath, spawnFailed, goldenJSONValue(failedSpawnMarker))
	}
	if unclassified > 0 {
		t.Errorf("%s holds %d is_error result(s) outside the three pinned failure shapes", goldenPassPath, unclassified)
	}
	if failed != exitCodeRun+refused+spawnFailed+unclassified || clean != 4 {
		t.Errorf("%s holds %d is_error and %d clean tool_result line(s), want 4 of each — two exit-code failures, one refusal, one failed spawn, and four results that succeeded", goldenPassPath, failed, clean)
	}
}
