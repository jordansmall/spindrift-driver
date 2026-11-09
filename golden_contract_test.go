package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
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

// goldenTokenSum is the log's own arithmetic: one running total per token field, plus
// the addends behind each total spelled out with the line and attribution they came
// from. A decoder that bills a session by adding the usage events it was shown has to
// land on the result event's numbers, and a mismatch that names only those two numbers
// cannot be checked by eye — the addends are the working.
type goldenTokenSum struct {
	total   map[string]float64
	addends map[string][]string
}

// goldenSumUsage adds every usage line of the artifact, the parent's and the relayed
// children's alike. That is the whole point of summing usage rather than result
// totals: foldChildUsage (subagent.go) charges a relayed usage event into the parent
// exactly as the parent charges its own, so the lines that make up the session total
// are all of them and not the parent's share.
func goldenSumUsage(t *testing.T, events []map[string]any) goldenTokenSum {
	t.Helper()
	sum := goldenTokenSum{total: map[string]float64{}, addends: map[string][]string{}}
	for i, ev := range events {
		line := i + 1
		kind := goldenKindOf(t, ev, line)
		if kind != "usage" {
			continue
		}
		totals, ok := ev["tokens"].(map[string]any)
		if !ok {
			goldenContractFail(t, kind, line, "tokens", "= %s, want a JSON object of token fields to add into the session total", goldenJSONValue(ev["tokens"]))
			continue
		}
		label := fmt.Sprintf("L%d", line)
		if ev["in_subagent"] == true {
			role, _ := ev["role"].(string)
			label = fmt.Sprintf("L%d(%s)", line, role)
		}
		for _, name := range goldenTokenFields {
			v, ok := totals[name].(float64)
			if !ok {
				goldenContractFail(t, kind, line, "tokens."+name, "= %s, want a JSON number to add into the session total", goldenJSONValue(totals[name]))
				continue
			}
			sum.total[name] += v
			sum.addends[name] = append(sum.addends[name], fmt.Sprintf("%s:%g", label, v))
		}
	}
	return sum
}

// TestGoldenPassResultTokensSumEveryUsageLine states the promise a host bills on: the
// session total is the sum of all the usage events in the log, relayed children
// included, field by field. It holds because foldChildUsage charges each forwarded
// usage event into the parent's delegated tally and run() closes with
// totals.add(delegated) (main.go) — a relay that forwarded a child's spend without
// charging it, or one that charged the child's result totals on top of its usage
// events, would break the equality below in one direction or the other.
func TestGoldenPassResultTokensSumEveryUsageLine(t *testing.T) {
	raw, events := goldenReadPass(t)
	result := parentResult(t, raw)
	resultLine := len(events) // parentResult pins that the parent's result is the last line

	resultTokens, ok := result["tokens"].(map[string]any)
	if !ok {
		t.Fatalf("the parent's result reports tokens as %s, want an object of the six token fields\nregenerate the artifact with: %s", goldenJSONValue(result["tokens"]), goldenRegenerateCmd)
	}
	sum := goldenSumUsage(t, events)
	if len(sum.addends["input"]) == 0 {
		t.Fatalf("%s holds no usage line to add up, so a session total cannot be reproduced from the stream\nregenerate the artifact with: %s", goldenPassPath, goldenRegenerateCmd)
	}

	for _, name := range goldenTokenFields {
		got, present := resultTokens[name]
		if !present {
			goldenContractFail(t, "result", resultLine, "tokens."+name, "missing while %d usage line(s) report it: a field a decoder can sum must be a field the total states", len(sum.addends[name]))
			continue
		}
		want, ok := got.(float64)
		if !ok {
			goldenContractFail(t, "result", resultLine, "tokens."+name, "= %s, want a JSON number, not a %T", goldenJSONValue(got), got)
			continue
		}
		if sum.total[name] != want {
			goldenContractFail(t, "result", resultLine, "tokens."+name, "= %v, want %v, the sum of every usage line in the file (%s) — the same addition a host does to bill the session, relayed children included",
				want, sum.total[name], strings.Join(sum.addends[name], " + "))
		}
	}
}

// TestGoldenPassResultTokensAreNotParentOnly pins that the equality above is about the
// children at all: on a log with no relayed usage the sum is trivially the parent's
// own share, and the test would pass without having checked a fold. So the relayed
// share has to be there, it has to be non-zero, and it has to be a share of at least
// one field — otherwise dropping every relayed line would leave the total unchanged
// and no fold would be observable.
func TestGoldenPassResultTokensAreNotParentOnly(t *testing.T) {
	raw, events := goldenReadPass(t)
	result := parentResult(t, raw)
	resultLine := len(events)

	var relayedLines []int
	perField := map[string]float64{}
	for i, ev := range events {
		if goldenKindOf(t, ev, i+1) != "usage" || ev["in_subagent"] != true {
			continue
		}
		relayedLines = append(relayedLines, i+1)
		totals, _ := ev["tokens"].(map[string]any)
		for _, name := range goldenTokenFields {
			v, _ := totals[name].(float64)
			perField[name] += v
		}
	}
	if len(relayedLines) == 0 {
		goldenContractFail(t, "usage", resultLine, "in_subagent", "no usage line of a relayed child exists to fold into tokens: the session total has to include spend the parent did not make itself")
		return
	}
	var folded []string
	for _, name := range goldenTokenFields {
		if perField[name] != 0 {
			folded = append(folded, fmt.Sprintf("%s=%g", name, perField[name]))
		}
	}
	if len(folded) == 0 {
		goldenContractFail(t, "usage", relayedLines[0], "tokens", "report zero in all six fields across %d relayed line(s): the session total (%s) is then identical with every child line dropped, so it proves nothing about folding — at least one field has to carry a child's share",
			len(relayedLines), goldenJSONValue(result["tokens"]))
	}
}

// TestGoldenPassResultNumTurnsCountsParentCallsOnly pins num_turns as the parent's own
// Model calls: each start line opens one call in the process that wrote it, and a
// relayed child's start lines belong to the child's own Pass — the child's result
// event states its own num_turns, and counting both would charge the parent for turns
// it never made.
func TestGoldenPassResultNumTurnsCountsParentCallsOnly(t *testing.T) {
	raw, events := goldenReadPass(t)
	result := parentResult(t, raw)
	resultLine := len(events)

	var parentStarts, allStarts int
	for i, ev := range events {
		if goldenKindOf(t, ev, i+1) != "start" {
			continue
		}
		allStarts++
		if ev["in_subagent"] != true {
			parentStarts++
		}
	}
	turns, ok := result["num_turns"].(float64)
	if !ok {
		goldenContractFail(t, "result", resultLine, "num_turns", "= %s, want a JSON number of Model calls", goldenJSONValue(result["num_turns"]))
		return
	}
	if int(turns) != parentStarts {
		goldenContractFail(t, "result", resultLine, "num_turns", "= %v, want %d, the number of start lines the parent wrote (the log holds %d start lines in all, %d of them relayed children's own calls)", turns, parentStarts, allStarts, allStarts-parentStarts)
	}
	if allStarts == parentStarts {
		goldenContractFail(t, "result", resultLine, "num_turns", "cannot be shown to count the parent alone: the log holds no relayed start line, so the parent-only and whole-file counts are the same number")
	}
}

// TestGoldenPassResultModelsIncludeTheAnsweringChild pins the run's model set as the
// ids that answered: foldChildUsage adds a relayed usage event's model id to the set,
// so the child model that answered appears without the parent ever having named it in
// a completion — and the rate-limited model stays out, because a 429 leaves no usage
// event to fold and no answer to name.
func TestGoldenPassResultModelsIncludeTheAnsweringChild(t *testing.T) {
	raw, events := goldenReadPass(t)
	result := parentResult(t, raw)
	resultLine := len(events)

	var answered []string
	for i, ev := range events {
		if ev["kind"] != "usage" || ev["in_subagent"] != true {
			continue
		}
		model, ok := ev["model"].(string)
		if !ok || model == "" {
			goldenContractFail(t, "usage", i+1, "model", "= %s, want the id of the child model whose spend this event folds into the run", goldenJSONValue(ev["model"]))
			continue
		}
		if !slices.Contains(answered, model) {
			answered = append(answered, model)
		}
	}
	if len(answered) == 0 {
		goldenContractFail(t, "result", resultLine, "models", "no relayed usage line names a child model, so nothing here pins a child's id into the run's model set")
		return
	}

	models, ok := result["models"].([]any)
	if !ok {
		goldenContractFail(t, "result", resultLine, "models", "= %s, want a JSON array of model ids", goldenJSONValue(result["models"]))
		return
	}
	present := map[string]bool{}
	for _, m := range models {
		id, _ := m.(string)
		present[id] = true
	}
	for _, model := range answered {
		if !present[model] {
			goldenContractFail(t, "result", resultLine, "models", "= %s, which omits %q: a child that answered spent tokens this run owes for, and its id belongs in the set the relay folds in", goldenJSONValue(result["models"]), model)
		}
	}
	if present[goldenRateLimitedModel] {
		goldenContractFail(t, "result", resultLine, "models", "includes %q: that spawn was rate-limited and never answered, so it earned no usage event and no id in the run's model set", goldenRateLimitedModel)
	}
}

// goldenDecodeEventLines reads a stream the way a host that pasted the Driver's stdout
// and stderr into one pass.jsonl has to read it: line by line, keeping every JSON
// object in order and ignoring everything else — blank lines, and the
// "spindrift-driver: …" notes main.go and subagent.go write.
//
// What counts as an event is the relay's own rule (relaySubagentOutput): the line has
// to unmarshal into a populated object, so a truncated object, a bare "null", an array
// and prose all fall out instead of arriving as a line with no kind.
//
// stdoutEvents is deliberately not this reader. It fails the run on any line that is
// not an event, which is the promise the Driver's own stdout owes and the opposite of
// what a merged capture can offer.
func goldenDecodeEventLines(raw string) []map[string]any {
	var out []map[string]any
	for _, line := range strings.Split(raw, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		var ev map[string]any
		// The nil check is what catches a bare "null", which unmarshals into a nil map
		// with no error at all — the same guard relaySubagentOutput needs.
		if err := json.Unmarshal([]byte(trimmed), &ev); err != nil || ev == nil {
			continue
		}
		out = append(out, ev)
	}
	return out
}

// goldenKindSequence spells a decoded stream as its kinds, because "9 events instead of
// 10" tells a reader nothing about which one went missing.
func goldenKindSequence(events []map[string]any) []string {
	kinds := make([]string, 0, len(events))
	for _, ev := range events {
		kind, _ := ev["kind"].(string)
		if kind == "" {
			kind = "(none)"
		}
		kinds = append(kinds, kind)
	}
	return kinds
}

// goldenStderrNotes are three of the shapes the Driver writes to stderr: the POST line
// and the retry notice from main.go — which only ever follows a 5xx or a transport
// failure, a 429 being a non-retryable 4xx that answers with a reset — and the relay's
// dropped-line note from subagent.go. None of them is JSON, so none may survive
// goldenDecodeEventLines — and they have to be interleaved rather than appended, because
// the promise is about order across a mixed stream, not about a reader that skips a
// header.
var goldenStderrNotes = []string{
	"spindrift-driver: POST http://127.0.0.1:43219/v1/chat/completions",
	"spindrift-driver: retry 2/3 in 100ms (server returned 500 Internal Server Error: upstream unavailable)",
	`spindrift-driver: subagent "scout": dropped non-event output line: heartbeat`,
}

// goldenInterleaveStderr rewrites an event stream as a merged capture: every event line
// followed by one of the Driver's stderr notes and a blank line, the notes cycling so
// more than one shape is in front of the reader.
func goldenInterleaveStderr(raw string) string {
	lines := strings.Split(strings.TrimSuffix(raw, "\n"), "\n")
	var b strings.Builder
	for i, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		b.WriteString(line)
		b.WriteString("\n")
		b.WriteString(goldenStderrNotes[i%len(goldenStderrNotes)])
		b.WriteString("\n\n")
	}
	return b.String()
}

// TestGoldenDecodeEventLinesSkipsNonEventLines is the unit half of the merged-capture
// promise: of a stream whose event lines are mixed with the Driver's stderr notes,
// blank and whitespace-only lines, a truncated object, a JSON array, a JSON string and
// a bare null, exactly the well-formed objects come back, in file order, with their
// values intact.
func TestGoldenDecodeEventLinesSkipsNonEventLines(t *testing.T) {
	stream := strings.Join([]string{
		`{"kind":"start","message_id":"msg-1","timestamp":1700000000250}`,
		"spindrift-driver: POST http://127.0.0.1:43219/v1/chat/completions",
		"",
		"   ",
		`{"kind":"text","message_id":"msg-1","timestamp":1700000000500,"text":"thinking out loud"}`,
		`{"kind":"usage","message_id":"msg-1","timestamp":1700000000750`,
		`["not","an","event"]`,
		`"not an object"`,
		`null`,
		`spindrift-driver: subagent "scout": dropped non-event output line: heartbeat`,
		`{"kind":"result","message_id":"msg-9","timestamp":1700000001000,"text":"done","num_turns":1}`,
	}, "\n") + "\n"

	got := goldenDecodeEventLines(stream)
	want := []struct {
		kind    string
		message string
		text    string
	}{
		{kind: "start", message: "msg-1"},
		{kind: "text", message: "msg-1", text: "thinking out loud"},
		{kind: "result", message: "msg-9", text: "done"},
	}
	wantKinds := make([]string, 0, len(want))
	for _, w := range want {
		wantKinds = append(wantKinds, w.kind)
	}
	if gotKinds := goldenKindSequence(got); strings.Join(gotKinds, ",") != strings.Join(wantKinds, ",") {
		t.Fatalf("goldenDecodeEventLines recovered %d event(s) %v from a %d-line stream whose events are %v, want exactly those, in order",
			len(gotKinds), strings.Join(gotKinds, ","), strings.Count(stream, "\n"), strings.Join(wantKinds, ","))
	}
	for i, w := range want {
		if got[i]["message_id"] != w.message {
			t.Errorf("%s event %d message_id = %v, want %q: a surviving object has to survive with its fields", w.kind, i+1, got[i]["message_id"], w.message)
		}
		if w.text != "" && got[i]["text"] != w.text {
			t.Errorf("%s event %d text = %v, want %q", w.kind, i+1, got[i]["text"], w.text)
		}
	}
}

// TestGoldenPassDecodesWithStderrInterleaved is the promise a host recorder relies on:
// paste this artifact's lines and the Driver's stderr into one file in the order they
// arrived, and every event still decodes — same count, same order, same kinds, same
// values. It is a property of a capture, so it is tested on a capture rather than on
// the committed bytes: the artifact itself stays clean JSONL, which goldenReadPass and
// stdoutEvents hold it to.
func TestGoldenPassDecodesWithStderrInterleaved(t *testing.T) {
	raw, events := goldenReadPass(t)
	merged := goldenInterleaveStderr(raw)

	// That the capture is a capture: three lines per event, one event line and two
	// non-event lines after it. Without this the assertion below could pass on a
	// goldenInterleaveStderr that interleaved nothing.
	if lines := strings.Count(merged, "\n"); lines != 3*len(events) {
		t.Fatalf("the built capture holds %d line(s), want 3 per event across %d event(s): every event has to be followed by a stderr note and a blank line", lines, len(events))
	}

	got := goldenDecodeEventLines(merged)

	if len(got) != len(events) {
		t.Fatalf("a merged capture of %s decodes to %d event(s) %v, want the artifact's %d %s\nregenerate the artifact with: %s",
			goldenPassPath, len(got), goldenKindSequence(got), len(events), strings.Join(goldenKindSequence(events), ","), goldenRegenerateCmd)
	}
	for i := range events {
		if !reflect.DeepEqual(got[i], events[i]) {
			t.Errorf("merged capture event %d is %s, want the artifact's line %d %s: interleaving stderr may not lose, reorder or alter an event",
				i+1, goldenJSONValue(got[i]), i+1, goldenJSONValue(events[i]))
		}
	}
}

// goldenCredentialShapes are the shapes a log handed to somebody else has to be free
// of: the prefixes the vendors themselves stamp on a live key (OpenAI, GitHub, AWS,
// Slack, Google), a PEM private-key header, and the two spellings of the HTTP header a
// token travels in. They are matched exactly and case-sensitively as the vendors spell
// them — the aim is to catch a real key that reached the log, not to guess at mangled
// ones, and a looser match is what goldenBenignText shows would flag ordinary output.
var goldenCredentialShapes = []string{
	"sk-", // OpenAI-style API key
	"ghp_",
	"gho_",
	"github_pat_", // GitHub personal-access and OAuth tokens
	"AKIA",        // AWS access key id
	"xox",
	"xapp", // Slack legacy and app-level tokens
	"AIza", // Google API key
	"Bearer ",
	"Authorization:", // the header a token is sent in
	"PRIVATE KEY",    // a PEM private-key header
}

// goldenCredentialShapedStrings reports which of goldenCredentialShapes occur in raw,
// each at most once and in the order above, so the report names what to look for
// without dumping a whole line that may itself be the secret.
func goldenCredentialShapedStrings(raw string) []string {
	var hits []string
	for _, shape := range goldenCredentialShapes {
		if strings.Contains(raw, shape) {
			hits = append(hits, shape)
		}
	}
	return hits
}

// goldenFieldHolding names which fields of an event carry a credential shape, so the
// scan's failure points a reader at the field to fix without reprinting the value that
// may itself be the secret.
func goldenFieldHolding(ev map[string]any, shape string) string {
	var fields []string
	for name, v := range ev {
		if strings.Contains(goldenJSONValue(v), shape) {
			fields = append(fields, name)
		}
	}
	sort.Strings(fields)
	if len(fields) == 0 {
		return "(whole line)"
	}
	return strings.Join(fields, ",")
}

// goldenCredentialSamples pair each credential shape goldenCredentialShapes scans for
// with an obviously-fake literal of it — the examples the vendors publish in their own
// docs, which are inert and safe to keep in a test. Every one has to be detected:
// an undetected shape is a credential this artifact could be carrying unnoticed.
var goldenCredentialSamples = []struct {
	shape  string
	sample string
}{
	{"sk-", "sk-test-0123456789abcdef0123456789abcdef"},
	{"ghp_", "ghp_0123456789ABCDEFGHIJKLMNOPQRSTUVWX"},
	{"gho_", "gho_0123456789ABCDEFGHIJKLMNOPQRSTUVWX"},
	{"github_pat_", "github_pat_11FAKE0T0fakefake0fake_FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"},
	{"AKIA", "AKIAIOSFODNN7EXAMPLE"},
	{"xox", "xoxb-0000000000-0000000000000-fakereplacementtoken"},
	{"xapp", "xapp-1-A0000000000-000000000000-0000000000000000000000000000"},
	{"Bearer ", "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.fake.jwt"},
	{"Authorization:", "Authorization: Bearer faketoken"},
	{"PRIVATE KEY", "-----BEGIN PRIVATE KEY-----"},
	{"AIza", "AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE0"},
}

// goldenBenignText is what this artifact actually carries and what must not read as
// credential-shaped: a normalised path, the Driver's own prose, the base64 a Tool's
// output can hold, a rate-limit rejection that names a quota, and a reset instant. A
// scanner loose enough to match on words like "key", "token" or "limit" would report
// several of these, and a scan that cries wolf is one a reader stops reading.
var goldenBenignText = []string{
	goldenRunDir + "/notes.txt",
	"Wrote and edited notes.txt and delegated a review; the repeated command failed with exit code 3.",
	"SGVsbG8gZnJvbSB0aGUgbm90ZXMgZmlsZS4gUmVhZCBpdCBiZWZvcmUgZWRpdGluZywgdGhlbiByZXBvcnQgdGhlIGRpZmYu",
	goldenRateLimitBody,
	"1787274123000",
	"spindrift-driver: subagent \"scout\": dropped non-event output line: heartbeat",
}

// TestGoldenCredentialShapedStringsDetectsEveryShape is the scanner's own unit test:
// each shape it claims to catch, it catches.
func TestGoldenCredentialShapedStringsDetectsEveryShape(t *testing.T) {
	if len(goldenCredentialSamples) != len(goldenCredentialShapes) {
		t.Fatalf("%d credential shapes are scanned but %d fake samples are tested: every shape needs a sample, or the scan silently stops covering one",
			len(goldenCredentialShapes), len(goldenCredentialSamples))
	}
	for _, c := range goldenCredentialSamples {
		hits := goldenCredentialShapedStrings(c.sample)
		if !slices.Contains(hits, c.shape) {
			t.Errorf("goldenCredentialShapedStrings does not detect %q in a literal of it (%q): a run could leak this shape and the scan would call the log clean", c.shape, c.sample)
		}
	}
}

// TestGoldenCredentialShapedStringsIgnoreBenignText is the other half of the unit test:
// the text this Driver really writes stays out of the report.
func TestGoldenCredentialShapedStringsIgnoreBenignText(t *testing.T) {
	for _, text := range goldenBenignText {
		if hits := goldenCredentialShapedStrings(text); len(hits) > 0 {
			t.Errorf("goldenCredentialShapedStrings reports %v in ordinary Driver output %q: a scan that flags the artifact's own prose is one a reader learns to ignore", hits, text)
		}
	}
	if hits := goldenCredentialShapedStrings(""); len(hits) > 0 {
		t.Errorf("goldenCredentialShapedStrings reports %v in empty text", hits)
	}
}

// TestGoldenPassCarriesNoCredentialShapedStrings scans the committed artifact, which is
// the file a downstream host keeps and may paste into an issue: nothing in it may read
// as a credential.
//
// The scripted run does authenticate — goldenPassStream sets $SPINDRIFT_API_KEY to
// goldenFakeKey, which is the value run() sends as the request's bearer token — so
// nothing here is clean because the run had no key to write. The key goes out in a
// header and never onto stdout, and that is the fact this test pins about the emitter.
func TestGoldenPassCarriesNoCredentialShapedStrings(t *testing.T) {
	raw, events := goldenReadPass(t)
	lines := strings.Split(strings.TrimSuffix(raw, "\n"), "\n")
	for _, shape := range goldenCredentialShapedStrings(raw) {
		line, kind, field := 0, "(none)", "line"
		for i, l := range lines {
			if !strings.Contains(l, shape) {
				continue
			}
			line = i + 1
			if line <= len(events) {
				kind, _ = events[line-1]["kind"].(string)
				field = goldenFieldHolding(events[line-1], shape)
			}
			break
		}
		// The offending value is deliberately not reproduced: a line reported as leaked
		// may be the leak, and a test log is read by more people than the artifact is.
		goldenContractFail(t, kind, line, field, "holds the credential shape %q (value elided, for the reason just stated): this file is handed to other people, and a Driver that writes a bearer token or a key into its event stream leaks every run it runs", shape)
	}

	// The run's own key, named rather than implied: were the emitter ever changed to
	// log what it authenticates with, the scan above would only catch it if the key
	// happened to be shaped like one of the samples, and goldenFakeKey is deliberately
	// not.
	if strings.Contains(raw, goldenFakeKey) {
		t.Errorf("%s carries the API key this run authenticated with (%q): the key is a header value and belongs on no event line\nregenerate the artifact with: %s", goldenPassPath, goldenFakeKey, goldenRegenerateCmd)
	}
	// The placeholder is the reason the scan is meaningful, so it has to stay
	// un-shapeable: if someone renames it to something credential-shaped, this fails
	// before the artifact does.
	if hits := goldenCredentialShapedStrings(goldenFakeKey); len(hits) > 0 {
		t.Errorf("goldenFakeKey %q matches the credential shapes %v: the scripted run's placeholder must not itself read as a credential, or the scan of the artifact cannot tell a placeholder from a leak", goldenFakeKey, hits)
	}
}
