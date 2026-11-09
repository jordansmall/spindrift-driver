package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
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
