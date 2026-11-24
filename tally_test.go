package main

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// The run tally's accounting rules, tested at the struct level rather than
// through a run: the two entry points (own usage, relayed child usage) and the
// projection the result event reads are all reachable without a Model call, a
// child process or a scripted server, and a malformed relayed line is far
// cheaper to spell as JSON here than to make a child emit.

// allSixTokens fills every token field with a distinct value so a fold that
// drops one is caught rather than averaged away.
var allSixTokens = tokenTotals{Input: 100, Output: 20, CacheRead: 7, CacheWrite5m: 3, CacheWrite1h: 2, CostUSD: 1.5}

// relayedEventFields builds one relayed child event line the way the relay hands
// it over: already parsed into raw JSON fields. A value of "" omits the key, so
// the missing-key cases need no second helper.
func relayedEventFields(fields map[string]string) map[string]json.RawMessage {
	out := make(map[string]json.RawMessage, len(fields))
	for k, v := range fields {
		if v == "" {
			continue
		}
		out[k] = json.RawMessage(v)
	}
	return out
}

// relayedUsageFields is one usage line carrying the given raw JSON tokens and
// raw JSON model id.
func relayedUsageFields(tokens, model string) map[string]json.RawMessage {
	return relayedEventFields(map[string]string{"kind": `"usage"`, "tokens": tokens, "model": model})
}

func TestRunTallyOwnUsageFoldsEveryTokenField(t *testing.T) {
	tally := newRunTally()
	tally.addOwnUsage("m-parent", allSixTokens)
	tally.addOwnUsage("m-parent", tokenTotals{Input: 30, Output: 5, CacheRead: 4, CacheWrite5m: 2, CacheWrite1h: 1, CostUSD: 0.25})

	want := tokenTotals{Input: 130, Output: 25, CacheRead: 11, CacheWrite5m: 5, CacheWrite1h: 3, CostUSD: 1.75}
	if tally.own != want {
		t.Errorf("own tally = %+v, want all six fields summed to %+v", tally.own, want)
	}
	if got := tally.totals(); got != want {
		t.Errorf("totals() = %+v, want the own usage projected field by field as %+v", got, want)
	}
}

// TestRunTallyOwnUsageRecordsModelUnconditionally pins the half of the asymmetry
// that belongs to the parent's own entry point: sendChat guarantees a non-empty
// model id, so recording it needs no guard, and a tally that silently dropped an
// unexpected "" would lose a model from the result event's set.
func TestRunTallyOwnUsageRecordsModelUnconditionally(t *testing.T) {
	tally := newRunTally()
	tally.addOwnUsage("m-b", tokenTotals{Input: 1})
	tally.addOwnUsage("m-a", tokenTotals{Input: 1})
	tally.addOwnUsage("", tokenTotals{Input: 1})

	if got, want := tally.modelIDs(), []string{"", "m-a", "m-b"}; !reflect.DeepEqual(got, want) {
		t.Errorf("modelIDs() = %v, want the empty own model id recorded too (%v)", got, want)
	}
}

func TestRunTallyRelayedUsageFoldsIntoDelegated(t *testing.T) {
	tally := newRunTally()
	tally.addOwnUsage("m-parent", tokenTotals{Input: 100, Output: 20})
	tally.addRelayedUsage(relayedUsageFields(`{"input":7,"output":1,"cost_usd":0.25}`, `"m-reviewer"`))

	if got, want := tally.delegated, (tokenTotals{Input: 7, Output: 1, CostUSD: 0.25}); got != want {
		t.Errorf("delegated tally = %+v, want %+v", got, want)
	}
	// The two sums stay distinguishable until the projection: the parent's own
	// usage events are one per own Model call and must not carry child spend.
	if got, want := tally.own, (tokenTotals{Input: 100, Output: 20}); got != want {
		t.Errorf("own tally = %+v, want the relayed fold to leave it at %+v", got, want)
	}
	if got, want := tally.totals(), (tokenTotals{Input: 107, Output: 21, CostUSD: 0.25}); got != want {
		t.Errorf("totals() = %+v, want own and delegated summed as %+v", got, want)
	}
}

// TestRunTallyChargesChildThatReportedNoResult is why the roll-up folds usage
// events rather than a child's result totals: a child that dies emits no result
// event at all, and the spend it already made is real.
func TestRunTallyChargesChildThatReportedNoResult(t *testing.T) {
	tally := newRunTally()
	tally.addRelayedUsage(relayedUsageFields(`{"input":7,"output":1}`, `"m-reviewer"`))
	tally.addRelayedUsage(relayedUsageFields(`{"input":3,"output":2,"cache_read":8}`, `"m-reviewer"`))

	if got, want := tally.totals(), (tokenTotals{Input: 10, Output: 3, CacheRead: 8}); got != want {
		t.Errorf("totals() = %+v, want both usage-only child calls charged as %+v", got, want)
	}
}

// TestRunTallyIgnoresChildResultTotals is the other half of that rule: a child's
// result event already sums the usage events that were folded as they relayed,
// so reading its tokens too would bill the same tokens twice.
func TestRunTallyIgnoresChildResultTotals(t *testing.T) {
	tally := newRunTally()
	tally.addRelayedUsage(relayedUsageFields(`{"input":7,"output":1}`, `"m-reviewer"`))
	baseline := tally.totals()

	tally.addRelayedUsage(relayedEventFields(map[string]string{
		"kind":   `"result"`,
		"tokens": `{"input":9000,"output":9000,"cost_usd":99}`,
		"model":  `"m-reviewer"`,
	}))
	if got := tally.totals(); got != baseline {
		t.Errorf("totals() = %+v after a child result line, want the usage-folded %+v unchanged", got, baseline)
	}

	// A run whose only child line was a result event charged nothing at all.
	fresh := newRunTally()
	fresh.addRelayedUsage(relayedEventFields(map[string]string{"kind": `"result"`, "tokens": `{"input":9000}`}))
	if got := fresh.totals(); got != (tokenTotals{}) {
		t.Errorf("totals() = %+v, want a result-only child line to contribute nothing", got)
	}
}

// TestRunTallyMalformedRelayedTokensContributeNothing: one odd child line must
// not corrupt the parent's totals, so a tokens value that is not a token object
// folds nothing rather than partially — or, worse, fails the run.
func TestRunTallyMalformedRelayedTokensContributeNothing(t *testing.T) {
	for name, tokens := range map[string]string{
		"absent":      "",
		"null":        "null",
		"string":      `"not an object"`,
		"number":      `42`,
		"array":       `[{"input":1}]`,
		"wrong shape": `{"input":"seven"}`,
	} {
		tally := newRunTally()
		tally.addOwnUsage("m-parent", tokenTotals{Input: 100, Output: 20})
		before := tally.totals()

		tally.addRelayedUsage(relayedUsageFields(tokens, `"m-reviewer"`))

		if tally.delegated != (tokenTotals{}) {
			t.Errorf("%s tokens: delegated tally = %+v, want nothing folded", name, tally.delegated)
		}
		if got := tally.totals(); got != before {
			t.Errorf("%s tokens: totals() = %+v, want the pre-fold %+v unchanged", name, got, before)
		}
		// Relaying and accounting are separate jobs: the line still counted as the
		// child's, model id and all.
		if _, ok := tally.models["m-reviewer"]; !ok {
			t.Errorf("%s tokens: model set = %v, want the model id recorded despite unusable tokens", name, tally.modelIDs())
		}
	}
}

func TestRunTallyModelIDsAreSortedAndRejectUnusableOnes(t *testing.T) {
	tally := newRunTally()
	tally.addRelayedUsage(relayedUsageFields(`{"input":1}`, `"m-reviewer"`))
	tally.addRelayedUsage(relayedUsageFields(`{"input":1}`, `"m-planner"`))
	// An empty relayed model id is a key the child never filled, not a model: the
	// result event's set would gain a blank entry a host cannot attribute.
	tally.addRelayedUsage(relayedUsageFields(`{"input":1}`, `""`))
	// A model id of the wrong type is unusable the same way malformed tokens are.
	tally.addRelayedUsage(relayedEventFields(map[string]string{"kind": `"usage"`, "tokens": `{"input":1}`, "model": `42`}))
	tally.addRelayedUsage(relayedEventFields(map[string]string{"kind": `"usage"`, "tokens": `{"input":1}`}))

	if got, want := tally.modelIDs(), []string{"m-planner", "m-reviewer"}; !reflect.DeepEqual(got, want) {
		t.Errorf("modelIDs() = %v, want %v sorted with the blank and non-string ids excluded", got, want)
	}
}

// TestRunTallyIgnoresRelayedKindsThatAreNotUsage: the tally listens for usage
// only — prose, tool results and errors carry no spend to fold.
func TestRunTallyIgnoresRelayedKindsThatAreNotUsage(t *testing.T) {
	tally := newRunTally()
	for _, kind := range []string{"start", "reasoning", "text", "tool_call", "tool_result", "error", ""} {
		tally.addRelayedUsage(relayedEventFields(map[string]string{
			"kind":   `"` + kind + `"`,
			"tokens": `{"input":5,"output":5}`,
			"model":  `"m-stray"`,
		}))
	}
	if got := tally.totals(); got != (tokenTotals{}) {
		t.Errorf("totals() = %+v, want only usage lines folded", got)
	}
	if got := tally.modelIDs(); len(got) != 0 {
		t.Errorf("modelIDs() = %v, want a non-usage line to record no model", got)
	}
}

func TestRunTallyAccruesCallsAndAPITime(t *testing.T) {
	tally := newRunTally()
	if tally.calls != 0 || tally.apiMS != 0 {
		t.Fatalf("new tally = calls %d apiMS %d, want both zero", tally.calls, tally.apiMS)
	}
	tally.recordCall()
	tally.recordCall()
	tally.addAPITime(120)
	tally.addAPITime(-20) // a clock that steps back must not panic or reset the sum

	if tally.calls != 2 {
		t.Errorf("calls = %d, want 2", tally.calls)
	}
	if tally.apiMS != 100 {
		t.Errorf("apiMS = %d, want the per-call durations summed to 100", tally.apiMS)
	}
}

// TestRunTallyProjectionFeedsResultPayload pins the projection the result event
// reads: the same three reads main.go makes off its five locals, and a set with
// nothing in it still projects as [] rather than null — resultPayload carries no
// omitempty, so a null models field would be a contract break.
func TestRunTallyProjectionFeedsResultPayload(t *testing.T) {
	tally := newRunTally()
	tally.recordCall()
	tally.addAPITime(40)
	tally.addOwnUsage("m-parent", tokenTotals{Input: 100, Output: 20})
	tally.addRelayedUsage(relayedUsageFields(`{"input":7,"output":1}`, `"m-reviewer"`))

	payload := resultPayload{
		Kind:          "result",
		Tokens:        tally.totals(),
		DurationAPIMs: tally.apiMS,
		NumTurns:      tally.calls,
		Models:        tally.modelIDs(),
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("result payload is not JSON: %v", err)
	}
	if got["num_turns"] != float64(1) || got["duration_api_ms"] != float64(40) {
		t.Errorf("result payload = %s, want num_turns 1 and duration_api_ms 40", raw)
	}
	tokens, _ := got["tokens"].(map[string]any)
	if tokens["input"] != float64(107) || tokens["output"] != float64(21) {
		t.Errorf("result tokens = %v, want own and delegated summed (107 input, 21 output)", tokens)
	}
	if want := []any{"m-parent", "m-reviewer"}; !reflect.DeepEqual(got["models"], want) {
		t.Errorf("result models = %v, want %v", got["models"], want)
	}

	// The empty projection has to encode as [] and never null, because resultPayload
	// carries no omitempty for a host to read a missing models key from.
	raw, err = json.Marshal(resultPayload{Kind: "result", Models: newRunTally().modelIDs()})
	if err != nil || !strings.Contains(string(raw), `"models":[]`) {
		t.Errorf("empty projection marshals to %s (%v), want \"models\":[]", raw, err)
	}
}
