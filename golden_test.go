package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// goldenUpdate is the regeneration switch for the committed golden log, bound to
// the -update flag TestMain registers. It is a *bool rather than a bool because
// the value only exists once m.Run() has parsed the command line.
var goldenUpdate *bool

// goldenPassPath is the committed contract artifact: the stdout of one scripted
// Pass, byte for byte. It is a file rather than a Go literal so a downstream
// decoder author can point a parser at it, and it lives in testdata so tooling
// skips it as source.
const goldenPassPath = "testdata/pass.jsonl"

// goldenRegenerateCmd is named in every failure this test can produce: an
// artifact that drifted is only actionable if the reader knows how to rewrite it.
const goldenRegenerateCmd = "go test -run TestGoldenPassLog -update"

// TestGoldenPassLog compares one scripted Pass through the whole Driver against
// the committed golden byte for byte.
//
// The stream is produced in-process — run() into a bytes.Buffer with
// pinSteppingClock fixing the clock and newScriptedChatServer fixing every
// response — because the nix check gives the suite bash and coreutils only (no
// jq, no git, no network), so regeneration cannot shell out.
//
// The run is shaped to put every interesting event form in one Pass: a Model
// call that streams thinking ahead of its prose and then a succeeding bash tool
// call, a second call with no prose at all (which is why it carries no text
// event) and a failing bash tool call, then a final answer. That yields start,
// reasoning, text, usage, tool_call, tool_result in both its success and
// is_error shapes, and result.
//
// The error kind is deliberately absent from this artifact: a single Pass cannot
// both succeed and fail, and issue #27 owns the conformance-shaped log that
// covers all seven kinds.
func TestGoldenPassLog(t *testing.T) {
	got := goldenPassStream(t)

	// Shape is asserted on the produced stream either way, so -update cannot
	// quietly commit a degenerate artifact (a run that emitted nothing, or lost a
	// tool_result pair) and then have the byte compare bless it.
	assertStreamContract(t, got)

	if *goldenUpdate {
		if err := os.MkdirAll(filepath.Dir(goldenPassPath), 0o755); err != nil {
			t.Fatalf("create %s: %v", filepath.Dir(goldenPassPath), err)
		}
		if err := os.WriteFile(goldenPassPath, got, 0o644); err != nil {
			t.Fatalf("write %s: %v", goldenPassPath, err)
		}
		t.Logf("wrote %s: %d bytes, %d event(s)", goldenPassPath, len(got), len(strings.Split(strings.TrimSuffix(string(got), "\n"), "\n")))
		return
	}

	want, err := os.ReadFile(goldenPassPath)
	if err != nil {
		t.Fatalf("read golden %s: %v\nregenerate it with: %s", goldenPassPath, err, goldenRegenerateCmd)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("%s drifted from what the Driver emits:\n%s\nregenerate with: %s", goldenPassPath, describeByteDrift(got, want), goldenRegenerateCmd)
	}
}

// goldenPassStream drives one Pass and returns its raw stdout. The scripted
// server supplies three responses; the two bash commands are printf-only and one
// exits non-zero, so nothing here reads the repo, the clock, or the environment.
//
// Only the first response is framed by hand: its thinking has to arrive as the
// delta.reasoning_content chunks a real reasoning endpoint sends, which a whole
// completion body cannot express. The answers themselves stay whole bodies run
// through sseChatCompletion, so the frames behind every other event are the ones
// this artifact has always pinned.
func goldenPassStream(t *testing.T) []byte {
	t.Helper()
	// A stepping clock, not a frozen one: the result event's durations have to be
	// non-zero to be worth pinning. The step is read-per-event, not wall clock, so
	// 250ms keeps the pinned numbers plausible rather than degenerate: a quarter
	// second between events, a quarter second per round trip, a several-second run.
	pinSteppingClock(t, 1700000000000, 250)

	srv := newScriptedSSERawServer(t,
		// Call 1: thinking in two fragments (so the event proves accumulation, not
		// one frame relabelled) ahead of prose with real newlines, which the emitter
		// must escape rather than trim, plus a bash call that succeeds.
		sseReasoningFrames("llama-3.1-8b",
			"reading the tree before saying anything about it; ",
			"one safe command first, then one allowed to fail")+
			sseChatCompletion(`{"model":"llama-3.1-8b","usage":{"prompt_tokens":104,"completion_tokens":21,"prompt_cache_hit_tokens":64},"choices":[{"message":{"role":"assistant","content":"I will look at the repo first,\nthen report.\n","tool_calls":[{"id":"call_alpha","type":"function","function":{"name":"bash","arguments":"{\"command\":\"printf 'hello world'\"}"}}]}}]}`),
		// Call 2: no content at all (so no text event, and no thinking either) and a
		// command that exits 3.
		sseChatCompletion(`{"model":"llama-3.1-8b","usage":{"prompt_tokens":148,"completion_tokens":18},"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_beta","type":"function","function":{"name":"bash","arguments":"{\"command\":\"printf 'this one fails'; exit 3\"}"}}]}}]}`),
		// Call 3: the final answer, from a different model id, reporting cache read
		// in the OpenAI spelling.
		sseChatCompletion(`{"model":"qwen2.5-coder","usage":{"prompt_tokens":177,"completion_tokens":12,"prompt_tokens_details":{"cached_tokens":25}},"choices":[{"message":{"role":"assistant","content":"Ran both commands; the second failed with exit code 3."}}]}`),
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"inspect the repo and report", "--model", "spindrift-router", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("golden Pass exited %d, want 0; stderr = %q", code, stderr.String())
	}
	if got := srv.requestBodies(t); len(got) != 3 {
		t.Fatalf("golden Pass made %d Model call(s), want 3", len(got))
	}
	return stdout.Bytes()
}

// sseReasoningFrames scripts thinking as the delta.reasoning_content chunks a
// reasoning endpoint streams before any content. Fragments are prose written
// through %q, so a fragment containing a quote or a backslash would need
// escaping this helper does not attempt.
func sseReasoningFrames(model string, fragments ...string) string {
	var b strings.Builder
	for _, f := range fragments {
		b.WriteString(sseFrame(fmt.Sprintf(`{"model":%q,"choices":[{"index":0,"delta":{"reasoning_content":%q}}]}`, model, f)))
	}
	return b.String()
}

// assertStreamContract checks what a byte compare cannot: that the file is a
// legal JSONL event stream and still the run this golden is meant to describe.
// Reading it as plain JSON is what issue #19 demands of a decoder — no driverkit
// import, no shared type.
func assertStreamContract(t *testing.T, raw []byte) {
	t.Helper()
	events := stdoutEvents(t, string(raw)) // every line one JSON object, no blanks, one trailing newline each
	var lastTS int64 = -1
	kinds := make([]string, 0, len(events))
	for i, ev := range events {
		kind, ok := ev["kind"].(string)
		if !ok || kind == "" {
			t.Errorf("event %d has no non-empty kind: %v", i+1, ev)
			continue
		}
		kinds = append(kinds, kind)
		if _, ok := ev["timestamp"]; !ok {
			t.Errorf("%s event %d carries no timestamp field", kind, i+1)
		}
		ts, ok := ev["timestamp"].(float64)
		if !ok || ts <= 0 {
			t.Errorf("%s event %d timestamp = %v, want epoch milliseconds", kind, i+1, ev["timestamp"])
			continue
		}
		if int64(ts) < lastTS {
			t.Errorf("event %d timestamp %d goes backwards from %d", i+1, int64(ts), lastTS)
		}
		lastTS = int64(ts)
	}

	wantKinds := []string{
		"start", "reasoning", "text", "usage", "tool_call", "tool_result",
		"start", "usage", "tool_call", "tool_result",
		"start", "text", "usage", "result",
	}
	if strings.Join(kinds, ",") != strings.Join(wantKinds, ",") {
		t.Fatalf("kind sequence = %s, want %s", strings.Join(kinds, ","), strings.Join(wantKinds, ","))
	}

	// The two tool_result shapes the golden exists to pin: one clean, one is_error.
	var failed, clean int
	for _, ev := range events {
		if ev["kind"] != "tool_result" {
			continue
		}
		isErr, present := ev["is_error"]
		if !present {
			clean++
			continue
		}
		if isErr != true {
			t.Errorf("is_error present but %v: omitempty must keep a clean result off the wire entirely", isErr)
		}
		failed++
	}
	if clean != 1 || failed != 1 {
		t.Errorf("tool_result shapes = %d clean and %d is_error, want one of each", clean, failed)
	}

	result := events[len(events)-1]
	if turns, _ := result["num_turns"].(float64); int(turns) != 3 {
		t.Errorf("result num_turns = %v, want 3 Model calls", result["num_turns"])
	}
	for _, name := range []string{"duration_ms", "duration_api_ms"} {
		if v, _ := result[name].(float64); v <= 0 {
			t.Errorf("result %s = %v, want a non-zero pinned duration", name, result[name])
		}
	}
	tokens, _ := result["tokens"].(map[string]any)
	// 104+148+177 input, 21+18+12 output, 64 (llama.cpp spelling) + 25 (OpenAI
	// details spelling) cache read, summed by the Driver from the usage events.
	for name, want := range map[string]float64{"input": 429, "output": 51, "cache_read": 89} {
		if got, _ := tokens[name].(float64); got != want {
			t.Errorf("result tokens[%s] = %v, want the three calls summed to %v", name, tokens[name], want)
		}
	}
	models, _ := result["models"].([]any)
	if len(models) != 2 || models[0] != "llama-3.1-8b" || models[1] != "qwen2.5-coder" {
		t.Errorf("result models = %v, want the two responding ids sorted", result["models"])
	}

	// The thinking the golden exists to pin: exactly one event, on the call that
	// streamed it, riding ahead of the text event it produced.
	thinkingAt := -1
	var thinking string
	for i, ev := range events {
		if ev["kind"] != "reasoning" {
			continue
		}
		if thinkingAt >= 0 {
			t.Errorf("second reasoning event %d: one message's fragments belong in one event", i+1)
			continue
		}
		thinkingAt = i
		thinking, _ = ev["text"].(string)
		if ev["message_id"] != "msg-1" {
			t.Errorf("reasoning event message_id = %v, want msg-1, the one call that thinks", ev["message_id"])
		}
	}
	if thinking == "" {
		t.Fatalf("no reasoning event carrying thinking: the first scripted call must stream it")
	}
	if next := thinkingAt + 1; next >= len(events) || events[next]["kind"] != "text" {
		t.Errorf("reasoning event %d does not ride ahead of a text event", thinkingAt+1)
	}
	// Thinking is its own channel: neither the prose event nor the result event may
	// carry any of it, or a decoder that reads only the answer gets it twice.
	for _, ev := range events {
		if ev["kind"] != "text" && ev["kind"] != "result" {
			continue
		}
		if prose, _ := ev["text"].(string); prose != "" && strings.Contains(prose, thinking) {
			t.Errorf("%s text %q carries the thinking, which stays out of the answer", ev["kind"], prose)
		}
	}
}

// describeByteDrift points at the first differing line rather than dumping two
// whole streams: the reader needs to know which event moved, not re-diff a dozen
// lines by eye.
func describeByteDrift(got, want []byte) string {
	g := strings.Split(string(got), "\n")
	w := strings.Split(string(want), "\n")
	for i := 0; i < len(g) || i < len(w); i++ {
		var a, b string
		if i < len(g) {
			a = g[i]
		}
		if i < len(w) {
			b = w[i]
		}
		if a != b {
			return fmt.Sprintf("line %d:\n  emitted: %s\n  golden:  %s\n(emitted %d bytes, golden %d bytes)", i+1, a, b, len(got), len(want))
		}
	}
	return fmt.Sprintf("bytes equal but framing differs (emitted %d bytes, golden %d bytes)", len(got), len(want))
}
