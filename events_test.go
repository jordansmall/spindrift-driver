package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// The tests below read emitted lines back as plain JSON maps: issue #19 bars
// any driverkit import (nix_flake_test.go keeps go.mod require-free), so the
// wire contract is asserted by name, not by shared type.

func decodeLine(t *testing.T, line string) map[string]any {
	t.Helper()
	var ev map[string]any
	if err := json.Unmarshal([]byte(line), &ev); err != nil {
		t.Fatalf("line %q is not a JSON object: %v", line, err)
	}
	return ev
}

// pinClock fixes the event clock for a test and restores the real one after.
func pinClock(t *testing.T, ms int64) {
	t.Helper()
	real := nowMS
	t.Cleanup(func() { nowMS = real })
	nowMS = func() int64 { return ms }
}

func TestEventEmitterStampsInjectedClock(t *testing.T) {
	pinClock(t, 1700000000123)

	var buf bytes.Buffer
	if err := (eventWriter{w: &buf}).emit(event{Kind: "start"}); err != nil {
		t.Fatalf("emit: %v", err)
	}
	got := decodeLine(t, strings.TrimSuffix(buf.String(), "\n"))
	if got["kind"] != "start" {
		t.Errorf("kind = %v, want %q", got["kind"], "start")
	}
	if ts, _ := got["timestamp"].(float64); int64(ts) != 1700000000123 {
		t.Errorf("timestamp = %v, want the injected clock 1700000000123", got["timestamp"])
	}
}

func TestEventEmitterWritesOneLinePerEvent(t *testing.T) {
	pinClock(t, 1700000000000)

	var buf bytes.Buffer
	ew := eventWriter{w: &buf}
	if err := ew.emit(event{Kind: "start", MessageID: "msg-1"}); err != nil {
		t.Fatalf("emit start: %v", err)
	}
	if err := ew.emit(event{Kind: "text", MessageID: "msg-1", Text: "hello"}); err != nil {
		t.Fatalf("emit text: %v", err)
	}
	out := buf.String()
	if !strings.HasSuffix(out, "\n") {
		t.Fatalf("output does not end with a newline: %q", out)
	}
	lines := strings.Split(strings.TrimSuffix(out, "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("got %d lines, want 2: %q", len(lines), out)
	}
	for i, line := range lines {
		if strings.TrimSpace(line) == "" {
			t.Fatalf("line %d is blank: %q", i+1, out)
		}
		decodeLine(t, line)
	}
	if got := decodeLine(t, lines[1])["kind"]; got != "text" {
		t.Errorf("second line kind = %v, want %q", got, "text")
	}
}

func TestEventWireFieldNamesAreSnakeCase(t *testing.T) {
	pinClock(t, 1700000000000)

	var buf bytes.Buffer
	err := (eventWriter{w: &buf}).emit(event{
		Kind:       "tool_call",
		Role:       "assistant",
		MessageID:  "msg-1",
		ToolID:     "call-1",
		ToolName:   "bash",
		ToolInput:  map[string]any{"command": "ls"},
		ToolTarget: "ls",
	})
	if err != nil {
		t.Fatalf("emit: %v", err)
	}
	got := decodeLine(t, strings.TrimSuffix(buf.String(), "\n"))
	for _, name := range []string{"kind", "role", "message_id", "timestamp", "tool_id", "tool_name", "tool_input", "tool_target"} {
		if _, ok := got[name]; !ok {
			t.Errorf("wire line %q has no %q key", buf.String(), name)
		}
	}
	if _, ok := got["tool_input"].(map[string]any); !ok {
		t.Errorf("tool_input = %#v, want a JSON object, not a string", got["tool_input"])
	}
}

func TestEventToolResultKeepsOutputVerbatimOnOneLine(t *testing.T) {
	pinClock(t, 1700000000000)

	// Shell- and server-controlled text: newlines, tabs, quotes, unicode and
	// HTML-significant bytes must all survive verbatim inside one line.
	const output = "line1\nline2\t\"quoted\" <tag> \x26 amp → \u2028\x01 exit code: 1"

	var buf bytes.Buffer
	err := (eventWriter{w: &buf}).emit(event{
		Kind:       "tool_result",
		Role:       "assistant",
		MessageID:  "msg-1",
		ToolID:     "call-1",
		ToolName:   "bash",
		ToolTarget: "make test",
		ToolOutput: output,
		IsError:    true,
	})
	if err != nil {
		t.Fatalf("emit: %v", err)
	}
	raw := buf.String()
	if strings.Count(raw, "\n") != 1 {
		t.Fatalf("event is not exactly one line: %q", raw)
	}
	// Chosen behaviour: HTML escaping is off, so '<', '>' and '&' go on the
	// wire as themselves, never as the \u003c, \u003e and \u0026 that
	// json.Marshal would emit by default.
	for _, lit := range []string{"<tag>", "\x26", "→"} {
		if !strings.Contains(raw, lit) {
			t.Errorf("raw line does not contain %q literally: %q", lit, raw)
		}
	}
	got := decodeLine(t, strings.TrimSuffix(raw, "\n"))
	if got["tool_output"] != output {
		t.Errorf("tool_output = %q, want %q", got["tool_output"], output)
	}
	if got["is_error"] != true {
		t.Errorf("is_error = %v, want true", got["is_error"])
	}
}

// A literal null is the only server argument value that takes tool_input off the
// wire: omitempty on an `any` field fires on a nil interface alone, so an absent
// key means "the server sent null" and never "the arguments were empty". Pinned
// because the fallback path (raw string on unparseable arguments) makes the
// encode rules easy to misread as "anything falsy drops out".
func TestEventToolInputOnlyAbsentForNullArguments(t *testing.T) {
	pinClock(t, 1700000000000)

	for _, tc := range []struct {
		name  string
		input any
		want  bool
	}{
		{"null", nil, false},
		{"empty object", map[string]any{}, true},
		{"zero", 0, true},
		{"empty string", "", true},
		{"false", false, true},
	} {
		var buf bytes.Buffer
		if err := (eventWriter{w: &buf}).emit(event{
			Kind:      "tool_call",
			MessageID: "msg-1",
			ToolID:    "call-1",
			ToolName:  "bash",
			ToolInput: tc.input,
		}); err != nil {
			t.Fatalf("%s arguments: emit: %v", tc.name, err)
		}
		got := decodeLine(t, strings.TrimSuffix(buf.String(), "\n"))
		if _, ok := got["tool_input"]; ok != tc.want {
			t.Errorf("%s arguments: tool_input present = %v, want %v: %q", tc.name, ok, tc.want, buf.String())
		}
	}
}

func TestEventOmitsFieldsTheKindDoesNotFill(t *testing.T) {
	pinClock(t, 1700000000000)

	var buf bytes.Buffer
	if err := (eventWriter{w: &buf}).emit(event{Kind: "start", MessageID: "msg-1"}); err != nil {
		t.Fatalf("emit: %v", err)
	}
	got := decodeLine(t, strings.TrimSuffix(buf.String(), "\n"))
	for _, name := range []string{
		"text", "role", "model", "tool_id", "tool_name", "tool_input", "tool_target",
		"tool_output", "is_error", "tokens", "models", "duration_ms", "duration_api_ms", "num_turns",
	} {
		if _, ok := got[name]; ok {
			t.Errorf("bare start line %q carries %q", buf.String(), name)
		}
	}
}

func TestEventUsageStatesAllSixTokenFields(t *testing.T) {
	pinClock(t, 1700000000000)

	var buf bytes.Buffer
	err := (eventWriter{w: &buf}).emit(event{
		Kind:      "usage",
		MessageID: "msg-1",
		Model:     "gpt-4o-mini",
		Tokens:    &tokenTotals{Input: 11, Output: 3},
	})
	if err != nil {
		t.Fatalf("emit: %v", err)
	}
	got := decodeLine(t, strings.TrimSuffix(buf.String(), "\n"))
	tokens, ok := got["tokens"].(map[string]any)
	if !ok {
		t.Fatalf("tokens = %#v, want a JSON object", got["tokens"])
	}
	want := map[string]float64{
		"input": 11, "output": 3, "cache_read": 0,
		"cache_write_5m": 0, "cache_write_1h": 0, "cost_usd": 0,
	}
	if len(tokens) != len(want) {
		t.Fatalf("tokens states %d fields (%v), want %d", len(tokens), tokens, len(want))
	}
	for name, v := range want {
		if got, ok := tokens[name].(float64); !ok || got != v {
			t.Errorf("tokens[%s] = %v (present=%v), want %v", name, tokens[name], ok, v)
		}
	}
}

// The result kind states its whole summary even when the run measured 0ms: a
// reader must never have to guess whether a missing duration means zero or "not
// reported" (#19), the same rule that keeps tokenTotals omitempty-free.
func TestEventResultStatesZeroDurationAndSummary(t *testing.T) {
	pinClock(t, 1700000000000)

	var buf bytes.Buffer
	err := (eventWriter{w: &buf}).emitResult(resultPayload{
		Kind:   "result",
		Tokens: tokenTotals{Input: 11, Output: 3},
		Models: []string{"m-a"},
	})
	if err != nil {
		t.Fatalf("emitResult: %v", err)
	}
	got := decodeLine(t, strings.TrimSuffix(buf.String(), "\n"))
	for _, name := range []string{"duration_ms", "duration_api_ms", "num_turns"} {
		v, ok := got[name]
		if !ok {
			t.Errorf("zero-duration result line %q omits %q", buf.String(), name)
			continue
		}
		if n, _ := v.(float64); n != 0 {
			t.Errorf("result %s = %v, want 0", name, v)
		}
	}
	if ts, _ := got["timestamp"].(float64); int64(ts) != 1700000000000 {
		t.Errorf("result timestamp = %v, want the injected clock 1700000000000", got["timestamp"])
	}
	if v, ok := got["text"]; !ok {
		t.Errorf("result line %q omits %q on an empty answer, want the empty string stated", buf.String(), "text")
	} else if s, _ := v.(string); s != "" {
		t.Errorf("result text = %v, want the empty string", v)
	}
	if _, ok := got["tokens"].(map[string]any); !ok {
		t.Errorf("result tokens = %#v, want a JSON object", got["tokens"])
	}
	if models, _ := got["models"].([]any); len(models) != 1 {
		t.Errorf("result models = %v, want the one responding id", got["models"])
	} else if models[0] != "m-a" {
		t.Errorf("result models[0] = %v, want %q", models[0], "m-a")
	}
}
