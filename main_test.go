package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"reflect"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// TestMain doubles as an escape helper: re-invoked with the env var set, it
// detaches into a new session — outside the process group a timeout SIGKILL
// targets — and sleeps holding the inherited stdout pipe. No setsid binary
// ships in this environment, so the test binary itself is the escape vehicle.
func TestMain(m *testing.M) {
	if os.Getenv("BASH_TOOL_ESCAPE_HELPER") == "1" {
		syscall.Setsid()
		time.Sleep(30 * time.Second)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func TestRunChatRoundTrip(t *testing.T) {
	// mutex: the handler goroutine writes, the test goroutine reads
	var mu sync.Mutex
	var gotReq *http.Request
	var gotRaw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		gotReq, gotRaw = r, body
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"here is the fix"}}]}`)
	}))
	defer srv.Close()

	// prompt before flags: the issue's argv shape, end to end
	var stdout, stderr bytes.Buffer
	code := run([]string{"fix the flaky test", "--model", "gpt-4o-mini", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	mu.Lock()
	req, reqRaw := gotReq, gotRaw
	mu.Unlock()
	if req == nil {
		t.Fatal("Driver sent no request to the fake server")
	}
	if req.Method != http.MethodPost {
		t.Errorf("method = %q, want POST", req.Method)
	}
	if req.URL.Path != "/chat/completions" {
		t.Errorf("path = %q, want /chat/completions", req.URL.Path)
	}
	if ct := req.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}

	var reqBody struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(reqRaw, &reqBody); err != nil {
		t.Fatalf("request body is not valid JSON: %v; body = %q", err, reqRaw)
	}
	if reqBody.Model != "gpt-4o-mini" {
		t.Errorf("model = %q, want %q", reqBody.Model, "gpt-4o-mini")
	}
	if len(reqBody.Messages) != 2 {
		t.Fatalf("messages = %d, want 2", len(reqBody.Messages))
	}
	if reqBody.Messages[0].Role != "system" || reqBody.Messages[0].Content != systemPrompt {
		t.Errorf("system message = %+v, want content %q", reqBody.Messages[0], systemPrompt)
	}
	if reqBody.Messages[1].Role != "user" || reqBody.Messages[1].Content != "fix the flaky test" {
		t.Errorf("user message = %+v, want prompt %q", reqBody.Messages[1], "fix the flaky test")
	}

	if got := resultText(t, stdout.String()); got != "here is the fix" {
		t.Errorf("result event text = %q, want exactly the response text with no added newline", got)
	}
	stderrLines := strings.Count(strings.TrimSuffix(stderr.String(), "\n"), "\n") + 1
	if stderrLines != 1 {
		t.Errorf("stderr has %d lines, want 1 progress line: %q", stderrLines, stderr.String())
	}
	if !strings.Contains(stderr.String(), "/chat/completions") {
		t.Errorf("stderr = %q, want a progress line naming the request target", stderr.String())
	}
}

// The whole shape of a single-Model-call run on the answer path: exactly four
// JSONL lines in contract order sharing one message id, closed by a result that
// reports the run's totals. A usage reader sums every usage event it sees, so a
// duplicated or missing line here breaks the run totals a reader computes.
func TestRunSingleModelCallEmitsStartTextUsageResult(t *testing.T) {
	pinSteppingClock(t, 1700000000000, 1000)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"model":"gpt-4o-mini","usage":{"prompt_tokens":11,"completion_tokens":3},"choices":[{"message":{"role":"assistant","content":"here is the fix"}}]}`)
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"fix the flaky test", "--model", "gpt-4o-mini", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	events := stdoutEvents(t, stdout.String())
	if len(events) != 4 {
		t.Fatalf("stdout carries %d event(s), want exactly 4: %q", len(events), stdout.String())
	}
	for i, want := range []string{"start", "text", "usage", "result"} {
		if got := events[i]["kind"]; got != want {
			t.Fatalf("event %d kind = %v, want %q: %q", i+1, got, want, stdout.String())
		}
	}
	// start names the call before the first byte is read, so the Driver assigns
	// the id itself and text and usage have to carry the same one.
	if got := events[0]["message_id"]; got != "msg-1" {
		t.Errorf("start message_id = %v, want %q", got, "msg-1")
	}
	for _, ev := range events[1:3] {
		if ev["message_id"] != events[0]["message_id"] {
			t.Errorf("%v event message_id = %v, want the call's %v", ev["kind"], ev["message_id"], events[0]["message_id"])
		}
	}
	for _, name := range []string{"text", "model", "tokens", "num_turns"} {
		if _, ok := events[0][name]; ok {
			t.Errorf("start event carries %q, which a start does not fill: %v", name, events[0])
		}
	}
	if events[1]["text"] != "here is the fix" {
		t.Errorf("text event text = %v, want the assistant content verbatim", events[1]["text"])
	}
	result := events[3]
	if result["text"] != "here is the fix" {
		t.Errorf("result text = %v, want the answer verbatim", result["text"])
	}
	if turns, _ := result["num_turns"].(float64); int(turns) != 1 {
		t.Errorf("result num_turns = %v, want 1", result["num_turns"])
	}
	if dur, _ := result["duration_ms"].(float64); dur <= 0 {
		t.Errorf("result duration_ms = %v, want a non-zero run duration", result["duration_ms"])
	}
	for _, ev := range events {
		if _, ok := ev["role"]; ok {
			t.Errorf("event %v carries role: role attribution belongs to subagents (#25)", ev)
		}
	}
	if _, ok := result["model_output"]; ok {
		t.Errorf("result event carries model_output, which stays unset: %v", result)
	}
}

// A loopback OpenAI-compatible stub answers well inside one millisecond, so a
// frozen clock makes every derived duration 0 — and 0 is a measurement, not an
// absence. The result event must still state both duration fields (#19), the
// way a usage event states its zero token fields.
func TestRunSubMillisecondAnswerStillStatesDurationFields(t *testing.T) {
	pinClock(t, 1700000000000)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"model":"gpt-4o-mini","usage":{"prompt_tokens":11,"completion_tokens":3},"choices":[{"message":{"role":"assistant","content":"done"}}]}`)
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "gpt-4o-mini", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	result := resultEvent(t, stdout.String())
	for _, name := range []string{"duration_ms", "duration_api_ms"} {
		v, ok := result[name]
		if !ok {
			t.Errorf("result event omits %q on a zero-duration run, want the key present with value 0: %v", name, result)
			continue
		}
		if n, _ := v.(float64); n != 0 {
			t.Errorf("result %s = %v, want 0 for a frozen clock", name, v)
		}
	}
}

// A Model that answers with no prose still owes an answer in the stream: the
// result event states text as the empty string rather than dropping the key, so
// a reader never mistakes "answered nothing" for "field not reported" — the
// same rule that keeps the duration fields on the wire at 0ms.
func TestRunEmptyFinalAnswerStatesEmptyResultText(t *testing.T) {
	srv := newScriptedChatServer(t,
		`{"model":"m-a","choices":[{"message":{"role":"assistant","content":""}}]}`,
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-a", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	text, ok := resultEvent(t, stdout.String())["text"]
	if !ok {
		t.Fatalf("result event omits %q on an empty answer, want it stated as %q: %q", "text", "", stdout.String())
	}
	if s, _ := text.(string); s != "" {
		t.Errorf("result text = %v, want the empty string", text)
	}
}

// A usage reader builds the run totals by summing every usage event, so a
// scripted run owes exactly one per Model call — not one per attempt, not a
// running total — and the result event carries their sum plus the models that
// answered.
func TestRunEmitsExactlyOneUsageEventPerModelCall(t *testing.T) {
	srv := newScriptedChatServer(t,
		`{"model":"m-a","usage":{"prompt_tokens":10,"completion_tokens":2},"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"true\"}"}}]}}]}`,
		`{"model":"m-b","usage":{"prompt_tokens":20,"completion_tokens":5},"choices":[{"message":{"role":"assistant","content":"done"}}]}`,
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	var usageEvents []map[string]any
	for _, ev := range stdoutEvents(t, stdout.String()) {
		if ev["kind"] == "usage" {
			usageEvents = append(usageEvents, ev)
		}
	}
	if len(usageEvents) != 2 {
		t.Fatalf("stdout carries %d usage event(s), want one per Model call: %q", len(usageEvents), stdout.String())
	}
	wantCall := []struct {
		messageID string
		model     string
		input     float64
		output    float64
	}{
		{"msg-1", "m-a", 10, 2},
		{"msg-2", "m-b", 20, 5},
	}
	for i, want := range wantCall {
		ev := usageEvents[i]
		if ev["message_id"] != want.messageID {
			t.Errorf("usage %d message_id = %v, want %q", i+1, ev["message_id"], want.messageID)
		}
		if ev["model"] != want.model {
			t.Errorf("usage %d model = %v, want the responding model %q", i+1, ev["model"], want.model)
		}
		tokens, _ := ev["tokens"].(map[string]any)
		if tokens["input"] != want.input || tokens["output"] != want.output {
			t.Errorf("usage %d tokens = %v, want this call's own tally input %v output %v", i+1, tokens, want.input, want.output)
		}
	}

	result := resultEvent(t, stdout.String())
	if turns, _ := result["num_turns"].(float64); int(turns) != 2 {
		t.Errorf("result num_turns = %v, want 2 Model calls", result["num_turns"])
	}
	resultTokens, _ := result["tokens"].(map[string]any)
	if resultTokens == nil || resultTokens["input"] != float64(30) || resultTokens["output"] != float64(7) {
		t.Errorf("result tokens = %v, want the calls summed (input 30, output 7)", result["tokens"])
	}
	models, _ := result["models"].([]any)
	if len(models) != 2 || models[0] != "m-a" || models[1] != "m-b" {
		t.Errorf("result models = %v, want the sorted unique ids [m-a m-b]", result["models"])
	}
}

// A plain llama.cpp endpoint answers with no usage object at all, and no model
// id either. The call still owes exactly one usage event stating all six token
// fields as zero — a reader counts Model calls from usage events, so skipping
// the quiet ones would understate every run.
func TestRunUsageEventCarriesZeroTokensWhenServerReportsNone(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"no usage here"}}]}`)
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	var usage map[string]any
	for _, ev := range stdoutEvents(t, stdout.String()) {
		if ev["kind"] == "usage" {
			if usage != nil {
				t.Fatalf("stdout carries more than one usage event: %q", stdout.String())
			}
			usage = ev
		}
	}
	if usage == nil {
		t.Fatalf("a call whose response has no usage object emitted no usage event: %q", stdout.String())
	}
	if usage["model"] != "m" {
		t.Errorf("usage model = %v, want the configured --model as fallback", usage["model"])
	}
	tokens, ok := usage["tokens"].(map[string]any)
	if !ok {
		t.Fatalf("usage tokens = %#v, want a JSON object of zeros", usage["tokens"])
	}
	if len(tokens) != 6 {
		t.Fatalf("usage tokens states %d field(s) (%v), want all six", len(tokens), tokens)
	}
	for _, name := range []string{"input", "output", "cache_read", "cache_write_5m", "cache_write_1h", "cost_usd"} {
		if v, ok := tokens[name].(float64); !ok || v != 0 {
			t.Errorf("usage tokens[%s] = %v (present=%v), want a stated zero", name, tokens[name], ok)
		}
	}
}

// The API key is env-only and never a flag: argv leaks into process listings.
// With SPINDRIFT_API_KEY unset the driver must send no Authorization header at
// all — an empty one is as wrong as a wrong one.
func TestRunSendsNoAuthorizationWithoutAPIKey(t *testing.T) {
	var mu sync.Mutex
	var gotReq *http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		gotReq = r
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	mu.Lock()
	req := gotReq
	mu.Unlock()
	if req == nil {
		t.Fatal("Driver sent no request to the fake server")
	}
	if _, ok := req.Header["Authorization"]; ok {
		t.Errorf("Authorization = %q, want the header absent when no API key is configured", req.Header.Get("Authorization"))
	}
	if !strings.Contains(usage, envAPIKey) {
		t.Errorf("usage does not document %s the way it documents %s:\n%s", envAPIKey, envBaseURL, usage)
	}
}

func TestRunSendsAuthorizationBearerWithAPIKey(t *testing.T) {
	var mu sync.Mutex
	var gotReq *http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		gotReq = r
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer srv.Close()

	// the stub is keyed by name: the key answers only for SPINDRIFT_API_KEY
	env := func(name string) (string, bool) {
		switch name {
		case envAPIKey:
			return "sk-test-123", true
		case envBaseURL:
			return "http://env:9/v1", true
		}
		return "", false
	}

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, env)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	mu.Lock()
	req := gotReq
	mu.Unlock()
	if req == nil {
		t.Fatal("Driver sent no request to the fake server")
	}
	if got := req.Header.Get("Authorization"); got != "Bearer sk-test-123" {
		t.Errorf("Authorization = %q, want %q", got, "Bearer sk-test-123")
	}
}

// scriptedChatServer returns the i-th scripted JSON response for the i-th
// POST and records every request body for later inspection.
type scriptedChatServer struct {
	mu        sync.Mutex
	responses []string
	rawBodies [][]byte
	srv       *httptest.Server
}

func newScriptedChatServer(t *testing.T, responses ...string) *scriptedChatServer {
	s := &scriptedChatServer{responses: responses}
	s.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		n := len(s.rawBodies)
		s.rawBodies = append(s.rawBodies, body)
		resp := ""
		exhausted := n >= len(s.responses)
		if !exhausted {
			resp = s.responses[n]
		}
		s.mu.Unlock()
		if exhausted {
			http.Error(w, "no scripted response left", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, resp)
	}))
	t.Cleanup(s.srv.Close)
	return s
}

type recordedMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []toolCall `json:"tool_calls"`
	ToolCallID string     `json:"tool_call_id"`
}

func (s *scriptedChatServer) requestBodies(t *testing.T) [][]byte {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([][]byte, len(s.rawBodies))
	copy(out, s.rawBodies)
	return out
}

func (s *scriptedChatServer) messageBodies(t *testing.T) [][]recordedMessage {
	t.Helper()
	var out [][]recordedMessage
	for _, raw := range s.requestBodies(t) {
		var reqBody struct {
			Messages []recordedMessage `json:"messages"`
		}
		if err := json.Unmarshal(raw, &reqBody); err != nil {
			t.Fatalf("request body is not valid JSON: %v; body = %q", err, raw)
		}
		out = append(out, reqBody.Messages)
	}
	return out
}

// statusSequenceServer answers the i-th POST with the i-th scripted HTTP
// status and, once the script runs out, with a valid chat completion carrying
// final. scriptedChatServer cannot express "fail twice, then succeed", so the
// retry tests need their own fake; it also records arrival times so backoff
// spacing can be asserted.
type statusSequenceServer struct {
	mu       sync.Mutex
	statuses []int
	arrivals []time.Time
	final    string
	srv      *httptest.Server
}

func newStatusSequenceServer(t *testing.T, final string, statuses ...int) *statusSequenceServer {
	s := &statusSequenceServer{statuses: statuses, final: final}
	s.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		s.mu.Lock()
		n := len(s.arrivals)
		s.arrivals = append(s.arrivals, time.Now())
		code := http.StatusOK
		if n < len(s.statuses) {
			code = s.statuses[n]
		}
		s.mu.Unlock()
		if code != http.StatusOK {
			http.Error(w, "scripted failure", code)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"`+final+`"}}]}`)
	}))
	t.Cleanup(s.srv.Close)
	return s
}

func (s *statusSequenceServer) postCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.arrivals)
}

// gaps returns the wait between consecutive POST arrivals.
func (s *statusSequenceServer) gaps() []time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]time.Duration, 0, len(s.arrivals))
	for i := 1; i < len(s.arrivals); i++ {
		out = append(out, s.arrivals[i].Sub(s.arrivals[i-1]))
	}
	return out
}

func TestRunAutonomousLoopFullSequence(t *testing.T) {
	// three rounds: tool call, tool call, final answer
	srv := newScriptedChatServer(t,
		`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"echo hello\"}"}}]}}]}`,
		`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_2","type":"function","function":{"name":"bash","arguments":"{\"command\":\"echo again\"}"}}]}}]}`,
		`{"choices":[{"message":{"role":"assistant","content":"all done"}}]}`,
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"do the thing", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}
	if got := resultText(t, stdout.String()); got != "all done" {
		t.Errorf("result event text = %q, want the final answer with no added newline", got)
	}

	bodies := srv.requestBodies(t)
	if len(bodies) != 3 {
		t.Fatalf("POSTs = %d, want 3", len(bodies))
	}

	// every request must advertise exactly one function tool named bash
	for i, body := range bodies {
		var parsed struct {
			Tools []toolSpec `json:"tools"`
		}
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatalf("unmarshal request %d: %v", i+1, err)
		}
		if len(parsed.Tools) != 1 || parsed.Tools[0].Type != "function" || parsed.Tools[0].Function.Name != "bash" {
			t.Errorf("request %d tools = %+v, want one function tool named bash", i+1, parsed.Tools)
		}
	}

	msgs := srv.messageBodies(t)

	// request 2 carries the assistant tool_calls message plus the tool result
	want2 := []recordedMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: "do the thing"},
		{Role: "assistant", ToolCalls: []toolCall{{ID: "call_1", Type: "function", Function: toolCallFunction{Name: "bash", Arguments: "{\"command\":\"echo hello\"}"}}}},
		{Role: "tool", Content: "hello\nexit code: 0", ToolCallID: "call_1"},
	}
	if !reflect.DeepEqual(msgs[1], want2) {
		t.Errorf("request 2 messages = %+v, want %+v", msgs[1], want2)
	}

	// request 3 carries both rounds: system, user, assistant(tool_calls), tool, assistant(tool_calls), tool
	want3 := append(append([]recordedMessage{}, want2...),
		recordedMessage{Role: "assistant", ToolCalls: []toolCall{{ID: "call_2", Type: "function", Function: toolCallFunction{Name: "bash", Arguments: "{\"command\":\"echo again\"}"}}}},
		recordedMessage{Role: "tool", Content: "again\nexit code: 0", ToolCallID: "call_2"},
	)
	if !reflect.DeepEqual(msgs[2], want3) {
		t.Errorf("request 3 messages = %+v, want %+v", msgs[2], want3)
	}

	// one progress line per POST
	stderrLines := strings.Count(strings.TrimSuffix(stderr.String(), "\n"), "\n") + 1
	if stderrLines != 3 {
		t.Errorf("stderr has %d lines, want 3 progress lines: %q", stderrLines, stderr.String())
	}
}

// A Tool Call is the only place the Driver must pair two events by tool id, and
// the pair belongs inside the Model call that asked for it: after that call's
// usage event and before the next start, so a reader can attribute the pair to
// msg-1 and never mistake a tool's output for assistant prose.
func TestRunEmitsToolCallAndToolResultPair(t *testing.T) {
	srv := newScriptedChatServer(t,
		`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"echo hello\"}"}}]}}]}`,
		`{"choices":[{"message":{"role":"assistant","content":"all done"}}]}`,
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	events := stdoutEvents(t, stdout.String())
	var kinds []string
	for _, ev := range events {
		kinds = append(kinds, fmt.Sprint(ev["kind"]))
	}
	// The first turn has no prose, so its only events are start, usage and the
	// tool pair; the pair sits between that usage and the next start.
	wantKinds := []string{"start", "usage", "tool_call", "tool_result", "start", "text", "usage", "result"}
	if !reflect.DeepEqual(kinds, wantKinds) {
		t.Fatalf("stdout kinds = %v, want %v: %q", kinds, wantKinds, stdout.String())
	}

	callEv, resultEv := events[2], events[3]
	if callEv["tool_id"] != "call_1" || resultEv["tool_id"] != "call_1" {
		t.Errorf("pair tool_id = %v / %v, want the same id call_1 on both", callEv["tool_id"], resultEv["tool_id"])
	}
	if callEv["tool_name"] != "bash" || resultEv["tool_name"] != "bash" {
		t.Errorf("pair tool_name = %v / %v, want bash on both", callEv["tool_name"], resultEv["tool_name"])
	}
	if callEv["tool_target"] != "echo hello" {
		t.Errorf("tool_call tool_target = %v, want the command the bash tool acts on", callEv["tool_target"])
	}
	input, ok := callEv["tool_input"].(map[string]any)
	if !ok || input["command"] != "echo hello" {
		t.Errorf("tool_call tool_input = %#v, want the decoded arguments object with command echo hello", callEv["tool_input"])
	}

	// tool_output is the Tool Result the Model got, byte for byte: same bounded
	// string with its exit-code line, not a re-derived summary of it.
	msgs := srv.messageBodies(t)
	if len(msgs) != 2 {
		t.Fatalf("POSTs = %d, want 2", len(msgs))
	}
	toolMsg := msgs[1][len(msgs[1])-1]
	if got, want := resultEv["tool_output"], toolMsg.Content; got != want {
		t.Errorf("tool_result tool_output = %v, want exactly the tool message content %q", got, want)
	}
	if _, ok := callEv["tool_output"]; ok {
		t.Errorf("tool_call carries tool_output, which only tool_result fills: %v", callEv)
	}
	if _, ok := resultEv["tool_input"]; ok {
		t.Errorf("tool_result carries tool_input, which only tool_call fills: %v", resultEv)
	}
}

// Absence of is_error means false on the wire, so a succeeding command must not
// state the key at all while a non-zero exit states it — both on one turn, so
// neither can be mistaken for the other's tool.
func TestRunToolResultIsErrorTracksExitCode(t *testing.T) {
	srv := newScriptedChatServer(t,
		`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_ok","type":"function","function":{"name":"bash","arguments":"{\"command\":\"echo ok\"}"}},{"id":"call_fail","type":"function","function":{"name":"bash","arguments":"{\"command\":\"exit 3\"}"}}]}}]}`,
		`{"choices":[{"message":{"role":"assistant","content":"noted"}}]}`,
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 (a failing command is an ordinary result); stderr = %q", code, stderr.String())
	}

	var order []string
	results := map[string]map[string]any{}
	for _, ev := range stdoutEvents(t, stdout.String()) {
		if ev["kind"] != "tool_result" {
			continue
		}
		id, _ := ev["tool_id"].(string)
		order = append(order, id)
		results[id] = ev
	}
	if !reflect.DeepEqual(order, []string{"call_ok", "call_fail"}) {
		t.Fatalf("tool_result order = %v, want the call order [call_ok call_fail]: %q", order, stdout.String())
	}
	if ev, ok := results["call_ok"]["is_error"]; ok {
		t.Errorf("a succeeding tool_result states is_error = %v, want the key absent (absence is false): %v", ev, results["call_ok"])
	}
	if results["call_fail"]["is_error"] != true {
		t.Errorf("exit 3 tool_result is_error = %v, want true: %v", results["call_fail"]["is_error"], results["call_fail"])
	}
}

// An unknown tool stays an ordinary Tool Result for the Model and becomes an
// is_error pair for the reader: tool_name still says what was asked for, and
// tool_target stays off the wire because the Driver never learned what it acted
// on. The event's tool_output is the same string the Model is handed.
func TestRunToolPairForUnknownTool(t *testing.T) {
	srv := newScriptedChatServer(t,
		`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_x","type":"function","function":{"name":"squiggle","arguments":"{\"command\":\"ls\"}"}}]}}]}`,
		`{"choices":[{"message":{"role":"assistant","content":"understood"}}]}`,
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 (an unknown tool must not crash the loop); stderr = %q", code, stderr.String())
	}

	events := stdoutEvents(t, stdout.String())
	callEv, resultEv := toolPair(t, events)
	if callEv["tool_name"] != "squiggle" || resultEv["tool_name"] != "squiggle" {
		t.Errorf("pair tool_name = %v / %v, want squiggle on both", callEv["tool_name"], resultEv["tool_name"])
	}
	for _, ev := range []map[string]any{callEv, resultEv} {
		if _, ok := ev["tool_target"]; ok {
			t.Errorf("%v carries tool_target for a tool that was never run, want it omitted: %v", ev["kind"], ev)
		}
	}
	if resultEv["is_error"] != true {
		t.Errorf("unknown tool tool_result is_error = %v, want true", resultEv["is_error"])
	}
	want := fmt.Sprintf("error: unknown tool %q", "squiggle")
	msgs := srv.messageBodies(t)
	toolMsg := msgs[1][len(msgs[1])-1]
	if resultEv["tool_output"] != want || toolMsg.Content != want {
		t.Errorf("tool_output = %v, tool message = %q, want both exactly %q", resultEv["tool_output"], toolMsg.Content, want)
	}
}

// Arguments that are not valid JSON cannot be decoded, so tool_input falls back
// to the raw arguments string — the malformed text is the fact worth reporting —
// and no target can be named. The parse error is the Tool Result.
func TestRunToolPairForUnparseableArguments(t *testing.T) {
	rawArgs := `{"command": "echo broken`
	argsJSON, err := json.Marshal(rawArgs)
	if err != nil {
		t.Fatalf("marshal scripted arguments: %v", err)
	}
	srv := newScriptedChatServer(t,
		fmt.Sprintf(`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_bad","type":"function","function":{"name":"bash","arguments":%s}}]}}]}`, argsJSON),
		`{"choices":[{"message":{"role":"assistant","content":"moving on"}}]}`,
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 (malformed arguments must not crash the loop); stderr = %q", code, stderr.String())
	}

	events := stdoutEvents(t, stdout.String())
	callEv, resultEv := toolPair(t, events)
	if got, ok := callEv["tool_input"].(string); !ok || got != rawArgs {
		t.Errorf("tool_call tool_input = %#v, want the raw arguments string %q", callEv["tool_input"], rawArgs)
	}
	for _, ev := range []map[string]any{callEv, resultEv} {
		if _, ok := ev["tool_target"]; ok {
			t.Errorf("%v carries tool_target for arguments that never parsed, want it omitted: %v", ev["kind"], ev)
		}
	}
	if resultEv["is_error"] != true {
		t.Errorf("unparseable arguments tool_result is_error = %v, want true", resultEv["is_error"])
	}
	msgs := srv.messageBodies(t)
	toolMsg := msgs[1][len(msgs[1])-1]
	if resultEv["tool_output"] != toolMsg.Content {
		t.Errorf("tool_output = %v, want exactly the tool message content %q", resultEv["tool_output"], toolMsg.Content)
	}
	if out, _ := resultEv["tool_output"].(string); !strings.HasPrefix(out, "error: cannot parse tool arguments: ") {
		t.Errorf("tool_output = %q, want the parse-error text the Model receives", out)
	}
}

// A server that sends no call id still owes pairs a reader can join, so the
// Driver synthesises call-<n> over the whole run; the Model-facing replay keeps
// the empty id it was given, because the synthesised key belongs to the stream.
func TestRunSynthesisesToolIDsWhenServerSendsNone(t *testing.T) {
	srv := newScriptedChatServer(t,
		`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"bash","arguments":"{\"command\":\"echo one\"}"}},{"type":"function","function":{"name":"bash","arguments":"{\"command\":\"echo two\"}"}}]}}]}`,
		`{"choices":[{"message":{"role":"assistant","content":"paired"}}]}`,
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	events := stdoutEvents(t, stdout.String())
	var kinds []string
	for _, ev := range events {
		kinds = append(kinds, fmt.Sprint(ev["kind"]))
	}
	wantKinds := []string{"start", "usage", "tool_call", "tool_result", "tool_call", "tool_result", "start", "text", "usage", "result"}
	if !reflect.DeepEqual(kinds, wantKinds) {
		t.Fatalf("stdout kinds = %v, want each tool_call immediately followed by its own tool_result (%v): %q", kinds, wantKinds, stdout.String())
	}
	for i, want := range []struct {
		toolID string
		target string
	}{
		{"call-1", "echo one"},
		{"call-2", "echo two"},
	} {
		callEv, resultEv := events[2+i*2], events[3+i*2]
		if callEv["tool_id"] != want.toolID || resultEv["tool_id"] != want.toolID {
			t.Errorf("pair %d tool_id = %v / %v, want %v on both lines", i+1, callEv["tool_id"], resultEv["tool_id"], want.toolID)
		}
		if callEv["tool_target"] != want.target || resultEv["tool_target"] != want.target {
			t.Errorf("pair %d tool_target = %v / %v, want %q on both lines", i+1, callEv["tool_target"], resultEv["tool_target"], want.target)
		}
	}
	for _, msg := range srv.messageBodies(t)[1] {
		if msg.Role == "tool" && msg.ToolCallID != "" {
			t.Errorf("replayed tool message carries tool_call_id %q, want the server's empty id replayed unchanged", msg.ToolCallID)
		}
	}
}

// The pair belongs to the Model call that asked for it: message_id is the usage
// dedup key and the reader's message boundary, so a pair stamped with the wrong
// turn would attribute a tool to a call that never requested it.
func TestRunToolPairCarriesTheMessageIDOfItsModelCall(t *testing.T) {
	srv := newScriptedChatServer(t,
		`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_a","type":"function","function":{"name":"bash","arguments":"{\"command\":\"echo one\"}"}}]}}]}`,
		`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_b","type":"function","function":{"name":"bash","arguments":"{\"command\":\"echo two\"}"}}]}}]}`,
		`{"choices":[{"message":{"role":"assistant","content":"both done"}}]}`,
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	seen := map[string][]string{}
	for _, ev := range stdoutEvents(t, stdout.String()) {
		id, _ := ev["message_id"].(string)
		seen[id] = append(seen[id], fmt.Sprint(ev["kind"]))
	}
	for id, want := range map[string][]string{
		"msg-1": {"start", "usage", "tool_call", "tool_result"},
		"msg-2": {"start", "usage", "tool_call", "tool_result"},
		"msg-3": {"start", "text", "usage"},
	} {
		if !reflect.DeepEqual(seen[id], want) {
			t.Errorf("message_id %q carries kinds %v, want %v", id, seen[id], want)
		}
	}
	// A pair that lost its message_id would land in the no-id bucket and break the
	// exact per-message kind lists above; the result event legitimately has none.
	if extra := seen[""]; len(extra) != 1 || extra[0] != "result" {
		t.Errorf("events with no message_id = %v, want only the result event", extra)
	}
}

// toolPair picks the single tool_call/tool_result pair out of a run's events and
// asserts it is one adjacent pair, so the caller reads the two lines that belong
// together rather than indexes into a stream.
func toolPair(t *testing.T, events []map[string]any) (map[string]any, map[string]any) {
	t.Helper()
	var callEv, resultEv map[string]any
	for i, ev := range events {
		switch ev["kind"] {
		case "tool_call":
			if callEv != nil {
				t.Fatalf("stdout carries more than one tool_call event: %v", events)
			}
			callEv = ev
			if i+1 >= len(events) || events[i+1]["kind"] != "tool_result" {
				t.Fatalf("tool_call is not immediately followed by its tool_result: %v", events)
			}
		case "tool_result":
			if resultEv != nil {
				t.Fatalf("stdout carries more than one tool_result event: %v", events)
			}
			resultEv = ev
		}
	}
	if callEv == nil || resultEv == nil {
		t.Fatalf("stdout carries no tool_call/tool_result pair: %v", events)
	}
	return callEv, resultEv
}

func TestRunFailingToolCommandIsOrdinaryResult(t *testing.T) {
	srv := newScriptedChatServer(t,
		`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_e","type":"function","function":{"name":"bash","arguments":"{\"command\":\"exit 7\"}"}}]}}]}`,
		`{"choices":[{"message":{"role":"assistant","content":"gave up gracefully"}}]}`,
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 (failing command must not crash); stderr = %q", code, stderr.String())
	}
	if got := resultText(t, stdout.String()); got != "gave up gracefully" {
		t.Errorf("result event text = %q, want the final answer", got)
	}
	msgs := srv.messageBodies(t)
	if len(msgs) != 2 {
		t.Fatalf("POSTs = %d, want 2", len(msgs))
	}
	tool := msgs[1][len(msgs[1])-1]
	if tool.Role != "tool" || tool.ToolCallID != "call_e" {
		t.Errorf("tool result message = %+v, want role tool with tool_call_id call_e", tool)
	}
	if !strings.Contains(tool.Content, "exit code: 7") {
		t.Errorf("tool content = %q, want it to carry the exit-code line for 7", tool.Content)
	}
}

func TestRunLoopDeliversTruncatedToolOutput(t *testing.T) {
	// Process-boundary proof: a real `seq 1 100000` generator floods the
	// pipes, and the Tool Result message the next request carries must be
	// the tail-truncated form — marker, bounded length, tail preserved,
	// exit-code line last.
	old := bashToolMaxOutput
	bashToolMaxOutput = 256
	defer func() { bashToolMaxOutput = old }()

	srv := newScriptedChatServer(t,
		`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_g","type":"function","function":{"name":"bash","arguments":"{\"command\":\"seq 1 100000\"}"}}]}}]}`,
		`{"choices":[{"message":{"role":"assistant","content":"truncation acknowledged"}}]}`,
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"flood", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 (truncation must not crash the loop); stderr = %q", code, stderr.String())
	}
	if got := resultText(t, stdout.String()); got != "truncation acknowledged" {
		t.Errorf("result event text = %q, want the final answer after the flooded turn", got)
	}
	msgs := srv.messageBodies(t)
	if len(msgs) != 2 {
		t.Fatalf("POSTs = %d, want 2", len(msgs))
	}
	tool := msgs[1][len(msgs[1])-1]
	if tool.Role != "tool" || tool.ToolCallID != "call_g" {
		t.Errorf("tool result message = %+v, want role tool with tool_call_id call_g", tool)
	}
	if len(tool.Content) > bashToolMaxOutput+len(truncatedOutputMarker(bashToolMaxOutput))+64 {
		t.Errorf("tool content = %d bytes, want it bounded by the %d-byte cap plus marker and exit-code line", len(tool.Content), bashToolMaxOutput)
	}
	if !strings.Contains(tool.Content, truncatedOutputMarker(bashToolMaxOutput)) {
		t.Errorf("tool content = %q, want the truncation marker", tool.Content)
	}
	if !strings.HasSuffix(tool.Content, "exit code: 0") {
		t.Errorf("tool content = %q, want it to end with the exit-code line", tool.Content)
	}
	if !strings.Contains(tool.Content, "99999\n100000") {
		t.Errorf("tool content = %q, want the tail of the generated output (99999, 100000) present", tool.Content)
	}
	// The event carries the very same bounded string, so a decoder never sees more
	// of a flood than the Model did.
	_, resultEv := toolPair(t, stdoutEvents(t, stdout.String()))
	if resultEv["tool_output"] != tool.Content {
		t.Errorf("tool_result tool_output = %d bytes, want the truncated tool message verbatim (%d bytes)", len(fmt.Sprint(resultEv["tool_output"])), len(tool.Content))
	}
}

func TestRunLoopContinuesAfterToolTimeout(t *testing.T) {
	// Process-boundary proof: a real `sleep 30` subprocess outlives the
	// injected short timeout, gets killed, and the Driver keeps looping —
	// the next request must carry the timeout as an ordinary Tool Result
	// message, and the scripted final answer must still be reached.
	old := bashToolTimeout
	bashToolTimeout = 200 * time.Millisecond
	defer func() { bashToolTimeout = old }()

	srv := newScriptedChatServer(t,
		`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_t","type":"function","function":{"name":"bash","arguments":"{\"command\":\"sleep 30\"}"}}]}}]}`,
		`{"choices":[{"message":{"role":"assistant","content":"recovered from timeout"}}]}`,
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"hang up", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 (timeout must not crash the loop); stderr = %q", code, stderr.String())
	}
	if got := resultText(t, stdout.String()); got != "recovered from timeout" {
		t.Errorf("result event text = %q, want the final answer after the timed-out turn", got)
	}
	msgs := srv.messageBodies(t)
	if len(msgs) != 2 {
		t.Fatalf("POSTs = %d, want 2 (loop must continue past the timeout)", len(msgs))
	}
	tool := msgs[1][len(msgs[1])-1]
	if tool.Role != "tool" || tool.ToolCallID != "call_t" {
		t.Errorf("tool result message = %+v, want role tool with tool_call_id call_t", tool)
	}
	if !strings.Contains(tool.Content, "timed out after 200ms") {
		t.Errorf("tool content = %q, want it to carry the timeout wording naming the injected duration", tool.Content)
	}
}

func TestRunTurnCap(t *testing.T) {
	srv := newScriptedChatServer(t) // server runs dry on purpose; cap must hit first
	forever := `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_x","type":"function","function":{"name":"bash","arguments":"{\"command\":\"true\"}"}}]}}]}`
	srv.responses = []string{forever, forever, forever, forever}

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--max-turns", "3", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 1 {
		t.Fatalf("exit code = %d, want 1 on turn cap; stderr = %q", code, stderr.String())
	}
	if got := oneErrorEvent(t, stdout.String()); !strings.Contains(got, "turn limit") {
		t.Errorf("error event text = %q, want the turn-limit sentence", got)
	}
	if bodies := srv.requestBodies(t); len(bodies) != 3 {
		t.Errorf("POSTs = %d, want exactly max-turns 3", len(bodies))
	}
	if !strings.Contains(stderr.String(), "turn") {
		t.Errorf("stderr = %q, want a message about exhausting the turns", stderr.String())
	}
}

// The turn cap is a non-zero exit like any other, so it owes an error event
// that states the limit and its number rather than just the count of POSTs.
func TestRunTurnCapEmitsErrorEventNamingTheLimit(t *testing.T) {
	srv := newScriptedChatServer(t)
	forever := `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_x","type":"function","function":{"name":"bash","arguments":"{\"command\":\"true\"}"}}]}}]}`
	srv.responses = []string{forever, forever, forever}

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--max-turns", "2", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 1 {
		t.Fatalf("exit code = %d, want 1 on turn cap; stderr = %q", code, stderr.String())
	}
	if got := oneErrorEvent(t, stdout.String()); got != "reached the turn limit of 2 without a final answer" {
		t.Errorf("error event text = %q, want the turn-limit sentence naming the limit 2", got)
	}
	if !strings.Contains(stderr.String(), "reached the turn limit of 2 without a final answer") {
		t.Errorf("stderr = %q, want the turn-limit line as today", stderr.String())
	}
}

func TestRunRetriesServerErrorThenSucceeds(t *testing.T) {
	oldAttempts, oldBackoff := chatRetryAttempts, chatRetryBackoff
	chatRetryAttempts, chatRetryBackoff = 3, time.Millisecond
	defer func() { chatRetryAttempts, chatRetryBackoff = oldAttempts, oldBackoff }()

	srv := newStatusSequenceServer(t, "recovered after retries", 500, 500)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 after the transient 5xxs; stderr = %q", code, stderr.String())
	}
	if got := resultText(t, stdout.String()); got != "recovered after retries" {
		t.Errorf("result event text = %q, want exactly the final answer", got)
	}
	if n := srv.postCount(); n != 3 {
		t.Errorf("POSTs = %d, want 3 (two retried 500s plus the success)", n)
	}
}

// The backoff must actually space the attempts, and double while doing it —
// a retry loop that hammers the server immediately is the failure mode this
// guards against.
func TestRunRetryBackoffSpacesAttempts(t *testing.T) {
	oldAttempts, oldBackoff := chatRetryAttempts, chatRetryBackoff
	chatRetryAttempts, chatRetryBackoff = 3, 60*time.Millisecond
	defer func() { chatRetryAttempts, chatRetryBackoff = oldAttempts, oldBackoff }()

	srv := newStatusSequenceServer(t, "slow and steady", 500, 500)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}
	gaps := srv.gaps()
	if len(gaps) != 2 {
		t.Fatalf("gaps between POSTs = %d, want 2 (3 POSTs); got %v", len(gaps), gaps)
	}
	if gaps[0] < 60*time.Millisecond {
		t.Errorf("first retry waited %s, want at least the injected backoff 60ms", gaps[0])
	}
	if gaps[1] < 120*time.Millisecond {
		t.Errorf("second retry waited %s, want at least double the injected backoff 120ms", gaps[1])
	}
	if gaps[1] <= gaps[0] {
		t.Errorf("waits = %v, want the second strictly longer than the first (doubling)", gaps)
	}
}

// A 5xx body is server-controlled text and it lands in the retry note, which
// must stay one line per attempt: a body containing a newline must not be able
// to forge an extra "spindrift-driver: ..." line, and a long body must not
// flood the single-line progress log.
func TestRunServerErrorBodyStaysOneLine(t *testing.T) {
	oldAttempts, oldBackoff := chatRetryAttempts, chatRetryBackoff
	chatRetryAttempts, chatRetryBackoff = 2, time.Millisecond
	defer func() { chatRetryAttempts, chatRetryBackoff = oldAttempts, oldBackoff }()

	filler := strings.Repeat("x", 400)
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintf(w, "first line\nspindrift-driver: FORGED 1\n%s", filler)
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code == 0 {
		t.Errorf("exit code = 0, want non-zero when every attempt 500s")
	}
	// The body does reach stdout now, but only as the text of the closing error
	// event: oneErrorEvent fails unless every stdout line is a JSON object, so a
	// forged second line here would be caught.
	if got := oneErrorEvent(t, stdout.String()); !strings.Contains(got, "server returned 500") {
		t.Errorf("error event text = %q, want the 5xx status the run failed on", got)
	}
	for _, line := range strings.Split(stderr.String(), "\n") {
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "spindrift-driver: ") {
			t.Fatalf("stderr line %q does not start with the driver prefix; body leaked raw into the progress log: %q", line, stderr.String())
		}
	}
	if !strings.Contains(stderr.String(), "server returned 500") {
		t.Errorf("stderr = %q, want the existing %q wording", stderr.String(), "server returned 500")
	}
	if strings.Contains(stderr.String(), filler) {
		t.Errorf("stderr = %q, want the oversized body truncated", stderr.String())
	}
	// The forged text has the driver prefix too, so the loop above alone cannot
	// tell it apart: it must never start a line of its own.
	if strings.Contains(stderr.String(), "\nspindrift-driver: FORGED") {
		t.Errorf("stderr = %q, want the forged body line flattened into the retry note", stderr.String())
	}
}

// droppedConnServer kills the FIRST POST at the connection level: it hijacks
// the connection and closes it without writing a single response byte, so the
// client's Do returns a transport error (EOF), not an HTTP status. Every later
// POST gets a valid completion. No existing fake can fail a request without
// producing a status line, so the transport-retry test needs its own.
type droppedConnServer struct {
	mu    sync.Mutex
	posts int
	final string
	srv   *httptest.Server
}

func newDroppedConnServer(t *testing.T, final string) *droppedConnServer {
	s := &droppedConnServer{final: final}
	s.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		s.mu.Lock()
		s.posts++
		first := s.posts == 1
		s.mu.Unlock()
		if first {
			hj, ok := w.(http.Hijacker)
			if !ok {
				t.Errorf("test server is not hijackable, cannot drop the connection")
				http.Error(w, "not hijackable", http.StatusNotImplemented)
				return
			}
			conn, _, err := hj.Hijack()
			if err != nil {
				t.Errorf("hijack: %v", err)
				return
			}
			conn.Close()
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"`+final+`"}}]}`)
	}))
	t.Cleanup(s.srv.Close)
	return s
}

func (s *droppedConnServer) postCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.posts
}

// A dead connection is the classic transient failure: there is no status code
// to branch on, only the error Do returns, and the retried attempt must carry
// the same body — the one-shot reader over the marshalled request means a
// request cannot simply be resent.
func TestRunRetriesDroppedConnectionThenSucceeds(t *testing.T) {
	oldAttempts, oldBackoff := chatRetryAttempts, chatRetryBackoff
	chatRetryAttempts, chatRetryBackoff = 3, time.Millisecond
	defer func() { chatRetryAttempts, chatRetryBackoff = oldAttempts, oldBackoff }()

	srv := newDroppedConnServer(t, "recovered after dropped conn")

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 after the dropped connection; stderr = %q", code, stderr.String())
	}
	if got := resultText(t, stdout.String()); got != "recovered after dropped conn" {
		t.Errorf("result event text = %q, want exactly the final answer", got)
	}
	// One POST per attempt: the dropped request never reached a response, and
	// Go's own transparent rewind-retry only applies to reused connections.
	if n := srv.postCount(); n != 2 {
		t.Errorf("POSTs = %d, want 2 (the dropped one plus the success)", n)
	}
	if !strings.Contains(stderr.String(), "retry") {
		t.Errorf("stderr = %q, want the retry to be reported to the operator", stderr.String())
	}
}

func TestRunServerErrorStatus(t *testing.T) {
	// this fake never recovers, so cap the attempts and shorten the backoff or
	// the suite would pay the full default retry schedule
	oldAttempts, oldBackoff := chatRetryAttempts, chatRetryBackoff
	chatRetryAttempts, chatRetryBackoff = 2, time.Millisecond
	defer func() { chatRetryAttempts, chatRetryBackoff = oldAttempts, oldBackoff }()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not found", http.StatusInternalServerError)
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"--model", "m", "--base-url", srv.URL, "p"}, nil, &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code == 0 {
		t.Fatal("non-200 response should exit non-zero")
	}
	if got := oneErrorEvent(t, stdout.String()); !strings.Contains(got, "500") {
		t.Errorf("error event text = %q, want the failing status", got)
	}
	if !strings.Contains(stderr.String(), "500") {
		t.Errorf("stderr = %q, want the failing status", stderr.String())
	}
}

// A transport failure is the same error on both surfaces: the operator's
// stderr line and the machine-readable event must carry the identical raw text,
// not a summarised version of it.
func TestRunUnreachableEmitsRawDialErrorEvent(t *testing.T) {
	oldAttempts, oldBackoff := chatRetryAttempts, chatRetryBackoff
	chatRetryAttempts, chatRetryBackoff = 2, time.Millisecond
	defer func() { chatRetryAttempts, chatRetryBackoff = oldAttempts, oldBackoff }()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	deadURL := srv.URL
	srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", deadURL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 1 {
		t.Fatalf("exit code = %d, want 1 when the endpoint refuses connections; stderr = %q", code, stderr.String())
	}
	got := oneErrorEvent(t, stdout.String())
	if !strings.Contains(got, "dial tcp") {
		t.Errorf("error event text = %q, want the raw dial failure", got)
	}
	if !strings.Contains(stderr.String(), got) {
		t.Errorf("stderr = %q does not contain the event text %q, want one error on both surfaces", stderr.String(), got)
	}
}

// The endpoint must be dead in a way that is certain on any host: a closed
// httptest.Server hands back a port that just refused connections a moment
// ago, whereas a hardcoded address such as 127.0.0.1:1 may be filtered (a
// hang, not a refusal) or already in use.
func TestRunExitsNonZeroWhenEndpointUnreachable(t *testing.T) {
	oldAttempts, oldBackoff := chatRetryAttempts, chatRetryBackoff
	chatRetryAttempts, chatRetryBackoff = 3, time.Millisecond
	defer func() { chatRetryAttempts, chatRetryBackoff = oldAttempts, oldBackoff }()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("dead endpoint received POST %s", r.URL.Path)
	}))
	deadURL := srv.URL
	srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", deadURL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code == 0 {
		t.Fatalf("exit code = 0, want non-zero when the endpoint refuses connections; stderr = %q", stderr.String())
	}
	if got := oneErrorEvent(t, stdout.String()); !strings.Contains(got, "connection refused") {
		t.Errorf("error event text = %q, want the connection failure", got)
	}
	if !strings.Contains(stderr.String(), "spindrift-driver:") || !strings.Contains(strings.ToLower(stderr.String()), "connection refused") {
		t.Errorf("stderr = %q, want the connection failure reported to the operator", stderr.String())
	}
	if !strings.Contains(stderr.String(), deadURL) {
		t.Errorf("stderr = %q, want the failing endpoint %s named", stderr.String(), deadURL)
	}
}

// The POST count is the discriminating assertion here: too few means the loop
// abandoned a retry it still had budget for, too many means it cannot stop.
func TestRunGivesUpAfterRetryBudgetExhausted(t *testing.T) {
	oldAttempts, oldBackoff := chatRetryAttempts, chatRetryBackoff
	chatRetryAttempts, chatRetryBackoff = 4, time.Millisecond
	defer func() { chatRetryAttempts, chatRetryBackoff = oldAttempts, oldBackoff }()

	var mu sync.Mutex
	posts := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		mu.Lock()
		posts++
		mu.Unlock()
		http.Error(w, "upstream exploded", http.StatusBadGateway)
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code == 0 {
		t.Fatalf("exit code = 0, want non-zero once the retry budget is spent; stderr = %q", stderr.String())
	}
	if got := oneErrorEvent(t, stdout.String()); !strings.Contains(got, "server returned 502") {
		t.Errorf("error event text = %q, want the last failing status", got)
	}
	if posts != 4 {
		t.Errorf("POSTs = %d, want exactly 4 (the full injected budget, no more)", posts)
	}
	if !strings.Contains(stderr.String(), "server returned 502") {
		t.Errorf("stderr = %q, want the last failing status in the existing server-returned wording", stderr.String())
	}
	if !strings.Contains(stderr.String(), "retry 4/4") {
		t.Errorf("stderr = %q, want the final attempt announced as retry 4/4", stderr.String())
	}
}

// A non-retryable rejection's body is the answer, so the error event carries it
// verbatim — newlines included. boundBody must never touch event text: JSON
// escaping is what keeps the multi-line body inside one stdout line.
func TestRunRejectionBodyStaysVerbatimInOneErrorEvent(t *testing.T) {
	const body = "context length exceeded\n\tset --max-turns lower\nend of rejection"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, body)
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 1 {
		t.Fatalf("exit code = %d, want 1 on a non-retryable rejection; stderr = %q", code, stderr.String())
	}
	want := "server returned 400 Bad Request: " + body
	if got := oneErrorEvent(t, stdout.String()); got != want {
		t.Errorf("error event text = %q, want the rejection verbatim %q", got, want)
	}
}

// Retry scope is narrow: a transport error plus 5xx only. The injected budget
// is generous, so these permanent answers must still cost exactly one POST --
// any retry at all shows up as an extra POST, not as a different exit code.
func TestRunPermanentFailuresAreNotRetried(t *testing.T) {
	oldAttempts, oldBackoff := chatRetryAttempts, chatRetryBackoff
	chatRetryAttempts, chatRetryBackoff = 4, time.Millisecond
	defer func() { chatRetryAttempts, chatRetryBackoff = oldAttempts, oldBackoff }()

	cases := []struct {
		name       string
		status     int
		body       string
		wantStderr string
	}{
		{"4xx", http.StatusBadRequest, "bad request", "server returned 400"},
		{"unparseable 200", http.StatusOK, "this is not json", "unparseable"},
		{"empty choices", http.StatusOK, `{"choices":[]}`, "choices"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			posts := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				io.Copy(io.Discard, r.Body)
				mu.Lock()
				posts++
				mu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.status)
				io.WriteString(w, tc.body)
			}))
			defer srv.Close()

			var stdout, stderr bytes.Buffer
			code := run([]string{"p", "--model", "m", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
			if code == 0 {
				t.Fatalf("exit code = 0, want non-zero; stderr = %q", stderr.String())
			}
			eventText := oneErrorEvent(t, stdout.String())
			if !strings.Contains(eventText, tc.wantStderr) {
				t.Errorf("error event text = %q, want it to mention %q", eventText, tc.wantStderr)
			}
			mu.Lock()
			n := posts
			mu.Unlock()
			if n != 1 {
				t.Errorf("POSTs = %d, want exactly 1 (no retry for a permanent failure)", n)
			}
			if !strings.Contains(stderr.String(), tc.wantStderr) {
				t.Errorf("stderr = %q, want it to mention %q", stderr.String(), tc.wantStderr)
			}
		})
	}
}

// driverErrorLine returns the terminal diagnostic run() prints when sendChat
// gives up: the per-attempt "POST" note and the "retry" notes are separate
// lines, so an operator reading only this line must learn which endpoint hit
// which bound.
func driverErrorLine(stderr, target string) string {
	postNote := "spindrift-driver: POST " + target
	lines := strings.Split(stderr, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := lines[i]
		if !strings.HasPrefix(line, "spindrift-driver:") {
			continue
		}
		if line == postNote || strings.HasPrefix(line, "spindrift-driver: retry ") {
			continue
		}
		return line
	}
	return ""
}

// assertTimeoutErrorLine pins the terminal diagnostic of a timed-out run to the
// endpoint and the per-attempt bound that fired: the raw Go error ("context
// deadline exceeded (Client.Timeout exceeded while awaiting headers)") names
// neither, so both must be added. The caller passes the bound it injected, so
// each is checked against the timeout it actually configured.
func assertTimeoutErrorLine(t *testing.T, stderr, target string, bound time.Duration) {
	t.Helper()
	if got := driverErrorLine(stderr, target); !strings.Contains(got, target) || !strings.Contains(got, fmt.Sprint(bound)) {
		t.Errorf("error line = %q, want it to name the endpoint %s and the %s bound that fired", got, target, bound)
	}
}

// A stalled endpoint is the black hole this test pins: the connection is
// accepted and nothing ever answers, so only a bounded per-attempt timeout can
// turn it into a non-zero exit. The elapsed bound is the load-bearing
// assertion -- without a timeout the run simply waits for the stall to end and
// exits 0, which no other assertion here can distinguish from success.
func TestRunStalledRequestTimesOut(t *testing.T) {
	oldTimeout, oldAttempts, oldBackoff := chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff
	chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = 50*time.Millisecond, 2, time.Millisecond
	defer func() {
		chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = oldTimeout, oldAttempts, oldBackoff
	}()

	const stall = 2 * time.Second
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		// the ctx branch fires as soon as the client hangs up, so a handler that
		// outlived its request cannot make the deferred Close() stall the suite
		select {
		case <-time.After(stall):
		case <-r.Context().Done():
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"too late"}}]}`)
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	start := time.Now()
	code := run([]string{"p", "--model", "m", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	elapsed := time.Since(start)

	if elapsed > time.Second {
		t.Errorf("run took %s, want well under the handler's %s stall (no per-attempt timeout)", elapsed, stall)
	}
	if code == 0 {
		t.Fatalf("exit code = 0, want non-zero once every attempt times out; elapsed = %s, stderr = %q", elapsed, stderr.String())
	}
	if got := oneErrorEvent(t, stdout.String()); !strings.Contains(got, "timed out") {
		t.Errorf("error event text = %q, want the attempt timeout", got)
	}
	if !strings.Contains(stderr.String(), "Client.Timeout exceeded") && !strings.Contains(strings.ToLower(stderr.String()), "timeout") {
		t.Errorf("stderr = %q, want the attempt timeout reported to the operator", stderr.String())
	}
	assertTimeoutErrorLine(t, stderr.String(), srv.URL+"/chat/completions", 50*time.Millisecond)
}

// A server that answers with headers and then dies mid-body is the same failure
// mode as never answering at all: the deadline fires inside the body read, so
// that diagnostic has to name the endpoint and the bound too.
func TestRunStalledResponseBodyTimesOut(t *testing.T) {
	oldTimeout, oldAttempts, oldBackoff := chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff
	chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = 50*time.Millisecond, 2, time.Millisecond
	defer func() {
		chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = oldTimeout, oldAttempts, oldBackoff
	}()

	const stall = 2 * time.Second
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		// headers plus a prefix of the body reach the client, then the stream
		// stops: the read, not the round trip, is what has to expire
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"`)
		w.(http.Flusher).Flush()
		select {
		case <-time.After(stall):
		case <-r.Context().Done():
			return
		}
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	start := time.Now()
	code := run([]string{"p", "--model", "m", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	elapsed := time.Since(start)

	if elapsed > time.Second {
		t.Errorf("run took %s, want well under the handler's %s stall (no per-attempt timeout on the body read)", elapsed, stall)
	}
	if code == 0 {
		t.Fatalf("exit code = 0, want non-zero once every attempt times out; elapsed = %s, stderr = %q", elapsed, stderr.String())
	}
	if got := oneErrorEvent(t, stdout.String()); !strings.Contains(got, "while reading the response") {
		t.Errorf("error event text = %q, want the response-reading leg named", got)
	}
	assertTimeoutErrorLine(t, stderr.String(), srv.URL+"/chat/completions", 50*time.Millisecond)
	// the deadline fired in the body read, so the diagnostic must name that leg
	// exactly once instead of nesting a second clause inside the reworded one
	if got := driverErrorLine(stderr.String(), srv.URL+"/chat/completions"); !strings.Contains(got, "while reading the response") || strings.Contains(got, "read response:") {
		t.Errorf("error line = %q, want it to name the response-reading leg once without nesting %q", got, "read response:")
	}
}

// A connection torn down mid-body is flakiness rather than a deadline, and the
// pre-timeout diagnostic named the failing leg ("read response:"); losing that
// would leave the operator with a bare "unexpected EOF".
func TestRunMidBodyTransportErrorNamesReadLeg(t *testing.T) {
	oldTimeout, oldAttempts, oldBackoff := chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff
	chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = 2*time.Second, 1, time.Millisecond
	defer func() {
		chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = oldTimeout, oldAttempts, oldBackoff
	}()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Length", "200")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"choices":[`)
		conn, _, err := w.(http.Hijacker).Hijack()
		if err == nil {
			conn.Close()
		}
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })

	if code == 0 {
		t.Fatalf("exit code = 0, want non-zero when the response body is cut short; stderr = %q", stderr.String())
	}
	if got := oneErrorEvent(t, stdout.String()); !strings.Contains(got, "read response:") {
		t.Errorf("error event text = %q, want the read-response leg named", got)
	}
	got := driverErrorLine(stderr.String(), srv.URL+"/chat/completions")
	if !strings.Contains(got, "read response:") {
		t.Errorf("error line = %q, want the read-response leg named", got)
	}
	if strings.Contains(got, "timed out") {
		t.Errorf("error line = %q, want the transport cause reported, not a timeout this failure is not", got)
	}
}

// The config value, not the package default, must bound the client: the package
// var here is far longer than the stall, so a run that finishes early can only
// be explained by --timeout reaching the http.Client.
func TestRunConfiguredTimeoutOverridesPackageDefault(t *testing.T) {
	oldTimeout, oldAttempts, oldBackoff := chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff
	chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = 30*time.Second, 2, time.Millisecond
	defer func() {
		chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = oldTimeout, oldAttempts, oldBackoff
	}()

	const stall = 2 * time.Second
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		select {
		case <-time.After(stall):
		case <-r.Context().Done():
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"too late"}}]}`)
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	start := time.Now()
	code := run([]string{"p", "--model", "m", "--base-url", srv.URL, "--timeout", "100ms"}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	elapsed := time.Since(start)

	if elapsed > time.Second {
		t.Errorf("run took %s, want the 100ms --timeout to bound it well before the %s package default", elapsed, 30*time.Second)
	}
	if code == 0 {
		t.Fatalf("exit code = 0, want non-zero once every attempt times out; elapsed = %s, stderr = %q", elapsed, stderr.String())
	}
}

func TestSendChatHandBuiltConfigFallsBackToPackageTimeout(t *testing.T) {
	oldTimeout, oldAttempts, oldBackoff := chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff
	chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = 50*time.Millisecond, 1, time.Millisecond
	defer func() {
		chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = oldTimeout, oldAttempts, oldBackoff
	}()

	const stall = 2 * time.Second
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		select {
		case <-time.After(stall):
		case <-r.Context().Done():
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"too late"}}]}`)
	}))
	defer srv.Close()

	// A hand-built config leaves cfg.timeout zero, and http.Client{Timeout: 0}
	// means no bound at all rather than no wait — the attempt must still land
	// on the finite package default.
	start := time.Now()
	reply, err := sendChat(config{baseURL: srv.URL, model: "m"}, []chatMessage{{Role: "user", Content: "p"}}, &bytes.Buffer{})
	elapsed := time.Since(start)

	if err == nil {
		t.Fatalf("sendChat = (%+v, nil), want the attempt to time out", reply)
	}
	if elapsed > time.Second {
		t.Errorf("sendChat took %s, want it bounded by the %s package default, well under the %s stall", elapsed, chatAttemptTimeout, stall)
	}
	if !strings.Contains(err.Error(), srv.URL) {
		t.Errorf("error = %q, want it to name the endpoint %s", err, srv.URL)
	}
	if !strings.Contains(err.Error(), fmt.Sprint(chatAttemptTimeout)) {
		t.Errorf("error = %q, want it to name the bound %s", err, chatAttemptTimeout)
	}
}

// One Model call's usage event needs its data source to survive the round
// trip: the response's model and its OpenAI-shaped usage.
func TestSendChatReturnsModelAndTokens(t *testing.T) {
	srv := newScriptedChatServer(t, `{"id":"chatcmpl-9","model":"llama-3.1-8b","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":120,"completion_tokens":34}}`)

	reply, err := sendChat(config{baseURL: srv.srv.URL, model: "configured-model"}, []chatMessage{{Role: "user", Content: "p"}}, &bytes.Buffer{})
	if err != nil {
		t.Fatalf("sendChat: %v", err)
	}
	if reply.Message.Content != "ok" {
		t.Errorf("Message = %+v, want the assistant content to survive", reply.Message)
	}
	if reply.Model != "llama-3.1-8b" {
		t.Errorf("Model = %q, want the response model %q", reply.Model, "llama-3.1-8b")
	}
	want := tokenTotals{Input: 120, Output: 34}
	if reply.Tokens != want {
		t.Errorf("Tokens = %+v, want %+v", reply.Tokens, want)
	}
}

// A server that sends no usage object at all still owes the call its usage
// event, so sendChat answers zero tokens — never an error, because a plain
// llama.cpp endpoint would otherwise fail every run (contract rule 10).
func TestSendChatWithoutUsageObjectReturnsZeroTokens(t *testing.T) {
	srv := newScriptedChatServer(t, `{"model":"llama-3.1-8b","choices":[{"message":{"role":"assistant","content":"ok"}}]}`)

	reply, err := sendChat(config{baseURL: srv.srv.URL, model: "configured-model"}, []chatMessage{{Role: "user", Content: "p"}}, &bytes.Buffer{})
	if err != nil {
		t.Fatalf("sendChat: %v", err)
	}
	if reply.Message.Content != "ok" {
		t.Errorf("Message = %+v, want the assistant content despite the missing usage object", reply.Message)
	}
	if reply.Tokens != (tokenTotals{}) {
		t.Errorf("Tokens = %+v, want all zeros", reply.Tokens)
	}
}

// Cache read has two spellings on the wire; llama.cpp and vLLM report it flat
// on the usage object.
func TestSendChatMapsPromptCacheHitTokensToCacheRead(t *testing.T) {
	srv := newScriptedChatServer(t, `{"model":"llama-3.1-8b","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":200,"completion_tokens":9,"prompt_cache_hit_tokens":80}}`)

	reply, err := sendChat(config{baseURL: srv.srv.URL, model: "configured-model"}, []chatMessage{{Role: "user", Content: "p"}}, &bytes.Buffer{})
	if err != nil {
		t.Fatalf("sendChat: %v", err)
	}
	want := tokenTotals{Input: 200, Output: 9, CacheRead: 80}
	if reply.Tokens != want {
		t.Errorf("Tokens = %+v, want %+v", reply.Tokens, want)
	}
}

// OpenAI spells the same number inside prompt_tokens_details instead.
func TestSendChatMapsOpenAIStyleCachedTokensToCacheRead(t *testing.T) {
	srv := newScriptedChatServer(t, `{"model":"gpt-4o-mini","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":200,"completion_tokens":9,"prompt_tokens_details":{"cached_tokens":64}}}`)

	reply, err := sendChat(config{baseURL: srv.srv.URL, model: "configured-model"}, []chatMessage{{Role: "user", Content: "p"}}, &bytes.Buffer{})
	if err != nil {
		t.Fatalf("sendChat: %v", err)
	}
	want := tokenTotals{Input: 200, Output: 9, CacheRead: 64}
	if reply.Tokens != want {
		t.Errorf("Tokens = %+v, want %+v", reply.Tokens, want)
	}
}

// The response model wins when present; when the server omits it or sends it
// empty, the configured --model stands in so no usage event goes out
// model-less.
func TestSendChatWithoutModelFallsBackToConfiguredModel(t *testing.T) {
	for _, body := range []string{
		`{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`,
		`{"model":"","choices":[{"message":{"role":"assistant","content":"ok"}}]}`,
	} {
		srv := newScriptedChatServer(t, body)
		reply, err := sendChat(config{baseURL: srv.srv.URL, model: "configured-model"}, []chatMessage{{Role: "user", Content: "p"}}, &bytes.Buffer{})
		if err != nil {
			t.Fatalf("sendChat(%s): %v", body, err)
		}
		if reply.Model != "configured-model" {
			t.Errorf("Model = %q for body %s, want the configured model %q", reply.Model, body, "configured-model")
		}
	}
}

func TestRunEmptyResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"choices":[]}`)
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"--model", "m", "--base-url", srv.URL, "p"}, nil, &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code == 0 {
		t.Fatal("empty choices should exit non-zero")
	}
	if got := oneErrorEvent(t, stdout.String()); got != "server returned no choices" {
		t.Errorf("error event text = %q, want %q", got, "server returned no choices")
	}
	if !strings.Contains(stderr.String(), "choices") {
		t.Errorf("stderr = %q, want a message about the empty response", stderr.String())
	}
}

// Every non-zero exit owes exactly one error event on stdout, usage errors
// included: stdout is a machine surface, so a decoder must be able to learn the
// reason without parsing the human usage block off stderr. The text is the raw
// parse error — no driver prefix, no wrapping.
func TestRunUsageErrorEmitsOneErrorEvent(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run([]string{"say hi"}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 2 {
		t.Fatalf("exit code = %d, want 2 for a usage error; stderr = %q", code, stderr.String())
	}
	if got := oneErrorEvent(t, stdout.String()); got != "--model is required" {
		t.Errorf("error event text = %q, want the raw parse error %q", got, "--model is required")
	}
	if !strings.Contains(stderr.String(), "usage:") {
		t.Errorf("stderr = %q, want the usage block as today", stderr.String())
	}
}

func TestRunMissingModel(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run([]string{"say hi"}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code == 0 {
		t.Fatal("run with missing --model should exit non-zero")
	}
	if got := oneErrorEvent(t, stdout.String()); got != "--model is required" {
		t.Errorf("error event text = %q, want %q", got, "--model is required")
	}
	if !strings.Contains(stderr.String(), "usage") {
		t.Errorf("stderr = %q, want usage text", stderr.String())
	}
}

func TestRunEmptyPrompt(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run([]string{"--model", "m"}, strings.NewReader("   \n"), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code == 0 {
		t.Fatal("run with empty prompt should exit non-zero")
	}
	if got := oneErrorEvent(t, stdout.String()); !strings.Contains(got, "prompt is required") {
		t.Errorf("error event text = %q, want the prompt requirement", got)
	}
	if !strings.Contains(stderr.String(), "usage") {
		t.Errorf("stderr = %q, want usage text", stderr.String())
	}
}

func TestParseArgsFlagsAfterPositional(t *testing.T) {
	// The issue's argv shape is: spindrift-driver "prompt" --model m —
	// positional first, but flag.Parse stops at the first positional.
	cfg, err := parseArgs([]string{"fix the flaky test", "--model", "gpt-4o-mini", "--base-url", "http://x:1/v1"}, nil, func(string) (string, bool) { return "", false })
	if err != nil {
		t.Fatalf("parseArgs returned error: %v", err)
	}
	if cfg.prompt != "fix the flaky test" || cfg.model != "gpt-4o-mini" || cfg.baseURL != "http://x:1/v1" {
		t.Errorf("cfg = %+v, want prompt/model/base-url all parsed", cfg)
	}

	// =-form flags must not swallow the following positional.
	cfg, err = parseArgs([]string{"--model=gpt-4o-mini", "another prompt"}, nil, func(string) (string, bool) { return "", false })
	if err != nil {
		t.Fatalf("parseArgs (= form) returned error: %v", err)
	}
	if cfg.prompt != "another prompt" || cfg.model != "gpt-4o-mini" {
		t.Errorf("cfg = %+v, want prompt and model from = form", cfg)
	}
}

func TestParseArgsEndOfFlags(t *testing.T) {
	// Everything after a standalone -- is positional, even if it looks like
	// a flag; without this there is no way to pass a dash-leading prompt.
	for _, tc := range []struct {
		args []string
		want string
	}{
		{[]string{"--model", "m", "--", "-weird"}, "-weird"},
	} {
		cfg, err := parseArgs(tc.args, nil, func(string) (string, bool) { return "", false })
		if err != nil {
			t.Errorf("parseArgs(%q) returned error: %v", tc.args, err)
			continue
		}
		if cfg.prompt != tc.want {
			t.Errorf("parseArgs(%q) prompt = %q, want %q", tc.args, cfg.prompt, tc.want)
		}
		if cfg.model != "m" {
			t.Errorf("parseArgs(%q) model = %q, want %q", tc.args, cfg.model, "m")
		}
	}

	// With no flags at all, -- still protects the prompt: the failure is the
	// missing model, not "flag provided but not defined: -weird".
	_, err := parseArgs([]string{"--", "-weird"}, nil, func(string) (string, bool) { return "", false })
	if err == nil || !strings.Contains(err.Error(), "--model is required") {
		t.Errorf("parseArgs([-- -weird]) err = %v, want missing-model (not a flag error)", err)
	}
}

func TestParseArgsStdinFallback(t *testing.T) {
	cfg, err := parseArgs([]string{"--model", "m"}, strings.NewReader("  explain the makefile \n"), func(string) (string, bool) { return "", false })
	if err != nil {
		t.Fatalf("parseArgs returned error: %v", err)
	}
	if cfg.prompt != "explain the makefile" {
		t.Errorf("prompt = %q, want stdin content trimmed", cfg.prompt)
	}
}

func TestParseArgsBaseURLPrecedence(t *testing.T) {
	env := func(string) (string, bool) { return "http://env:9/v1", true }

	// flag beats env var
	cfg, err := parseArgs([]string{"--model", "m", "--base-url", "http://flag:1/v1", "p"}, nil, env)
	if err != nil || cfg.baseURL != "http://flag:1/v1" {
		t.Errorf("flag override: cfg = %+v, err = %v", cfg, err)
	}

	// env var beats default
	cfg, err = parseArgs([]string{"--model", "m", "p"}, nil, env)
	if err != nil || cfg.baseURL != "http://env:9/v1" {
		t.Errorf("env override: cfg = %+v, err = %v", cfg, err)
	}

	// default when neither
	cfg, err = parseArgs([]string{"--model", "m", "p"}, nil, func(string) (string, bool) { return "", false })
	if err != nil || cfg.baseURL != "http://localhost:8080/v1" {
		t.Errorf("default: cfg = %+v, err = %v", cfg, err)
	}
}

func TestParseArgsPositionalPrompt(t *testing.T) {
	args := []string{"--model", "gpt-4o-mini", "fix the flaky test"}
	cfg, err := parseArgs(args, nil, func(string) (string, bool) { return "", false })
	if err != nil {
		t.Fatalf("parseArgs returned error: %v", err)
	}
	if cfg.model != "gpt-4o-mini" {
		t.Errorf("model = %q, want %q", cfg.model, "gpt-4o-mini")
	}
	if cfg.prompt != "fix the flaky test" {
		t.Errorf("prompt = %q, want %q", cfg.prompt, "fix the flaky test")
	}
	if cfg.baseURL != "http://localhost:8080/v1" {
		t.Errorf("baseURL = %q, want default %q", cfg.baseURL, "http://localhost:8080/v1")
	}
}

func TestParseArgsStdinReadError(t *testing.T) {
	_, err := parseArgs([]string{"--model", "m"}, io.Reader(failingReader{}), func(string) (string, bool) { return "", false })
	if err == nil {
		t.Fatal("parseArgs should fail when stdin cannot be read")
	}
	if strings.Contains(err.Error(), "prompt is required") {
		t.Errorf("err = %v, want the stdin I/O error, not the missing-prompt usage error", err)
	}
	if !strings.Contains(err.Error(), "read stdin") {
		t.Errorf("err = %v, want it to mention reading stdin", err)
	}
}

func TestParseArgsMaxTurns(t *testing.T) {
	// default when the flag is absent
	cfg, err := parseArgs([]string{"--model", "m", "p"}, nil, func(string) (string, bool) { return "", false })
	if err != nil {
		t.Fatalf("parseArgs returned error: %v", err)
	}
	if cfg.maxTurns != 50 {
		t.Errorf("default maxTurns = %d, want 50", cfg.maxTurns)
	}

	// prompt before flags: the issue's argv shape
	cfg, err = parseArgs([]string{"p", "--model", "m", "--max-turns", "7"}, nil, func(string) (string, bool) { return "", false })
	if err != nil {
		t.Fatalf("parseArgs (prompt first) returned error: %v", err)
	}
	if cfg.maxTurns != 7 {
		t.Errorf("maxTurns = %d, want 7", cfg.maxTurns)
	}

	// = form
	cfg, err = parseArgs([]string{"--max-turns=3", "--model", "m", "p"}, nil, func(string) (string, bool) { return "", false })
	if err != nil {
		t.Fatalf("parseArgs (= form) returned error: %v", err)
	}
	if cfg.maxTurns != 3 {
		t.Errorf("maxTurns = %d, want 3", cfg.maxTurns)
	}
}

func TestParseArgsTimeout(t *testing.T) {
	// default when the flag is absent
	cfg, err := parseArgs([]string{"--model", "m", "p"}, nil, func(string) (string, bool) { return "", false })
	if err != nil {
		t.Fatalf("parseArgs returned error: %v", err)
	}
	if cfg.timeout != chatAttemptTimeout {
		t.Errorf("default timeout = %v, want %v", cfg.timeout, chatAttemptTimeout)
	}

	// prompt before flags: the issue's argv shape
	cfg, err = parseArgs([]string{"p", "--model", "m", "--timeout", "90s"}, nil, func(string) (string, bool) { return "", false })
	if err != nil {
		t.Fatalf("parseArgs (prompt first) returned error: %v", err)
	}
	if cfg.timeout != 90*time.Second {
		t.Errorf("timeout = %v, want %v", cfg.timeout, 90*time.Second)
	}

	// = form
	cfg, err = parseArgs([]string{"--timeout=2m", "--model", "m", "p"}, nil, func(string) (string, bool) { return "", false })
	if err != nil {
		t.Fatalf("parseArgs (= form) returned error: %v", err)
	}
	if cfg.timeout != 2*time.Minute {
		t.Errorf("timeout = %v, want %v", cfg.timeout, 2*time.Minute)
	}
}

func TestUsageDocumentsTimeoutFlag(t *testing.T) {
	// --help output is the only place an operator learns the knob exists, so it
	// must name the flag and the default actually in effect. The default is read
	// from chatAttemptTimeout rather than typed here, so bumping it fails this
	// test until the usage text is updated too.
	if !strings.Contains(usage, "--timeout") {
		t.Errorf("usage does not document the --timeout flag:\n%s", usage)
	}
	if want := chatAttemptTimeout.String(); !strings.Contains(usage, want) {
		t.Errorf("usage does not state the default timeout %s:\n%s", want, usage)
	}
}

func TestChatAttemptTimeoutIsMinutesScale(t *testing.T) {
	// One attempt can stream a whole long generation, so the default must be
	// minutes-scale; a sub-minute default would expire healthy turns.
	if chatAttemptTimeout < time.Minute {
		t.Errorf("chatAttemptTimeout = %v, want at least %v", chatAttemptTimeout, time.Minute)
	}
}

func TestChatRequestWireShape(t *testing.T) {
	req := chatRequest{
		Model: "m",
		Messages: []chatMessage{
			{Role: "assistant", ToolCalls: []toolCall{{ID: "call_1", Type: "function", Function: toolCallFunction{Name: "bash", Arguments: "{\"command\":\"ls\"}"}}}},
			{Role: "tool", Content: "file listing", ToolCallID: "call_1"},
		},
		Tools: []toolSpec{bashTool},
	}
	raw, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var top map[string]any
	if err := json.Unmarshal(raw, &top); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, key := range []string{"model", "messages", "tools"} {
		if _, ok := top[key]; !ok {
			t.Errorf("request JSON missing key %q; body = %s", key, raw)
		}
	}

	// exactly one function tool, named bash, with a command:string schema
	tools, ok := top["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %#v, want an array with exactly one entry", top["tools"])
	}
	tool, _ := tools[0].(map[string]any)
	if tool["type"] != "function" {
		t.Errorf("tools[0].type = %v, want function", tool["type"])
	}
	fn, _ := tool["function"].(map[string]any)
	if fn["name"] != "bash" {
		t.Errorf("tools[0].function.name = %v, want bash", fn["name"])
	}
	params, _ := fn["parameters"].(map[string]any)
	props, _ := params["properties"].(map[string]any)
	cmd, _ := props["command"].(map[string]any)
	if cmd == nil || cmd["type"] != "string" {
		t.Errorf("tools[0].function.parameters = %v, want a command:string property", fn["parameters"])
	}

	// assistant tool_calls and tool-role message marshal to the OpenAI keys
	msgs, _ := top["messages"].([]any)
	assistant, _ := msgs[0].(map[string]any)
	calls, ok := assistant["tool_calls"].([]any)
	if !ok || len(calls) != 1 {
		t.Fatalf("assistant message = %v, want one tool_calls entry", assistant)
	}
	call, _ := calls[0].(map[string]any)
	for _, key := range []string{"id", "type", "function"} {
		if _, ok := call[key]; !ok {
			t.Errorf("tool_calls[0] missing key %q: %v", key, call)
		}
	}
	if assistant["role"] != "assistant" {
		t.Errorf("assistant role = %v", assistant["role"])
	}
	toolMsg, _ := msgs[1].(map[string]any)
	if toolMsg["role"] != "tool" || toolMsg["content"] != "file listing" || toolMsg["tool_call_id"] != "call_1" {
		t.Errorf("tool message = %v, want role/content/tool_call_id", toolMsg)
	}
}

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, io.ErrClosedPipe }

// The old rule "stdout ends in exactly one trailing newline" died with text
// mode: the answer travels verbatim inside the result event, so a server
// content that already ends in one keeps exactly one and gains nothing, and the
// only newline stdout still guarantees belongs to each event line.
func TestRunResponseEndingInNewline(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"ends with newline\n"}}]}`)
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"--model", "m", "--base-url", srv.URL, "p"}, nil, &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}
	if got := resultText(t, stdout.String()); got != "ends with newline\n" {
		t.Errorf("result event text = %q, want the server content verbatim: one trailing newline, never doubled or appended", got)
	}
	// stdout itself still ends in exactly one newline per event line, so the
	// answer's own newline cannot leak a blank line into the stream.
	raw := stdout.String()
	if n := strings.Count(raw, "\n"); n != len(stdoutEvents(t, raw)) {
		t.Errorf("stdout has %d newline(s) for %d event line(s), want one per line: %q", n, len(stdoutEvents(t, raw)), raw)
	}
}

// -h is not a failure, so it gets no error event; but stdout carries only
// JSONL on every exit path, so the usage text itself belongs on stderr.
func TestRunHelpExitsZeroWithUsageOnStderr(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run([]string{"-h"}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for -h", code)
	}
	if stdout.Len() != 0 {
		t.Errorf("stdout = %q, want nothing: stdout carries only JSONL events", stdout.String())
	}
	if !strings.Contains(stderr.String(), "usage:") {
		t.Errorf("stderr = %q, want usage text", stderr.String())
	}
}

func TestParseArgsPositionalPromptNotTrimmed(t *testing.T) {
	cfg, err := parseArgs([]string{"--model", "m", "  padded prompt  "}, nil, func(string) (string, bool) { return "", false })
	if err != nil {
		t.Fatalf("parseArgs returned error: %v", err)
	}
	if cfg.prompt != "  padded prompt  " {
		t.Errorf("prompt = %q, want the argv prompt verbatim (no trimming)", cfg.prompt)
	}
}

func TestRunBashToolSuccess(t *testing.T) {
	result := runBashTool("echo hello")
	want := "hello\nexit code: 0"
	if result != want {
		t.Errorf("runBashTool(echo hello) = %q, want %q", result, want)
	}
}

func TestRunBashToolFailureIsResultNotError(t *testing.T) {
	// A non-zero exit is ordinary tool output for the model, not a Go error.
	result := runBashTool("echo before; exit 3")
	want := "before\nexit code: 3"
	if result != want {
		t.Errorf("runBashTool(echo before; exit 3) = %q, want %q", result, want)
	}
}

func TestRunBashToolIncludesStderr(t *testing.T) {
	result := runBashTool("echo out; echo err >&2")
	if !strings.Contains(result, "out") || !strings.Contains(result, "err") {
		t.Errorf("runBashTool = %q, want both stdout and stderr combined", result)
	}
	if !strings.HasSuffix(result, "exit code: 0") {
		t.Errorf("runBashTool = %q, want trailing exit-code line", result)
	}
}

func TestRunBashToolRunsInDriverCwd(t *testing.T) {
	// bash -c inherits the Driver's cwd; a relative write must land there.
	// O_EXCL: a stale probe file from an earlier run must fail the test
	// loudly, not get silently deleted by the pre-clean.
	const name = "spindrift-cwd-probe.txt"
	f, err := os.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatalf("create probe %s: %v (stale file from a previous run?)", name, err)
	}
	f.Close()
	defer os.Remove(name)

	result := runBashTool("echo written > " + name)
	if !strings.HasSuffix(result, "exit code: 0") {
		t.Errorf("runBashTool = %q, want success", result)
	}
	data, err := os.ReadFile(name)
	if err != nil {
		t.Fatalf("file %s not written to the process cwd: %v (result %q)", name, err, result)
	}
	if strings.TrimSpace(string(data)) != "written" {
		t.Errorf("file content = %q, want %q", data, "written")
	}
}

func TestRunBashToolNonexistentCommandExits127(t *testing.T) {
	// bash itself starts fine and reports 127 for a command it cannot find;
	// the model still gets an ordinary Tool Result.
	result := runBashTool("nonexistent-spindrift-cmd-xyz")
	if !strings.Contains(result, "not found") || !strings.Contains(result, "exit code: 127") {
		t.Errorf("runBashTool = %q, want error text plus exit-code line", result)
	}
}

func TestRunBashToolUnstartableBashIsWellFormedResult(t *testing.T) {
	// With PATH emptied, bash itself cannot be started; the Tool Result must
	// still be a well-formed string, not a glued "exit code: error:" line.
	// Setenv before the call: exec.Command resolves bash via LookPath at
	// command-construction time, inheriting the current env.
	t.Setenv("PATH", "")
	result := runBashTool("echo hi")
	if !strings.HasPrefix(result, "error: cannot start command:") {
		t.Errorf("runBashTool = %q, want a plain 'error: cannot start command: ...' result", result)
	}
	if strings.Contains(result, "exit code:") {
		t.Errorf("runBashTool = %q, want no exit-code line when the command never started", result)
	}
}

func TestRunBashToolPreservesTrailingBlankLines(t *testing.T) {
	// Only the final newline is trimmed so the exit-code line sits flush;
	// intentional blank lines before it must survive.
	result := runBashTool("printf 'a\\n\\n\\n'")
	want := "a\n\n\nexit code: 0"
	if result != want {
		t.Errorf("runBashTool = %q, want %q", result, want)
	}
}

func TestRunBashToolSignalKillReportsSignal(t *testing.T) {
	// A signal-killed command has no exit code (ExitCode is -1); report the
	// signal instead of the meaningless -1.
	result := runBashTool("kill -9 $$")
	if strings.Contains(result, "exit code: -1") {
		t.Errorf("runBashTool = %q, want the signal reported instead of -1", result)
	}
	if !strings.HasSuffix(result, "exit code: signal killed") {
		t.Errorf("runBashTool = %q, want trailing 'exit code: signal killed' line", result)
	}
}

func TestRunBashToolTimeoutKillsAndReports(t *testing.T) {
	// A command that outlives the tool timeout must be killed and reported
	// as a timeout naming the duration — distinct from a plain signal kill,
	// so the model can tell them apart — and no process of that command may
	// survive the kill.
	old := bashToolTimeout
	bashToolTimeout = 100 * time.Millisecond
	defer func() { bashToolTimeout = old }()

	// Reach sleep through a per-test temp dir so the exec'd sleeper's argv[0]
	// (its full path) is unique: pgrep -f can only match this test's process,
	// not an unrelated host "sleep 5". The symlink keeps the basename "sleep"
	// because coreutils multicall dispatches on argv[0]'s basename.
	sleepBin, err := exec.LookPath("sleep")
	if err != nil {
		t.Fatalf("sleep not found: %v", err)
	}
	dir := t.TempDir()
	probe := dir + "/sleep"
	if err := os.Symlink(sleepBin, probe); err != nil {
		t.Fatalf("cannot link sleep into %s: %v", dir, err)
	}

	result := runBashTool("exec " + probe + " 5")
	if !strings.Contains(result, "timed out after 100ms") {
		t.Errorf("runBashTool(%q) = %q, want a timeout result naming the injected duration", "exec "+probe+" 5", result)
	}
	out, err := exec.Command("pgrep", "-f", dir).Output()
	if err == nil && len(bytes.TrimSpace(out)) > 0 {
		t.Errorf("processes from the timed-out command survived: pgrep -f %q = %s", dir, out)
	}
}

func TestRunBashToolTimeoutWithOversizedOutput(t *testing.T) {
	// Truncation and the timeout exit line are separate code paths (truncate
	// the tail, then append the exit line); this pins their interaction on a
	// command that both floods the cap and outlives the timeout.
	oldTimeout := bashToolTimeout
	oldMax := bashToolMaxOutput
	bashToolTimeout = 100 * time.Millisecond
	bashToolMaxOutput = 256
	defer func() {
		bashToolTimeout = oldTimeout
		bashToolMaxOutput = oldMax
	}()

	result := runBashTool("while true; do seq 1 100000; done")

	// pins the timeout path
	if !strings.Contains(result, "timed out after 100ms") {
		t.Errorf("runBashTool = %q, want a timeout result naming the injected duration", result)
	}
	// pins the truncation path; seq is pure ASCII so the marker names exactly
	// the injected cap
	if !strings.Contains(result, truncatedOutputMarker(bashToolMaxOutput)) {
		t.Errorf("runBashTool = %q, want the truncation marker naming %d kept bytes", result, bashToolMaxOutput)
	}
	if len(result) > bashToolMaxOutput+len(truncatedOutputMarker(bashToolMaxOutput))+64 {
		t.Errorf("runBashTool returned %d bytes, want it bounded by the %d-byte cap plus marker and exit-code line", len(result), bashToolMaxOutput)
	}
	// pins the ordering invariant: the exit line is appended after truncation,
	// so it must survive as the very last thing in the result
	if !strings.HasSuffix(result, "exit code: timed out after 100ms") {
		t.Errorf("runBashTool = %q, want trailing 'exit code: timed out after 100ms' line", result)
	}
}

func TestRunBashToolReturnsWhenEscapedChildHoldsPipes(t *testing.T) {
	// A child that detaches into a new session survives the process-group
	// SIGKILL and inherits the output-pipe write-ends. os/exec's Wait then
	// blocks on their EOF unless WaitDelay bounds it — and a runBashTool
	// that never returns wedges the Driver even though the timeout fired.
	old := bashToolTimeout
	bashToolTimeout = 100 * time.Millisecond
	defer func() { bashToolTimeout = old }()

	t.Setenv("BASH_TOOL_ESCAPE_HELPER", "1")
	escaping := os.Args[0] + " & wait"
	done := make(chan string, 1)
	go func() { done <- runBashTool(escaping) }()
	select {
	case result := <-done:
		if !strings.Contains(result, "timed out after 100ms") {
			t.Errorf("runBashTool(%q) = %q, want a timeout result", escaping, result)
		}
	case <-time.After(10 * time.Second):
		t.Fatalf("runBashTool(%q) did not return within 10s: Wait blocks on the escaped child's pipe", escaping)
	}
}

func TestRunBashToolTruncatesToTail(t *testing.T) {
	// A generator producing far more output than the cap must come back as
	// the TAIL of that output, capped before the exit-code line is appended
	// — so the exit-code line always survives and the model sees where the
	// output ended, not where it began.
	old := bashToolMaxOutput
	bashToolMaxOutput = 256
	defer func() { bashToolMaxOutput = old }()

	result := runBashTool("seq 1 100000")

	if len(result) > bashToolMaxOutput+len(truncatedOutputMarker(bashToolMaxOutput))+64 {
		t.Errorf("runBashTool(seq 1 100000) returned %d bytes, want it bounded by the %d-byte cap plus marker and exit-code line", len(result), bashToolMaxOutput)
	}
	if !strings.HasSuffix(result, "exit code: 0") {
		t.Errorf("runBashTool = %q, want trailing exit-code line even under truncation", result)
	}
	if !strings.Contains(result, "99999\n100000") {
		t.Errorf("runBashTool = %q, want the tail of the generated output (99999, 100000) present", result)
	}
	// seq never emits a bare "1" line except as the first line, so finding
	// "\n1\n" would mean the head survived truncation.
	if strings.Contains(result, "\n1\n") {
		t.Errorf("runBashTool = %q, want the head of the generated output dropped", result)
	}
	if !strings.Contains(result, "truncated") {
		t.Errorf("runBashTool = %q, want a truncation marker", result)
	}
	// seq output is pure ASCII, so the UTF-8 guard drops nothing and the
	// kept tail is exactly the cap; the marker must name that real count,
	// not a literal "N".
	if !strings.Contains(result, fmt.Sprintf("showing last %d bytes", bashToolMaxOutput)) {
		t.Errorf("runBashTool = %q, want marker naming the kept byte count %d", result, bashToolMaxOutput)
	}
}

func TestParseArgsRejectsNonPositiveMaxTurns(t *testing.T) {
	for _, value := range []string{"0", "-2"} {
		_, err := parseArgs([]string{"--model", "m", "--max-turns", value, "p"}, nil, func(string) (string, bool) { return "", false })
		if err == nil {
			t.Errorf("parseArgs --max-turns %s: want an error, got nil", value)
			continue
		}
		if !strings.Contains(err.Error(), "--max-turns") || !strings.Contains(err.Error(), value) {
			t.Errorf("parseArgs --max-turns %s: err = %v, want it to name the flag and value", value, err)
		}
	}
}

func TestParseArgsRejectsNonPositiveTimeout(t *testing.T) {
	for _, value := range []string{"0", "-1s"} {
		_, err := parseArgs([]string{"--model", "m", "--timeout", value, "p"}, nil, func(string) (string, bool) { return "", false })
		if err == nil {
			t.Errorf("parseArgs --timeout %s: want an error, got nil", value)
			continue
		}
		if !strings.Contains(err.Error(), "--timeout") || !strings.Contains(err.Error(), value) {
			t.Errorf("parseArgs --timeout %s: err = %v, want it to name the flag and value", value, err)
		}
	}
}

func TestParseArgsExtraPositional(t *testing.T) {
	_, err := parseArgs([]string{"--model", "m", "p1", "p2"}, nil, func(string) (string, bool) { return "", false })
	if err == nil {
		t.Fatal("parseArgs should reject more than one positional prompt")
	}
	if !strings.Contains(err.Error(), "p2") {
		t.Errorf("err = %v, want it to name the extra argument", err)
	}
}
