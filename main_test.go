package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// TestMain doubles as helper-process entry points before any test runs: the
// escape helper — re-invoked with the env var set, it detaches into a new
// session, outside the process group a timeout SIGKILL targets, and sleeps
// holding the inherited stdout pipe (no setsid binary ships in this environment,
// so the test binary itself is the escape vehicle) — and the child Driver a
// subagent spawn test re-invokes.
func TestMain(m *testing.M) {
	if os.Getenv("BASH_TOOL_ESCAPE_HELPER") == "1" {
		syscall.Setsid()
		time.Sleep(30 * time.Second)
		os.Exit(0)
	}
	// The subagent spawn tests start this binary as a child Driver, so the branch
	// that turns it into one has to sit ahead of every flag parse too — the child
	// is handed the Driver's argv, never a test flag set.
	if os.Getenv(subagentHelperEnv) == "1" {
		os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr, os.LookupEnv))
	}
	// Registered here, before m.Run() parses the command line: the Driver takes a
	// built argv rather than os.Args, so the test binary has no other flag set, and
	// the escape-helper branch above must stay first — a re-invoked helper would
	// otherwise trip over the test flags it is handed.
	goldenUpdate = flag.Bool("update", false, "rewrite the committed golden artifacts ("+goldenPassPath+") instead of comparing them")
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
		writeSSE(w, sseChatCompletion(`{"choices":[{"message":{"role":"assistant","content":"here is the fix"}}]}`))
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
		writeSSE(w, sseChatCompletion(`{"model":"gpt-4o-mini","usage":{"prompt_tokens":11,"completion_tokens":3},"choices":[{"message":{"role":"assistant","content":"here is the fix"}}]}`))
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
		writeSSE(w, sseChatCompletion(`{"model":"gpt-4o-mini","usage":{"prompt_tokens":11,"completion_tokens":3},"choices":[{"message":{"role":"assistant","content":"done"}}]}`))
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
		writeSSE(w, sseChatCompletion(`{"choices":[{"message":{"role":"assistant","content":"no usage here"}}]}`))
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
		writeSSE(w, sseChatCompletion(`{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
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
		writeSSE(w, sseChatCompletion(`{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
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

// sseFrame wraps one payload in the data: frame a chat completion chunk travels
// in.
func sseFrame(payload string) string { return "data: " + payload + "\n\n" }

// sseDone is the marker a streaming endpoint closes with. It ends the stream and
// is not itself a chunk.
const sseDone = "data: [DONE]\n\n"

// sseStream assembles raw chunk payloads into one SSE body, each in its own
// frame and closed with the done marker.
func sseStream(chunks ...string) string {
	var b strings.Builder
	for _, chunk := range chunks {
		b.WriteString(sseFrame(chunk))
	}
	return b.String() + sseDone
}

// writeSSE answers with SSE frames the way a streaming endpoint does: event
// stream content type, then a flush, so the client sees the body without
// waiting for the handler to return.
func writeSSE(w http.ResponseWriter, frames string) {
	w.Header().Set("Content-Type", "text/event-stream")
	io.WriteString(w, frames)
	w.(http.Flusher).Flush()
}

// sseChunk mirrors a streamed chunk: the answer under delta rather than message,
// and usage carried on its own final chunk.
type sseChunk struct {
	Model   string          `json:"model,omitempty"`
	Choices []sseChoice     `json:"choices,omitempty"`
	Usage   json.RawMessage `json:"usage,omitempty"`
}

type sseChoice struct {
	Index        int         `json:"index"`
	Delta        chatMessage `json:"delta"`
	FinishReason *string     `json:"finish_reason,omitempty"`
}

// sseChatCompletion rewrites one non-streaming completion body into the frames a
// streaming endpoint sends for the same answer: a chunk carrying the message as
// its delta (plus the model), a usage chunk when the body had a usage object,
// then the done marker.
func sseChatCompletion(whole string) string {
	var body struct {
		Model   string `json:"model"`
		Choices []struct {
			Message      chatMessage `json:"message"`
			FinishReason *string     `json:"finish_reason"`
		} `json:"choices"`
		Usage json.RawMessage `json:"usage"`
	}
	if err := json.Unmarshal([]byte(whole), &body); err != nil {
		// Not a completion body at all — the unparseable-response cases. It still
		// has to reach the client as one data payload, so it goes out verbatim
		// (newlines flattened, or the frame would not be one line).
		return sseFrame(strings.ReplaceAll(whole, "\n", " ")) + sseDone
	}
	var b strings.Builder
	for i, choice := range body.Choices {
		b.WriteString(sseChunkFrame(sseChunk{Model: body.Model, Choices: []sseChoice{{Index: i, Delta: choice.Message, FinishReason: choice.FinishReason}}}))
	}
	if len(body.Usage) > 0 {
		b.WriteString(sseChunkFrame(sseChunk{Model: body.Model, Usage: body.Usage}))
	}
	b.WriteString(sseDone)
	return b.String()
}

func sseChunkFrame(chunk sseChunk) string {
	encoded, err := json.Marshal(chunk)
	if err != nil {
		// unreachable: every field was decoded out of a valid JSON document
		panic("sse frame: " + err.Error())
	}
	return sseFrame(string(encoded))
}

// scriptedChatServer returns the i-th scripted JSON response for the i-th
// POST and records every request body for later inspection.
type scriptedChatServer struct {
	mu        sync.Mutex
	responses []string
	// frame turns a scripted entry into the response body. The completion fake
	// rewrites whole JSON bodies; the raw-chunk fake pre-frames them itself and
	// leaves this nil.
	frame     func(string) string
	rawBodies [][]byte
	srv       *httptest.Server
}

func newScriptedChatServer(t *testing.T, responses ...string) *scriptedChatServer {
	s := &scriptedChatServer{responses: responses, frame: sseChatCompletion}
	return s.serve(t)
}

// newScriptedSSEServer scripts one raw SSE stream per POST: every entry of the
// i-th list goes out as its own data: frame, so a test owns the chunk
// boundaries — split answers, split tool-call arguments, a lone usage chunk.
func newScriptedSSEServer(t *testing.T, streams ...[]string) *scriptedChatServer {
	bodies := make([]string, len(streams))
	for i, chunks := range streams {
		bodies[i] = sseStream(chunks...)
	}
	return newScriptedSSERawServer(t, bodies...)
}

// newScriptedSSERawServer scripts one SSE response body per POST, verbatim, for
// the streams a chunk list cannot express: keep-alive comments, event: lines, a
// frame left unterminated, and the stalls sseStall describes.
func newScriptedSSERawServer(t *testing.T, bodies ...string) *scriptedChatServer {
	return (&scriptedChatServer{responses: bodies}).serve(t)
}

// sseStallPrefix marks a scripted response that never finishes.
const sseStallPrefix = "\x00stall:"

// sseStall scripts a stalled endpoint: everything given as frames reaches the
// client and is flushed, then the stream says nothing more until the client
// hangs up. sseStall("") is the stall before the first byte — not even a status
// line is written, so the round trip itself must expire.
func sseStall(frames string) string { return sseStallPrefix + frames }

// SSE carries more than chunks: a ':' line is a keep-alive comment, an event:
// line names the frame that follows, and the done marker closes the stream. None
// of them is a delta, so the answer stays exactly what the data frames said.
func TestRunStreamIgnoresHeartbeatsEventLinesAndDoneMarker(t *testing.T) {
	srv := newScriptedSSERawServer(t, ": keep-alive\n\n"+
		"event: message\n\n"+
		sseFrame(`{"model":"m-a","choices":[{"index":0,"delta":{"content":"only this"}}]}`)+
		sseDone+
		sseFrame(`{"model":"m-a","choices":[{"index":0,"delta":{"content":" after the marker"}}]}`))

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-a", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}
	if got := resultText(t, stdout.String()); got != "only this" {
		t.Errorf("result text = %q, want only the chunk before the done marker", got)
	}
	if n := strings.Count(stderr.String(), "POST "); n != 1 {
		t.Errorf("stderr carries %d POST line(s), want the one progress line: %q", n, stderr.String())
	}
}

func (s *scriptedChatServer) serve(t *testing.T) *scriptedChatServer {
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
		if s.frame != nil {
			resp = s.frame(resp)
		}
		s.mu.Unlock()
		if exhausted {
			http.Error(w, "no scripted response left", http.StatusInternalServerError)
			return
		}
		if frames, stalled := strings.CutPrefix(resp, sseStallPrefix); stalled {
			// The handler lets go the moment the client does, so a stalled stream
			// cannot wedge the deferred server close if the bound ever stops
			// firing: only an endpoint still being waited on keeps stalling.
			if frames != "" {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				io.WriteString(w, frames)
				w.(http.Flusher).Flush()
			}
			<-r.Context().Done()
			return
		}
		writeSSE(w, resp)
	}))
	t.Cleanup(s.srv.Close)
	return s
}

// A streamed answer is still exactly one text event and exactly one usage event:
// the Driver assembles the chunks below the sendChat seam and emits nothing per
// frame, so stdout does not grow with the number of chunks the server used.
func TestRunStreamedChunksAssembleOneTextAndOneUsageEvent(t *testing.T) {
	srv := newScriptedSSEServer(t,
		[]string{
			`{"model":"m-a","choices":[{"index":0,"delta":{"content":"hello "}}]}`,
			`{"model":"m-a","choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop"}]}`,
			`{"model":"m-a","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3}}`,
		},
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-a", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	var textEvents, usageEvents []map[string]any
	for _, ev := range stdoutEvents(t, stdout.String()) {
		switch ev["kind"] {
		case "text":
			textEvents = append(textEvents, ev)
		case "usage":
			usageEvents = append(usageEvents, ev)
		}
	}
	if len(textEvents) != 1 {
		t.Fatalf("stdout carries %d text event(s), want the chunks joined into one: %q", len(textEvents), stdout.String())
	}
	if textEvents[0]["text"] != "hello world" {
		t.Errorf("text event = %v, want both chunks concatenated in order", textEvents[0]["text"])
	}
	if len(usageEvents) != 1 {
		t.Fatalf("stdout carries %d usage event(s), want one for the call: %q", len(usageEvents), stdout.String())
	}
	tokens, _ := usageEvents[0]["tokens"].(map[string]any)
	if tokens["input"] != float64(12) || tokens["output"] != float64(3) {
		t.Errorf("usage tokens = %v, want the usage chunk's tally input 12 output 3", tokens)
	}
	if got := resultText(t, stdout.String()); got != "hello world" {
		t.Errorf("result text = %q, want the assembled answer", got)
	}
}

// A second choice on a stream is an alternative completion, not a continuation
// of the first: this Driver never asks for n>1, and the agent loop replays
// exactly one assistant message. Deltas beyond index 0 are dropped rather than
// spliced into the one text event, while a delta naming no index at all still
// belongs to choice 0.
func TestRunStreamIgnoresDeltasBeyondTheFirstChoice(t *testing.T) {
	srv := newScriptedSSEServer(t,
		[]string{
			`{"model":"m-a","choices":[{"index":0,"delta":{"content":"hello "}}]}`,
			`{"model":"m-a","choices":[{"index":1,"delta":{"content":"ALTERNATIVE"}}]}`,
			`{"model":"m-a","choices":[{"delta":{"content":"again "}}]}`,
			`{"model":"m-a","choices":[{"index":1,"delta":{"content":" ATIVE"},"finish_reason":"stop"}]}`,
			`{"model":"m-a","choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop"}]}`,
			`{"model":"m-a","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3}}`,
		},
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-a", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	var textEvents []map[string]any
	for _, ev := range stdoutEvents(t, stdout.String()) {
		if ev["kind"] == "text" {
			textEvents = append(textEvents, ev)
		}
	}
	if len(textEvents) != 1 {
		t.Fatalf("stdout carries %d text event(s), want choice 0 only: %q", len(textEvents), stdout.String())
	}
	if textEvents[0]["text"] != "hello again world" {
		t.Errorf("text event = %v, want choice 0's deltas with the index-less delta folded in", textEvents[0]["text"])
	}
	if got := resultText(t, stdout.String()); got != "hello again world" {
		t.Errorf("result text = %q, want choice 0's answer alone", got)
	}
	if strings.Contains(stdout.String(), "ALTERNATIVE") {
		t.Errorf("stdout carries an alternative choice's prose: %q", stdout.String())
	}
}

// A reasoning model thinks before it answers, and the thinking streams in
// delta.reasoning_content — its own channel, never the answer's. A message that
// only thinks still completes, so it owes exactly one reasoning event holding
// the joined fragments and no text event at all, and the reasoning stays out of
// the result event's prose.
func TestRunStreamReasoningOnlyAnswerEmitsOneReasoningEvent(t *testing.T) {
	srv := newScriptedSSEServer(t,
		[]string{
			`{"model":"m-a","choices":[{"index":0,"delta":{"reasoning_content":"thinking "}}]}`,
			`{"model":"m-a","choices":[{"index":0,"delta":{"reasoning_content":"about it"},"finish_reason":"stop"}]}`,
			`{"model":"m-a","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":7}}`,
		},
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-a", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	var reasoningEvents, textEvents []map[string]any
	for _, ev := range stdoutEvents(t, stdout.String()) {
		switch ev["kind"] {
		case "reasoning":
			reasoningEvents = append(reasoningEvents, ev)
		case "text":
			textEvents = append(textEvents, ev)
		}
	}
	if len(reasoningEvents) != 1 {
		t.Fatalf("stdout carries %d reasoning event(s), want the fragments joined into one: %q", len(reasoningEvents), stdout.String())
	}
	if reasoningEvents[0]["text"] != "thinking about it" {
		t.Errorf("reasoning event text = %v, want both fragments concatenated in order", reasoningEvents[0]["text"])
	}
	if reasoningEvents[0]["message_id"] != "msg-1" {
		t.Errorf("reasoning event message_id = %v, want the msg-1 of the call that thought it", reasoningEvents[0]["message_id"])
	}
	if len(textEvents) != 0 {
		t.Errorf("stdout carries %d text event(s), want none when the message only reasoned: %q", len(textEvents), stdout.String())
	}
	if got := resultText(t, stdout.String()); strings.Contains(got, "thinking") {
		t.Errorf("result text = %q, want none of the reasoning in it", got)
	}
}

// Reasoning and prose in one message stay two accumulators and two events: the
// thought goes out first, then the answer alone, and the result event reports
// the prose only.
func TestRunStreamReasoningAndProseEmitReasoningThenText(t *testing.T) {
	srv := newScriptedSSEServer(t,
		[]string{
			`{"model":"m-a","choices":[{"index":0,"delta":{"reasoning_content":"let me "}}]}`,
			`{"model":"m-a","choices":[{"index":0,"delta":{"reasoning_content":"check"}}]}`,
			`{"model":"m-a","choices":[{"index":0,"delta":{"content":"the "}}]}`,
			`{"model":"m-a","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop"}]}`,
			`{"model":"m-a","choices":[],"usage":{"prompt_tokens":15,"completion_tokens":11}}`,
		},
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-a", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	events := stdoutEvents(t, stdout.String())
	var kinds []string
	var reasoningText, proseText string
	usageEvents := 0
	for _, ev := range events {
		switch ev["kind"] {
		case "reasoning":
			kinds = append(kinds, "reasoning")
			reasoningText = fmt.Sprint(ev["text"])
		case "text":
			kinds = append(kinds, "text")
			proseText = fmt.Sprint(ev["text"])
		case "usage":
			usageEvents++
		}
	}
	if want := []string{"reasoning", "text"}; !reflect.DeepEqual(kinds, want) {
		t.Fatalf("stdout prose kinds = %v, want %v (one reasoning event then one text event): %q", kinds, want, stdout.String())
	}
	if reasoningText != "let me check" {
		t.Errorf("reasoning event text = %q, want only the reasoning fragments joined", reasoningText)
	}
	if proseText != "the answer" {
		t.Errorf("text event text = %q, want only the content fragments joined", proseText)
	}
	if got := resultText(t, stdout.String()); got != "the answer" {
		t.Errorf("result text = %q, want the prose alone, without the reasoning", got)
	}
	if usageEvents != 1 {
		t.Errorf("stdout carries %d usage event(s), want the one this call owes: %q", usageEvents, stdout.String())
	}
}

// A streamed Tool Call arrives in fragments: the first delta names the call and
// the rest only append to its arguments, so the fragments are not calls. The
// Driver has to hand the loop one whole call — the same shape a non-streamed
// tool_calls entry always had — so exactly one tool_call event goes out with the
// id, name and complete arguments, paired with its tool_result.
func TestRunStreamReassemblesToolCallSplitAcrossChunks(t *testing.T) {
	srv := newScriptedSSEServer(t,
		[]string{
			`{"model":"m-a","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":""}}]}}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"comm"}}]}}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"and\":\"ec"}}]}}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ho hel"}}]}}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"lo\"}"}}]},"finish_reason":"tool_calls"}]}`,
			`{"model":"m-a","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":9}}`,
		},
		[]string{
			`{"model":"m-a","choices":[{"index":0,"delta":{"content":"all done"},"finish_reason":"stop"}]}`,
		},
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-a", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	events := stdoutEvents(t, stdout.String())
	callEv, resultEv := toolPair(t, events)
	if callEv["tool_id"] != "call_1" || resultEv["tool_id"] != "call_1" {
		t.Errorf("pair tool_id = %v / %v, want the first delta's id call_1 on both", callEv["tool_id"], resultEv["tool_id"])
	}
	if callEv["tool_name"] != "bash" {
		t.Errorf("tool_call tool_name = %v, want the first delta's name bash", callEv["tool_name"])
	}
	input, ok := callEv["tool_input"].(map[string]any)
	if !ok || input["command"] != "echo hello" {
		t.Errorf("tool_call tool_input = %#v, want the argument fragments joined into {command: echo hello}", callEv["tool_input"])
	}
	if callEv["tool_target"] != "echo hello" {
		t.Errorf("tool_call tool_target = %v, want the reassembled command", callEv["tool_target"])
	}
	if got := resultText(t, stdout.String()); got != "all done" {
		t.Errorf("result text = %q, want the loop to continue past the reassembled call", got)
	}
}

// A Model that asks for two things at once streams both calls' fragments into
// the same deltas, alternating. index is what tells the fragments apart, so the
// two calls stay separate and each still gets one whole pair, in the order the
// Model listed them.
func TestRunStreamReassemblesInterleavedToolCallFragments(t *testing.T) {
	srv := newScriptedSSEServer(t,
		[]string{
			`{"model":"m-a","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"bash","arguments":""}}]}}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"bash","arguments":""}}]}}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":\"ec"}}]}}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"command\":\"ec"}}]}}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ho a\"}"}},{"index":1,"function":{"arguments":"ho b\"}"}}]},"finish_reason":"tool_calls"}]}`,
			`{"model":"m-a","choices":[],"usage":{"prompt_tokens":30,"completion_tokens":14}}`,
		},
		[]string{
			`{"model":"m-a","choices":[{"index":0,"delta":{"content":"both ran"},"finish_reason":"stop"}]}`,
		},
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-a", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	type pair struct {
		id    string
		name  string
		input any
	}
	var calls []pair
	// each call's tool_result has to be the very next event, carrying its id
	var pending string
	seen := 0
	for _, ev := range stdoutEvents(t, stdout.String()) {
		switch ev["kind"] {
		case "tool_call":
			if pending != "" {
				t.Fatalf("tool_call %v follows %v without its tool_result: %v", ev["tool_id"], pending, calls)
			}
			calls = append(calls, pair{id: fmt.Sprint(ev["tool_id"]), name: fmt.Sprint(ev["tool_name"]), input: ev["tool_input"]})
			pending = fmt.Sprint(ev["tool_id"])
		case "tool_result":
			seen++
			if got := fmt.Sprint(ev["tool_id"]); pending == "" || got != pending {
				t.Errorf("tool_result tool_id = %v, want the pending call %v", got, pending)
			}
			pending = ""
		}
	}
	want := []pair{
		{id: "call_a", name: "bash", input: map[string]any{"command": "echo a"}},
		{id: "call_b", name: "bash", input: map[string]any{"command": "echo b"}},
	}
	if !reflect.DeepEqual(calls, want) {
		t.Errorf("tool_call events = %+v, want %v", calls, want)
	}
	if seen != 2 {
		t.Errorf("stdout carries %d tool_result event(s), want one per reassembled call", seen)
	}
	if got := resultText(t, stdout.String()); got != "both ran" {
		t.Errorf("result text = %q, want the loop to continue past both calls", got)
	}
}

// A server that omits index can still deliver each whole call in its own
// delta, both landing at position 0 of their delta. Position alone would merge
// them into one call whose arguments are two JSON objects spliced together, so
// the differing ids have to keep them apart.
func TestRunStreamSplitsIndexlessWholeCallsInSeparateDeltas(t *testing.T) {
	srv := newScriptedSSEServer(t,
		[]string{
			`{"model":"m-a","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"echo one\"}"}}]}}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_2","type":"function","function":{"name":"bash","arguments":"{\"command\":\"echo two\"}"}}]},"finish_reason":"tool_calls"}]}`,
			`{"model":"m-a","choices":[],"usage":{"prompt_tokens":25,"completion_tokens":12}}`,
		},
		[]string{
			`{"model":"m-a","choices":[{"index":0,"delta":{"content":"all done"},"finish_reason":"stop"}]}`,
		},
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-a", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}

	type pair struct {
		id    string
		name  string
		input any
	}
	var calls []pair
	// each call's tool_result has to be the very next event, carrying its id
	var pending string
	seen := 0
	for _, ev := range stdoutEvents(t, stdout.String()) {
		switch ev["kind"] {
		case "tool_call":
			if pending != "" {
				t.Fatalf("tool_call %v follows %v without its tool_result: %v", ev["tool_id"], pending, calls)
			}
			calls = append(calls, pair{id: fmt.Sprint(ev["tool_id"]), name: fmt.Sprint(ev["tool_name"]), input: ev["tool_input"]})
			pending = fmt.Sprint(ev["tool_id"])
		case "tool_result":
			seen++
			if got := fmt.Sprint(ev["tool_id"]); pending == "" || got != pending {
				t.Errorf("tool_result tool_id = %v, want the pending call %v", got, pending)
			}
			pending = ""
		}
	}
	want := []pair{
		{id: "call_1", name: "bash", input: map[string]any{"command": "echo one"}},
		{id: "call_2", name: "bash", input: map[string]any{"command": "echo two"}},
	}
	if !reflect.DeepEqual(calls, want) {
		t.Errorf("tool_call events = %+v, want %v", calls, want)
	}
	if seen != 2 {
		t.Errorf("stdout carries %d tool_result event(s), want one per whole call", seen)
	}
	if got := resultText(t, stdout.String()); got != "all done" {
		t.Errorf("result text = %q, want the loop to continue past both calls", got)
	}
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
		writeSSE(w, sseChatCompletion(`{"choices":[{"message":{"role":"assistant","content":"`+final+`"}}]}`))
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

// A plan that names an agent is the only thing that puts subagent on a
// tool_result line. The dispatch case that sets it is not written yet, so the
// plumbing is pinned at the emit seam with a plan built directly rather than a
// scripted subagent call planToolCall cannot yet plan.
func TestToolResultEventCarriesThePlansSubagent(t *testing.T) {
	pinClock(t, 1700000000000)

	plan := toolPlan{
		input:    map[string]any{"agent": "reviewer", "task": "read main.go"},
		target:   "reviewer",
		subagent: "reviewer",
		run:      func() string { return "the child's answer" },
	}

	var buf bytes.Buffer
	ev := toolResultEvent("msg-1", "call_1", "subagent", plan, plan.run())
	if err := (eventWriter{w: &buf}).emit(ev); err != nil {
		t.Fatalf("emit: %v", err)
	}
	got := decodeLine(t, strings.TrimSuffix(buf.String(), "\n"))
	if got["subagent"] != "reviewer" {
		t.Errorf("tool_result subagent = %v, want the agent the Subagent Tool acted on: %v", got["subagent"], got)
	}
	for _, pin := range []struct {
		key  string
		want any
	}{
		{"kind", "tool_result"},
		{"message_id", "msg-1"},
		{"tool_id", "call_1"},
		{"tool_name", "subagent"},
		{"tool_target", "reviewer"},
		{"tool_output", "the child's answer"},
	} {
		if got[pin.key] != pin.want {
			t.Errorf("tool_result %s = %v, want %v", pin.key, got[pin.key], pin.want)
		}
	}
	// tool_input belongs to the tool_call half of the pair, so naming an agent
	// must not turn the result line into a dump of the plan's input.
	if _, ok := got["tool_input"]; ok {
		t.Errorf("tool_result carries tool_input, which only tool_call fills: %v", got)
	}
}

// The bash exit-code suffix rule is the default, not the only rule: a Subagent
// Tool Result is the child's answer text and never ends in the literal a command
// appends, so a plan carrying its own rule must have that rule consulted — the
// default alone reports every successful delegation as a failure.
func TestToolResultIsErrorUsesThePlanRuleWhenOneIsSet(t *testing.T) {
	pinClock(t, 1700000000000)

	agentRule := func(output string) bool { return strings.HasPrefix(output, "error:") }
	for _, tc := range []struct {
		name   string
		plan   toolPlan
		output string
		want   bool
	}{
		{"default rule, clean exit", toolPlan{}, "ran\nexit code: 0", false},
		{"default rule, non-zero exit", toolPlan{}, "blew up\nexit code: 4", true},
		{"default rule, error prefix", toolPlan{}, "error: unknown tool \"squiggle\"", true},
		{"plan rule accepts child prose", toolPlan{isError: agentRule}, "the child's answer", false},
		{"plan rule rejects child failure", toolPlan{isError: agentRule}, "error: agent \"reviewer\" failed", true},
	} {
		ev := toolResultEvent("msg-1", "call_1", "bash", tc.plan, tc.output)
		if ev.IsError != tc.want {
			t.Errorf("%s: is_error = %v, want %v", tc.name, ev.IsError, tc.want)
		}
		var buf bytes.Buffer
		if err := (eventWriter{w: &buf}).emit(ev); err != nil {
			t.Fatalf("%s: emit: %v", tc.name, err)
		}
		_, onWire := decodeLine(t, strings.TrimSuffix(buf.String(), "\n"))["is_error"]
		if onWire != tc.want {
			t.Errorf("%s: is_error key on wire = %v, want it present only when true: %q", tc.name, onWire, buf.String())
		}
	}
}

// A run that delegates to nobody must look exactly like the run it was before
// the field existed: no event line may grow a subagent key.
func TestRunWithNoSubagentNeverStatesSubagent(t *testing.T) {
	srv := newScriptedChatServer(t,
		`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"echo hello\"}"}}]}}]}`,
		`{"choices":[{"message":{"role":"assistant","content":"done"}}]}`,
	)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}
	for _, ev := range stdoutEvents(t, stdout.String()) {
		if v, ok := ev["subagent"]; ok {
			t.Errorf("%v states subagent = %v for a run with no delegation, want the key absent", ev["kind"], v)
		}
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
		writeSSE(w, sseChatCompletion(`{"choices":[{"message":{"role":"assistant","content":"`+final+`"}}]}`))
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
		{"unparseable chunk", http.StatusOK, sseFrame("this is not json") + sseDone, "unparseable"},
		{"empty stream", http.StatusOK, sseDone, "stream"},
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
				// only a 200 is the event stream; a rejection body stays plain
				if tc.status == http.StatusOK {
					w.Header().Set("Content-Type", "text/event-stream")
				} else {
					w.Header().Set("Content-Type", "application/json")
				}
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
// endpoint and the idle bound that fired: the raw Go error (a bare "context
// deadline exceeded", or a net-poller timeout) names neither, so both must be added. The caller passes the bound it injected, so
// each is checked against the timeout it actually configured.
func assertTimeoutErrorLine(t *testing.T, stderr, target string, bound time.Duration) {
	t.Helper()
	if got := driverErrorLine(stderr, target); !strings.Contains(got, target) || !strings.Contains(got, fmt.Sprint(bound)) {
		t.Errorf("error line = %q, want it to name the endpoint %s and the %s bound that fired", got, target, bound)
	}
}

// A stalled endpoint is the black hole this test pins: the connection is
// accepted and nothing ever answers, so only the idle bound can turn it into a
// non-zero exit. The elapsed bound is the load-bearing
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
		writeSSE(w, sseChatCompletion(`{"choices":[{"message":{"role":"assistant","content":"too late"}}]}`))
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	start := time.Now()
	code := run([]string{"p", "--model", "m", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	elapsed := time.Since(start)

	if elapsed > time.Second {
		t.Errorf("run took %s, want well under the handler's %s stall (no idle bound fired)", elapsed, stall)
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
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		// headers plus the start of one frame reach the client, then the stream
		// stops: the read, not the round trip, is what has to expire. The frame is
		// left unterminated so no chunk can be assembled out of it.
		io.WriteString(w, `data: {"choices":[{"index":0,"delta":{"content":"`)
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
		t.Errorf("run took %s, want well under the handler's %s stall (no idle bound fired on the body read)", elapsed, stall)
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
// be explained by --timeout reaching the client's transport.
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
		writeSSE(w, sseChatCompletion(`{"choices":[{"message":{"role":"assistant","content":"too late"}}]}`))
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
		writeSSE(w, sseChatCompletion(`{"choices":[{"message":{"role":"assistant","content":"too late"}}]}`))
	}))
	defer srv.Close()

	// A hand-built config leaves cfg.timeout zero, and a zero bound means no wait
	// at all rather than no limit — the attempt must still land on the finite
	// package default.
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

// --timeout bounds the silence between bytes, not the length of the answer:
// a server that trickles one chunk every 10ms for 400ms is a healthy stream
// under a 50ms bound, because every gap stays under it. A whole-attempt cap
// cannot tell the two apart and kills the run at 50ms.
func TestRunStreamedTrickleRunsOnPastTheIdleBound(t *testing.T) {
	oldTimeout, oldAttempts, oldBackoff := chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff
	chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = 50*time.Millisecond, 1, time.Millisecond
	defer func() {
		chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = oldTimeout, oldAttempts, oldBackoff
	}()

	const chunks, gap = 40, 10 * time.Millisecond
	var want strings.Builder
	stream := make([]string, 0, chunks+1)
	for i := 0; i < chunks; i++ {
		fmt.Fprintf(&want, "%d ", i)
		stream = append(stream, fmt.Sprintf(`{"model":"m-a","choices":[{"index":0,"delta":{"content":"%d "}}]}`, i))
	}
	stream = append(stream, `{"model":"m-a","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":4}}`)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for _, chunk := range stream {
			// one flush per chunk, so the client watches a live stream rather
			// than receiving one write when the handler returns
			io.WriteString(w, sseFrame(chunk))
			w.(http.Flusher).Flush()
			select {
			case <-time.After(gap):
			case <-r.Context().Done():
				return
			}
		}
		io.WriteString(w, sseDone)
		w.(http.Flusher).Flush()
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	start := time.Now()
	code := run([]string{"p", "--model", "m-a", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	elapsed := time.Since(start)

	if elapsed < time.Duration(chunks)*gap {
		t.Fatalf("run finished in %s, before the scripted stream could even finish trickling (%s): %q", elapsed, time.Duration(chunks)*gap, stderr.String())
	}
	if code != 0 {
		t.Fatalf("exit code = %d, want 0: a stream that keeps sending bytes has not idled; elapsed = %s stderr = %q", code, elapsed, stderr.String())
	}
	if got := resultText(t, stdout.String()); got != want.String() {
		t.Errorf("result text = %q, want every chunk joined in order %q", got, want.String())
	}
	usageEvents := 0
	for _, ev := range stdoutEvents(t, stdout.String()) {
		if ev["kind"] != "usage" {
			continue
		}
		usageEvents++
		tokens, _ := ev["tokens"].(map[string]any)
		if tokens["input"] != float64(7) || tokens["output"] != float64(4) {
			t.Errorf("usage tokens = %v, want the final usage chunk's tally input 7 output 4", tokens)
		}
	}
	if usageEvents != 1 {
		t.Errorf("stdout carries %d usage event(s), want exactly one for the call: %q", usageEvents, stdout.String())
	}
	// the per-attempt note only: the retry note and the terminal error line
	// mention the endpoint too
	if n := strings.Count(stderr.String(), "spindrift-driver: POST "+srv.URL+"/chat/completions\n"); n != 1 {
		t.Errorf("stderr carries %d POST lines, want the one attempt (a live stream is not retried): %q", n, stderr.String())
	}
}

// A stream that says one thing and then stops talking is stalled: the idle bound
// fires inside the body read, so the diagnostic has to name that leg exactly
// once, without nesting the non-deadline "read response:" wording.
func TestRunMidStreamStallTimesOutOnTheReadLeg(t *testing.T) {
	oldTimeout, oldAttempts, oldBackoff := chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff
	chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = 50*time.Millisecond, 2, time.Millisecond
	defer func() {
		chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = oldTimeout, oldAttempts, oldBackoff
	}()

	// one completed chunk, then silence: the answer is half an answer and the
	// attempt has to expire rather than wait for the rest
	first := sseFrame(`{"model":"m-a","choices":[{"index":0,"delta":{"content":"half an answer"}}]}`)
	srv := newScriptedSSERawServer(t, sseStall(first), sseStall(first))

	var stdout, stderr bytes.Buffer
	start := time.Now()
	code := run([]string{"p", "--model", "m-a", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	elapsed := time.Since(start)

	if elapsed > time.Second {
		t.Errorf("run took %s, want each attempt bounded by the 50ms idle bound, not by the scripted stall", elapsed)
	}
	if code == 0 {
		t.Fatalf("exit code = 0, want non-zero once every attempt stalls on a half answer; stderr = %q", stderr.String())
	}
	if got := oneErrorEvent(t, stdout.String()); !strings.Contains(got, "while reading the response") {
		t.Errorf("error event text = %q, want the response-reading leg named", got)
	}
	target := srv.srv.URL + "/chat/completions"
	assertTimeoutErrorLine(t, stderr.String(), target, 50*time.Millisecond)
	if got := driverErrorLine(stderr.String(), target); !strings.Contains(got, "while reading the response") || strings.Contains(got, "read response:") {
		t.Errorf("error line = %q, want it to name the response-reading leg once without nesting %q", got, "read response:")
	}
	// the stall is flakiness, so the retry budget is spent on it and only the
	// last attempt's error reaches the operator; count the per-attempt notes,
	// which are the only lines that are exactly "POST <target>"
	if n := strings.Count(stderr.String(), "spindrift-driver: POST "+target+"\n"); n != 2 {
		t.Errorf("stderr carries %d POST lines, want 2 (a stalled stream is retried): %q", n, stderr.String())
	}
}

// The same bound covers the silence before the first byte: an endpoint that
// accepts the POST and answers with nothing at all expires on the round trip,
// so the wording stays the plain one that names the endpoint and the bound and
// says nothing about a read that never started.
func TestRunStallBeforeFirstByteTimesOutOnTheRoundTrip(t *testing.T) {
	oldTimeout, oldAttempts, oldBackoff := chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff
	chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = 50*time.Millisecond, 2, time.Millisecond
	defer func() {
		chatAttemptTimeout, chatRetryAttempts, chatRetryBackoff = oldTimeout, oldAttempts, oldBackoff
	}()

	srv := newScriptedSSERawServer(t, sseStall(""), sseStall(""))

	var stdout, stderr bytes.Buffer
	start := time.Now()
	code := run([]string{"p", "--model", "m-a", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	elapsed := time.Since(start)

	if elapsed > time.Second {
		t.Errorf("run took %s, want each attempt bounded by the 50ms idle bound, not by the scripted stall", elapsed)
	}
	if code == 0 {
		t.Fatalf("exit code = 0, want non-zero once every attempt goes unanswered; stderr = %q", stderr.String())
	}
	target := srv.srv.URL + "/chat/completions"
	assertTimeoutErrorLine(t, stderr.String(), target, 50*time.Millisecond)
	if got := oneErrorEvent(t, stdout.String()); !strings.Contains(got, "timed out") {
		t.Errorf("error event text = %q, want the attempt timeout", got)
	}
	if got := driverErrorLine(stderr.String(), target); strings.Contains(got, "while reading the response") {
		t.Errorf("error line = %q, want the round-trip wording: no byte ever arrived, so no read leg began", got)
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
		// the streaming analogue of the old {"choices":[]}: a stream that closes
		// without ever carrying a chunk
		writeSSE(w, sseDone)
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"--model", "m", "--base-url", srv.URL, "p"}, nil, &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code == 0 {
		t.Fatal("a stream with no chunks should exit non-zero")
	}
	const wantErr = "server returned a stream with no chunks"
	if got := oneErrorEvent(t, stdout.String()); got != wantErr {
		t.Errorf("error event text = %q, want %q", got, wantErr)
	}
	if !strings.Contains(stderr.String(), "stream") {
		t.Errorf("stderr = %q, want a message about the empty stream", stderr.String())
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
	// keyed by name: the value answers only for SPINDRIFT_BASE_URL, so it does
	// not double as a nonsense SPINDRIFT_CONFIG
	env := func(name string) (string, bool) {
		switch name {
		case envBaseURL:
			return "http://env:9/v1", true
		}
		return "", false
	}

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

func TestParseArgsBaseURLPrecedenceIncludesDriverConfig(t *testing.T) {
	configPath := writeDriverConfig(t, `{"base_url":"http://from-config:8/v1"}`)

	for _, tc := range []struct {
		name        string
		args        []string
		envBaseURL  string
		envConfig   string
		wantBaseURL string
	}{
		{
			name:        "flag alone",
			args:        []string{"--base-url", "http://flag:1/v1"},
			wantBaseURL: "http://flag:1/v1",
		},
		{
			name:        "env alone",
			envBaseURL:  "http://env:9/v1",
			wantBaseURL: "http://env:9/v1",
		},
		{
			name:        "config alone",
			envConfig:   configPath,
			wantBaseURL: "http://from-config:8/v1",
		},
		{
			name:        "flag beats env and config",
			args:        []string{"--base-url", "http://flag:1/v1"},
			envBaseURL:  "http://env:9/v1",
			envConfig:   configPath,
			wantBaseURL: "http://flag:1/v1",
		},
		{
			name:        "env beats config",
			envBaseURL:  "http://env:9/v1",
			envConfig:   configPath,
			wantBaseURL: "http://env:9/v1",
		},
		{
			name:        "nothing set",
			wantBaseURL: "http://localhost:8080/v1",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// keyed by name: anything this case does not name (HOME included)
			// reads as unset
			lookup := func(name string) (string, bool) {
				switch name {
				case envBaseURL:
					return tc.envBaseURL, tc.envBaseURL != ""
				case envConfigPath:
					return tc.envConfig, tc.envConfig != ""
				}
				return "", false
			}
			args := append([]string{"--model", "m"}, tc.args...)
			args = append(args, "p")
			cfg, err := parseArgs(args, nil, lookup)
			if err != nil {
				t.Fatalf("parseArgs returned error: %v", err)
			}
			if cfg.baseURL != tc.wantBaseURL {
				t.Errorf("baseURL = %q, want %q", cfg.baseURL, tc.wantBaseURL)
			}
		})
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

func TestParseArgsTimeoutIncludesDriverConfig(t *testing.T) {
	configPath := writeDriverConfig(t, `{"timeout":"2m"}`)

	for _, tc := range []struct {
		name      string
		args      []string
		envConfig string
		want      time.Duration
	}{
		{
			name: "flag alone",
			args: []string{"--timeout", "30s"},
			want: 30 * time.Second,
		},
		{
			name:      "config alone",
			envConfig: configPath,
			want:      2 * time.Minute,
		},
		{
			name:      "flag beats config",
			args:      []string{"--timeout", "30s"},
			envConfig: configPath,
			want:      30 * time.Second,
		},
		{
			name: "nothing set",
			want: chatAttemptTimeout,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// keyed by name: anything this case does not name (HOME included)
			// reads as unset
			lookup := func(name string) (string, bool) {
				switch name {
				case envConfigPath:
					return tc.envConfig, tc.envConfig != ""
				default:
					return "", false
				}
			}
			args := append([]string{"--model", "m"}, tc.args...)
			args = append(args, "p")
			cfg, err := parseArgs(args, nil, lookup)
			if err != nil {
				t.Fatalf("parseArgs returned error: %v", err)
			}
			if cfg.timeout != tc.want {
				t.Errorf("timeout = %v, want %v", cfg.timeout, tc.want)
			}
		})
	}
}

// The key reaches the Driver from the environment or the Driver Config, never
// from argv. A set-but-empty env value reads as unset, so it must fall through
// to the file instead of blanking it.
func TestParseArgsAPIKeyEnvAndDriverConfig(t *testing.T) {
	configPath := writeDriverConfig(t, `{"api_key":"sk-from-file"}`)
	empty := ""
	set := "sk-from-env"

	for _, tc := range []struct {
		name       string
		envAPIKey  *string
		envConfig  string
		wantAPIKey string
	}{
		{
			name:       "env alone",
			envAPIKey:  &set,
			wantAPIKey: "sk-from-env",
		},
		{
			name:       "config alone",
			envConfig:  configPath,
			wantAPIKey: "sk-from-file",
		},
		{
			name:       "env beats config",
			envAPIKey:  &set,
			envConfig:  configPath,
			wantAPIKey: "sk-from-env",
		},
		{
			name:       "set-but-empty env does not blank the config",
			envAPIKey:  &empty,
			envConfig:  configPath,
			wantAPIKey: "sk-from-file",
		},
		{
			name:       "nothing set",
			wantAPIKey: "",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// keyed by name: anything this case does not name (HOME included)
			// reads as unset
			lookup := func(name string) (string, bool) {
				switch name {
				case envAPIKey:
					if tc.envAPIKey == nil {
						return "", false
					}
					return *tc.envAPIKey, true
				case envConfigPath:
					return tc.envConfig, tc.envConfig != ""
				default:
					return "", false
				}
			}
			cfg, err := parseArgs([]string{"--model", "m", "p"}, nil, lookup)
			if err != nil {
				t.Fatalf("parseArgs returned error: %v", err)
			}
			if cfg.apiKey != tc.wantAPIKey {
				t.Errorf("apiKey = %q, want %q", cfg.apiKey, tc.wantAPIKey)
			}
		})
	}
}

// The two knobs that must never become flags: --api-key would put the secret in
// argv where process listings and shell history expose it, and --config would
// let a caller swap in a Driver Config the operator never approved. Both stay
// env-only, so the flag package itself has to reject them.
func TestParseArgsHasNoAPIKeyOrConfigFlag(t *testing.T) {
	for _, name := range []string{"--api-key", "--config"} {
		_, err := parseArgs([]string{"--model", "m", name, "sk-x", "p"}, nil, func(string) (string, bool) { return "", false })
		if err == nil {
			t.Fatalf("%s was accepted; it must stay env-only", name)
		}
		want := "flag provided but not defined: -" + strings.TrimPrefix(name, "--")
		if !strings.Contains(err.Error(), want) {
			t.Errorf("%s error = %v, want it to report %q", name, err, want)
		}
	}
}

// TestParseArgsAgentsDir: the Agent Files directory has no flag and no
// environment variable, so the Driver Config is the only thing above the
// home-relative default, and what it supplies is used verbatim.
func TestParseArgsAgentsDir(t *testing.T) {
	const homeDefault = "/home/dev/.config/spindrift-driver/agents"
	// keyed by name: an unnamed HOME stays unset in named(), an unnamed
	// override stays unset in homeOnly()
	named := func(path, home string) func(string) (string, bool) {
		return func(name string) (string, bool) {
			switch name {
			case envConfigPath:
				return path, true
			case "HOME":
				return home, home != ""
			}
			return "", false
		}
	}
	homeOnly := func(home string) func(string) (string, bool) {
		return func(name string) (string, bool) {
			if name == "HOME" {
				return home, true
			}
			return "", false
		}
	}
	resolve := func(t *testing.T, lookup func(string) (string, bool)) string {
		t.Helper()
		cfg, err := parseArgs([]string{"--model", "m", "p"}, nil, lookup)
		if err != nil {
			t.Fatalf("parseArgs returned error: %v", err)
		}
		return cfg.agentsDir
	}

	// The directory a configured agents_dir names has to exist for parseArgs to
	// return at all, so these resolve against real directories.
	t.Run("file supplies the directory and beats the default", func(t *testing.T) {
		agents := t.TempDir()
		path := writeDriverConfig(t, `{"agents_dir":"`+agents+`"}`)
		if got := resolve(t, named(path, "/home/dev")); got != agents {
			t.Errorf("agentsDir = %q, want %q", got, agents)
		}
	})

	t.Run("file value is verbatim", func(t *testing.T) {
		agents := t.TempDir()
		if err := os.Mkdir(filepath.Join(agents, "srv"), 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
		// Written with a "." element the string has to survive: filepath.Join
		// would clean it away and the test would assert nothing.
		verbatim := agents + "/./srv"
		path := writeDriverConfig(t, `{"agents_dir":"`+verbatim+`"}`)
		if got := resolve(t, named(path, "")); got != verbatim {
			t.Errorf("agentsDir = %q, want %q uncleaned", got, verbatim)
		}
	})

	t.Run("empty file value reads as unset", func(t *testing.T) {
		path := writeDriverConfig(t, `{"agents_dir":""}`)
		if got := resolve(t, named(path, "/home/dev")); got != homeDefault {
			t.Errorf("agentsDir = %q, want the default %q", got, homeDefault)
		}
	})

	t.Run("file without the key uses the default", func(t *testing.T) {
		path := writeDriverConfig(t, `{"base_url":"http://named:6/v1"}`)
		if got := resolve(t, named(path, "/home/dev")); got != homeDefault {
			t.Errorf("agentsDir = %q, want the default %q", got, homeDefault)
		}
	})

	t.Run("default file supplies the directory", func(t *testing.T) {
		agents := t.TempDir()
		home := writeDefaultDriverConfig(t, `{"agents_dir":"`+agents+`"}`)
		if got := resolve(t, homeOnly(home)); got != agents {
			t.Errorf("agentsDir = %q, want %q", got, agents)
		}
	})

	t.Run("HOME alone gives the home-relative default", func(t *testing.T) {
		if got := resolve(t, homeOnly("/home/dev")); got != homeDefault {
			t.Errorf("agentsDir = %q, want the default %q", got, homeDefault)
		}
	})

	t.Run("no HOME and no file resolves to nothing", func(t *testing.T) {
		if got := resolve(t, func(string) (string, bool) { return "", false }); got != "" {
			t.Errorf("agentsDir = %q, want empty, not an error", got)
		}
	})
}

// TestParseArgsAbsentDefaultAgentsDir: the home-relative default is never the
// operator's promise, so a directory that is simply not there reads as "no
// Agent Files" — no entries, no error, bash-only offer. A configured directory
// missing is fatal, which is the asymmetry this test pins from the other side.
func TestParseArgsAbsentDefaultAgentsDir(t *testing.T) {
	home := t.TempDir()
	cfg, err := parseArgs([]string{"--model", "m", "p"}, nil, func(name string) (string, bool) {
		if name == "HOME" {
			return home, true
		}
		return "", false
	})
	if err != nil {
		t.Fatalf("parseArgs error = %v, want none for an absent default directory", err)
	}
	if len(cfg.agents) != 0 {
		t.Errorf("agents = %+v, want none", cfg.agents)
	}
	if got := offeredNames(cfg.tools); !reflect.DeepEqual(got, []string{"bash"}) {
		t.Errorf("tools = %v, want [bash]", got)
	}
}

// TestParseArgsAbsentConfiguredAgentsDir: an agents_dir the Driver Config names
// is a directory the operator expects to be used, so its absence is a startup
// error naming it — exactly how SPINDRIFT_CONFIG treats a file it names.
func TestParseArgsAbsentConfiguredAgentsDir(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "agents")
	path := writeDriverConfig(t, `{"agents_dir":"`+missing+`"}`)
	_, err := parseArgs([]string{"--model", "m", "p"}, nil, func(name string) (string, bool) {
		if name == envConfigPath {
			return path, true
		}
		return "", false
	})
	if err == nil {
		t.Fatalf("parseArgs error = nil, want a startup error naming %q", missing)
	}
	if !strings.Contains(err.Error(), missing) {
		t.Errorf("error = %v, want it to name %q", err, missing)
	}
	if !errors.Is(err, errNoAgentDir) {
		t.Errorf("error = %v, want it to wrap errNoAgentDir", err)
	}
}

// TestParseArgsMalformedAgentFile: a broken Agent File is fatal wherever the
// directory came from — reading the operator's roster strictly is the only way
// a typo surfaces instead of silently shrinking the offer.
func TestParseArgsMalformedAgentFile(t *testing.T) {
	dir := t.TempDir()
	file := writeAgentFile(t, dir, "reviewer.md", "name: reviewer\nmodel: m-r\n")
	path := writeDriverConfig(t, `{"agents_dir":"`+dir+`"}`)
	_, err := parseArgs([]string{"--model", "m", "p"}, nil, func(name string) (string, bool) {
		if name == envConfigPath {
			return path, true
		}
		return "", false
	})
	if err == nil {
		t.Fatalf("parseArgs error = nil, want a startup error naming %q", file)
	}
	if !strings.Contains(err.Error(), file) {
		t.Errorf("error = %v, want it to name %q", err, file)
	}
}

// TestParseArgsLoadsRosterAndOffer: the roster loaded at startup is what the
// offer is derived from, so the two cannot disagree by the time sendChat runs.
func TestParseArgsLoadsRosterAndOffer(t *testing.T) {
	dir := t.TempDir()
	writeAgentFile(t, dir, "beta.md", "---\nname: beta\nmodel: m-b\n---\nb\n")
	writeAgentFile(t, dir, "alpha.md", "---\nname: alpha\nmodel: m-a\n---\na\n")
	path := writeDriverConfig(t, `{"agents_dir":"`+dir+`"}`)
	cfg, err := parseArgs([]string{"--model", "m", "p"}, nil, func(name string) (string, bool) {
		if name == envConfigPath {
			return path, true
		}
		return "", false
	})
	if err != nil {
		t.Fatalf("parseArgs error = %v", err)
	}
	if len(cfg.agents) != 2 || cfg.agents[0].Name != "alpha" || cfg.agents[1].Name != "beta" {
		t.Errorf("agents = %+v, want [alpha beta] in file-name order", cfg.agents)
	}
	if got := offeredNames(cfg.tools); !reflect.DeepEqual(got, []string{"bash", "subagent"}) {
		t.Errorf("tools = %v, want [bash subagent]", got)
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

	// The bound is idle, not wall-clock: responses stream, so what trips it is
	// silence, and a healthy stream may run far past the value. Text that still
	// describes a whole-attempt cap sends an operator to raise a knob that was
	// never the problem.
	if !strings.Contains(usage, "idle") {
		t.Errorf("usage does not describe --timeout as an idle bound:\n%s", usage)
	}
	if !regexp.MustCompile(`gap between [^.]*bytes`).MatchString(usage) {
		t.Errorf("usage does not say --timeout caps the gap between received bytes:\n%s", usage)
	}
	if !strings.Contains(usage, "response headers") {
		t.Errorf("usage does not name the response headers as a leg --timeout caps:\n%s", usage)
	}
	// The one case that really needs a raise: a server silent before its first
	// token, or a proxy that buffers. Ordinary long generations no longer do.
	if !regexp.MustCompile(`(?s)raise it[^.]*first token`).MatchString(usage) {
		t.Errorf("usage does not say to raise --timeout when the server is slow to emit its first token:\n%s", usage)
	}
	if regexp.MustCompile(`(?s)raise it[^.]*generat`).MatchString(usage) {
		t.Errorf("usage still tells the operator to raise --timeout for slow generation; a streaming turn no longer needs a raise:\n%s", usage)
	}
}

// -h is the only manual the Driver ships, so it has to point at the Driver
// Config: an operator who never opens the README still needs to learn that the
// file exists, where it lives and what beats what.
func TestUsageNamesDriverConfig(t *testing.T) {
	for _, want := range []string{"Driver Config", envConfigPath, defaultDriverConfigPath} {
		if !strings.Contains(usage, want) {
			t.Errorf("usage does not mention %q:\n%s", want, usage)
		}
	}
	if !indexOrder(usage, "flag", "environment", "Driver Config", "default") {
		t.Errorf("usage does not state the precedence flag, environment, Driver Config, default in that order:\n%s", usage)
	}
	// The key stays flag-less, and the usage text must not seed the idea that it
	// is otherwise.
	if strings.Contains(usage, "--api-key") {
		t.Errorf("usage advertises --api-key:\n%s", usage)
	}
	// The variable holds the path to the file; it is not a directory the file
	// sits in, and an operator who reads it the other way exports a folder.
	if regexp.MustCompile(`\bat \$` + envConfigPath + `\b`).MatchString(usage) {
		t.Errorf("usage says the Driver Config is at $%s itself; the variable names a path:\n%s", envConfigPath, usage)
	}
	if !regexp.MustCompile(`path \$` + envConfigPath).MatchString(usage) {
		t.Errorf("usage does not put the Driver Config at the path $%s names:\n%s", envConfigPath, usage)
	}
	// -h is the only manual the Driver ships, so the one lenient case has to be
	// stated there too: text that makes every unreadable file fatal sends an
	// operator to create a file that was never meant to exist.
	if !near(usage, "absent", "not an error", 160) {
		t.Errorf("usage does not say an absent file at the default path is not an error:\n%s", usage)
	}
}

func TestChatAttemptTimeoutIsMinutesScale(t *testing.T) {
	// One attempt can stream a whole long generation, so the default must be
	// minutes-scale; a sub-minute default would expire healthy turns.
	if chatAttemptTimeout < time.Minute {
		t.Errorf("chatAttemptTimeout = %v, want at least %v", chatAttemptTimeout, time.Minute)
	}
}

// assertStreamFields pins the streaming declaration on a chat request body.
// Every request the Driver makes is streaming, so there is no state in which
// these keys may be absent.
func assertStreamFields(t *testing.T, label string, raw []byte) {
	t.Helper()
	var top map[string]any
	if err := json.Unmarshal(raw, &top); err != nil {
		t.Fatalf("%s: body is not valid JSON: %v; body = %q", label, err, raw)
	}
	if v, ok := top["stream"]; !ok {
		t.Errorf("%s missing key %q; body = %s", label, "stream", raw)
	} else if v != true {
		t.Errorf("%s stream = %v, want true", label, v)
	}
	opts, ok := top["stream_options"].(map[string]any)
	if !ok {
		t.Errorf("%s stream_options = %v, want an object", label, top["stream_options"])
		return
	}
	if v, ok := opts["include_usage"]; !ok {
		t.Errorf("%s stream_options missing key %q; body = %s", label, "include_usage", raw)
	} else if v != true {
		t.Errorf("%s stream_options.include_usage = %v, want true", label, v)
	}
}

func TestChatRequestWireShape(t *testing.T) {
	req := chatRequest{
		Model: "m",
		Messages: []chatMessage{
			{Role: "assistant", ToolCalls: []toolCall{{ID: "call_1", Type: "function", Function: toolCallFunction{Name: "bash", Arguments: "{\"command\":\"ls\"}"}}}},
			{Role: "tool", Content: "file listing", ToolCallID: "call_1"},
		},
		Tools:         []toolSpec{bashTool},
		Stream:        true,
		StreamOptions: streamOptions{IncludeUsage: true},
	}
	raw, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var top map[string]any
	if err := json.Unmarshal(raw, &top); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, key := range []string{"model", "messages", "tools", "stream", "stream_options"} {
		if _, ok := top[key]; !ok {
			t.Errorf("request JSON missing key %q; body = %s", key, raw)
		}
	}
	assertStreamFields(t, "request JSON", raw)

	// The same fields must be on the body the Driver actually POSTs, not just
	// on a hand-built struct: sendChat owns the envelope.
	drv := newScriptedChatServer(t, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", drv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}
	sent := drv.requestBodies(t)
	if len(sent) != 1 {
		t.Fatalf("POSTs = %d, want 1", len(sent))
	}
	assertStreamFields(t, "sent request body", sent[0])

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

// TestSubagentToolWireShape: the Subagent Tool travels to the Model in the same
// envelope bash does, and its schema is the only thing the Model has to go on —
// the agent names themselves arrive in a later issue, so the description must
// not enumerate them.
func TestSubagentToolWireShape(t *testing.T) {
	raw, err := json.Marshal(subagentTool)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var top map[string]any
	if err := json.Unmarshal(raw, &top); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if top["type"] != "function" {
		t.Errorf("type = %v, want function", top["type"])
	}
	fn, _ := top["function"].(map[string]any)
	if fn["name"] != "subagent" {
		t.Errorf("function.name = %v, want subagent", fn["name"])
	}
	desc, _ := fn["description"].(string)
	if desc == "" {
		t.Error("function.description is empty: the Model cannot tell what the tool is for")
	}
	params, _ := fn["parameters"].(map[string]any)
	if params["type"] != "object" {
		t.Errorf("parameters.type = %v, want object", params["type"])
	}
	props, _ := params["properties"].(map[string]any)
	for _, name := range []string{"agent", "task"} {
		prop, _ := props[name].(map[string]any)
		if prop == nil || prop["type"] != "string" {
			t.Errorf("parameters.properties.%s = %v, want a string property", name, props[name])
		}
	}
	required, _ := params["required"].([]any)
	if len(required) != 2 {
		t.Fatalf("parameters.required = %v, want agent and task", params["required"])
	}
	want := map[string]bool{"agent": true, "task": true}
	for _, item := range required {
		name, _ := item.(string)
		if !want[name] {
			t.Errorf("parameters.required contains %v, want only agent and task", item)
		}
		delete(want, name)
	}
	if len(want) != 0 {
		t.Errorf("parameters.required = %v, want both agent and task", params["required"])
	}
}

// postedToolNames reads the tool offer back off a recorded POST body: what the
// Model was actually told, not what a hand-built struct was meant to hold.
func postedToolNames(t *testing.T, raw []byte) []string {
	t.Helper()
	var body struct {
		Tools []toolSpec `json:"tools"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("POST body is not valid JSON: %v; body = %q", err, raw)
	}
	return offeredNames(body.Tools)
}

// TestRunOffersBashOnlyWithNoAgentFiles is the issue's first acceptance
// criterion: with no Agent Files the run offers no Subagent Tool at all.
func TestRunOffersBashOnlyWithNoAgentFiles(t *testing.T) {
	srv := newScriptedChatServer(t, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}
	sent := srv.requestBodies(t)
	if len(sent) != 1 {
		t.Fatalf("POSTs = %d, want 1", len(sent))
	}
	if got := postedToolNames(t, sent[0]); !reflect.DeepEqual(got, []string{"bash"}) {
		t.Errorf("offered tools = %v, want [bash] only", got)
	}
}

// TestRunOffersSubagentToolWithAgentFiles: a roster the Driver Config points at
// is the only thing that puts the Subagent Tool in front of the Model.
func TestRunOffersSubagentToolWithAgentFiles(t *testing.T) {
	dir := t.TempDir()
	writeAgentFile(t, dir, "reviewer.md", "---\nname: reviewer\nmodel: m-r\n---\nreview things\n")
	cfgPath := writeDriverConfig(t, `{"agents_dir":"`+dir+`"}`)
	srv := newScriptedChatServer(t, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(name string) (string, bool) {
		if name == envConfigPath {
			return cfgPath, true
		}
		return "", false
	})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}
	sent := srv.requestBodies(t)
	if len(sent) != 1 {
		t.Fatalf("POSTs = %d, want 1", len(sent))
	}
	if got := postedToolNames(t, sent[0]); !reflect.DeepEqual(got, []string{"bash", "subagent"}) {
		t.Errorf("offered tools = %v, want [bash subagent]", got)
	}
}

// agentFixture writes one Agent File into a fresh directory and returns the path
// of a Driver Config naming that directory: the roster a --agent test resolves
// its entry against.
func agentFixture(t *testing.T, name, model, body string) string {
	t.Helper()
	dir := t.TempDir()
	writeAgentFile(t, dir, name+".md", "---\nname: "+name+"\nmodel: "+model+"\n---\n"+body+"\n")
	return writeDriverConfig(t, `{"agents_dir":"`+dir+`"}`)
}

// TestParseArgsAgentSelectsRosterEntry: --agent is what makes a run a child, so
// all three facts the entry carries land on the config — the name it was
// selected by, its model id and its body as the system prompt.
func TestParseArgsAgentSelectsRosterEntry(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-r", "review things")
	cfg, err := parseArgs([]string{"--agent", "reviewer", "p"}, nil, driverConfigEnv(cfgPath))
	if err != nil {
		t.Fatalf("parseArgs returned error: %v", err)
	}
	if cfg.agent != "reviewer" {
		t.Errorf("agent = %q, want %q", cfg.agent, "reviewer")
	}
	if cfg.model != "m-r" {
		t.Errorf("model = %q, want the entry's model id %q", cfg.model, "m-r")
	}
	if cfg.systemPrompt != "review things" {
		t.Errorf("systemPrompt = %q, want the Agent File body %q", cfg.systemPrompt, "review things")
	}
}

// TestParseArgsModelFlagBeatsAgentEntry keeps the documented ladder intact: an
// explicit --model still outranks the entry, while the entry keeps supplying the
// system prompt it exists for.
func TestParseArgsModelFlagBeatsAgentEntry(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-r", "review things")
	cfg, err := parseArgs([]string{"--agent", "reviewer", "--model", "m-flag", "p"}, nil, driverConfigEnv(cfgPath))
	if err != nil {
		t.Fatalf("parseArgs returned error: %v", err)
	}
	if cfg.model != "m-flag" {
		t.Errorf("model = %q, want the flag value %q to win over the entry", cfg.model, "m-flag")
	}
	if cfg.systemPrompt != "review things" {
		t.Errorf("systemPrompt = %q, want the entry body even when --model wins", cfg.systemPrompt)
	}
}

// TestParseArgsModelRequiredStillFires guards the move of the --model check to
// after the roster resolves (an entry can satisfy it): with neither flag the
// message is the one every existing test pins, word for word.
func TestParseArgsModelRequiredStillFires(t *testing.T) {
	_, err := parseArgs([]string{"p"}, nil, func(string) (string, bool) { return "", false })
	if err == nil || err.Error() != "--model is required" {
		t.Errorf("err = %v, want exactly %q", err, "--model is required")
	}
}

// TestParseArgsUnknownAgent: a name the roster does not contain is a startup
// error naming it. The available names are deliberately not listed — issue #26
// owns the roster-policy wording.
func TestParseArgsUnknownAgent(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-r", "review things")
	_, err := parseArgs([]string{"--agent", "ghost", "--model", "m", "p"}, nil, driverConfigEnv(cfgPath))
	if err == nil {
		t.Fatal("parseArgs accepted an agent the roster does not name")
	}
	if want := `unknown agent "ghost"`; err.Error() != want {
		t.Errorf("err = %v, want exactly %q", err, want)
	}
}

// TestRunUnknownAgentExitsTwo: the unknown name travels the same startup-error
// path as every other parseArgs error — exit 2, one error event, and the Model
// is never contacted.
func TestRunUnknownAgentExitsTwo(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-r", "review things")
	srv := newScriptedChatServer(t, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--agent", "ghost", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, driverConfigEnv(cfgPath))
	if code != 2 {
		t.Fatalf("exit code = %d, want 2 for a startup error; stderr = %q", code, stderr.String())
	}
	if got := oneErrorEvent(t, stdout.String()); got != `unknown agent "ghost"` {
		t.Errorf("error event text = %q, want %q", got, `unknown agent "ghost"`)
	}
	if n := len(srv.requestBodies(t)); n != 0 {
		t.Errorf("POSTs = %d, want 0: a startup error never reaches the server", n)
	}
}

// TestRunAgentModelPostedVerbatim: the entry's model id is the request's model
// id. An id carrying a vendor prefix, a "{"-free colon and an effort suffix is
// what a router actually names, and nothing in the Driver splits or normalises
// it — asserted on the POST body, because a server that omits a model from its
// reply would mask any mangle in the usage event.
func TestRunAgentModelPostedVerbatim(t *testing.T) {
	const modelID = "vendor/model-x:2025-01-01-effort-high"
	cfgPath := agentFixture(t, "thinker", modelID, "think hard")
	srv := newScriptedChatServer(t, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--agent", "thinker", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, driverConfigEnv(cfgPath))
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}
	sent := srv.requestBodies(t)
	if len(sent) != 1 {
		t.Fatalf("POSTs = %d, want 1", len(sent))
	}
	var body struct {
		Model    string            `json:"model"`
		Messages []recordedMessage `json:"messages"`
	}
	if err := json.Unmarshal(sent[0], &body); err != nil {
		t.Fatalf("POST body is not valid JSON: %v; body = %q", err, sent[0])
	}
	if body.Model != modelID {
		t.Errorf("posted model = %q, want the entry id verbatim %q", body.Model, modelID)
	}
	if len(body.Messages) != 2 {
		t.Fatalf("messages = %d, want 2", len(body.Messages))
	}
	if body.Messages[0].Role != "system" || body.Messages[0].Content != "think hard" {
		t.Errorf("system message = %+v, want the Agent File body as the system prompt", body.Messages[0])
	}
}

// TestRunChildNotOfferedSubagentTool: a run selected by --agent is a child, and
// a child holding the Subagent Tool could spawn without bound. The top-level
// case rides along so the suppression cannot leak upward out of the child.
func TestRunChildNotOfferedSubagentTool(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-r", "review things")
	for _, tc := range []struct {
		name  string
		agent []string
		want  []string
	}{
		{name: "child", agent: []string{"--agent", "reviewer"}, want: []string{"bash"}},
		{name: "top level keeps the tool", agent: nil, want: []string{"bash", "subagent"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := newScriptedChatServer(t, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
			args := append([]string{"p"}, tc.agent...)
			args = append(args, "--model", "m", "--base-url", srv.srv.URL)
			var stdout, stderr bytes.Buffer
			code := run(args, strings.NewReader(""), &stdout, &stderr, driverConfigEnv(cfgPath))
			if code != 0 {
				t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
			}
			sent := srv.requestBodies(t)
			if len(sent) != 1 {
				t.Fatalf("POSTs = %d, want 1", len(sent))
			}
			if got := postedToolNames(t, sent[0]); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("offered tools = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestRunSystemPromptFallsBackToDefault: every caller of run that never went
// through parseArgs hands over a config with an empty systemPrompt field, and
// those runs have always started from the package prompt. The fallback keeps
// them byte-identical.
func TestRunSystemPromptFallsBackToDefault(t *testing.T) {
	srv := newScriptedChatServer(t, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q", code, stderr.String())
	}
	msgs := srv.messageBodies(t)
	if len(msgs) != 1 || len(msgs[0]) != 2 {
		t.Fatalf("messages = %+v, want one POST of two messages", msgs)
	}
	if msgs[0][0].Role != "system" || msgs[0][0].Content != systemPrompt {
		t.Errorf("system message = %+v, want the package default prompt %q", msgs[0][0], systemPrompt)
	}
}

// TestUsageDocumentsAgentFlag: --help is the only manual the Driver ships, so it
// has to carry the flag in the copyable usage line and say what the entry
// supplies — an operator who cannot tell that from -h passes --model by habit
// and wonders why the prompt changed.
func TestUsageDocumentsAgentFlag(t *testing.T) {
	if !strings.Contains(usage, "--agent") {
		t.Errorf("usage does not document the --agent flag:\n%s", usage)
	}
	if !regexp.MustCompile(`usage: spindrift-driver.*\[--agent <name>\]`).MatchString(usage) {
		t.Errorf("usage line does not list [--agent <name>]:\n%s", usage)
	}
	if !strings.Contains(usage, "Agent File") {
		t.Errorf("usage does not say --agent selects an Agent File entry:\n%s", usage)
	}
	if !regexp.MustCompile(`(?s)--agent <name>[^.]*model`).MatchString(usage) {
		t.Errorf("usage does not say --agent supplies the entry's model id:\n%s", usage)
	}
	if !regexp.MustCompile(`(?s)--agent <name>[^.]*system prompt`).MatchString(usage) {
		t.Errorf("usage does not say the Agent File body is the system prompt:\n%s", usage)
	}
	if !regexp.MustCompile(`(?s)--agent.*never offered\s+the Subagent Tool`).MatchString(usage) {
		t.Errorf("usage does not say a run selected by --agent is never offered the Subagent Tool:\n%s", usage)
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
		writeSSE(w, sseChatCompletion(`{"choices":[{"message":{"role":"assistant","content":"ends with newline\n"}}]}`))
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

// TestParseArgsBrokenDriverConfigOutranksFlagValidation pins the order: a file
// the operator must fix is reported before any complaint about argv, because
// fixing the arguments would not make the run possible anyway.
func TestParseArgsBrokenDriverConfigOutranksFlagValidation(t *testing.T) {
	path := writeDriverConfig(t, "nope")
	lookup := func(name string) (string, bool) {
		if name == envConfigPath {
			return path, true
		}
		return "", false
	}
	_, err := parseArgs([]string{"--model", "m", "--max-turns", "0", "p"}, nil, lookup)
	if err == nil {
		t.Fatal("parseArgs returned nil, want the config error")
	}
	if !strings.Contains(err.Error(), "driver config") || !strings.Contains(err.Error(), path) {
		t.Errorf("err = %v, want it to name the driver config file %q", err, path)
	}
	if strings.Contains(err.Error(), "--max-turns") {
		t.Errorf("err = %v, want the config error, not the flag complaint", err)
	}
}

// driverConfigEnv is the env stub for a run whose Driver Config is named by
// hand: SPINDRIFT_CONFIG points at path, every other variable reads as unset --
// including HOME, so no default file can stand in for the one that is broken.
func driverConfigEnv(path string) func(string) (string, bool) {
	return func(name string) (string, bool) {
		switch name {
		case envConfigPath:
			return path, true
		default:
			return "", false
		}
	}
}

// assertNoStartEvent checks the stream never opened a Model call: a startup
// failure must be the only thing stdout carries.
func assertNoStartEvent(t *testing.T, raw string) {
	t.Helper()
	for _, ev := range stdoutEvents(t, raw) {
		if ev["kind"] == "start" {
			t.Errorf("stdout carries a start event before the config error: %q", raw)
		}
	}
}

// A Driver Config the operator named by hand has to exist. Startup fails in the
// existing exit-2 class, and the one error event names the file and the problem
// so a reader of stdout alone learns both; stderr carries the same news for the
// human at the terminal.
func TestRunDriverConfigMissingNamedFileFailsStartup(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.json")
	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m"}, strings.NewReader(""), &stdout, &stderr, driverConfigEnv(path))
	if code != 2 {
		t.Fatalf("exit code = %d, want 2 for a startup error; stderr = %q", code, stderr.String())
	}
	if got := oneErrorEvent(t, stdout.String()); !strings.Contains(got, path) || !strings.Contains(got, "file not found") {
		t.Errorf("error event text = %q, want it to name %q and report the file missing", got, path)
	}
	assertNoStartEvent(t, stdout.String())
	if !strings.Contains(stderr.String(), path) {
		t.Errorf("stderr = %q, want the operator copy of the failure naming %q", stderr.String(), path)
	}
}

// A truncated file is not "a file that said nothing": parsing it fails, the run
// exits 2 with exactly one error event and no start event, so a reader knows no
// Model call was ever attempted with half-read settings.
func TestRunDriverConfigMalformedJSONFailsStartup(t *testing.T) {
	path := writeDriverConfig(t, `{"base_url": `)
	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m"}, strings.NewReader(""), &stdout, &stderr, driverConfigEnv(path))
	if code != 2 {
		t.Fatalf("exit code = %d, want 2 for a startup error; stderr = %q", code, stderr.String())
	}
	if got := oneErrorEvent(t, stdout.String()); !strings.Contains(got, path) || !strings.Contains(got, "malformed JSON") {
		t.Errorf("error event text = %q, want it to name %q and say malformed JSON", got, path)
	}
	assertNoStartEvent(t, stdout.String())
	if !strings.Contains(stderr.String(), path) {
		t.Errorf("stderr = %q, want the operator copy naming %q", stderr.String(), path)
	}
}

// An unrecognised key is refused rather than ignored: the nearest known field is
// not guessed at, so a typo cannot quietly leave the run on defaults -- and a
// key that names a runtime knob cannot smuggle it in either.
func TestRunDriverConfigUnknownFieldFailsStartup(t *testing.T) {
	path := writeDriverConfig(t, `{"apiKEy":"sk-x"}`)
	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m"}, strings.NewReader(""), &stdout, &stderr, driverConfigEnv(path))
	if code != 2 {
		t.Fatalf("exit code = %d, want 2 for a startup error; stderr = %q", code, stderr.String())
	}
	got := oneErrorEvent(t, stdout.String())
	if !strings.Contains(got, path) || !strings.Contains(got, "apiKEy") {
		t.Errorf("error event text = %q, want it to name %q and the offending key apiKEy", got, path)
	}
	assertNoStartEvent(t, stdout.String())
	if !strings.Contains(stderr.String(), "apiKEy") {
		t.Errorf("stderr = %q, want the operator copy to name the offending key", stderr.String())
	}

	// model is deliberately outside the file schema: a config that sets it must
	// be refused outright, never honoured in place of the required --model.
	smuggle := writeDriverConfig(t, `{"model":"sneaky"}`)
	var stdout2, stderr2 bytes.Buffer
	code = run([]string{"p"}, strings.NewReader(""), &stdout2, &stderr2, driverConfigEnv(smuggle))
	if code == 0 {
		t.Fatal("run exited 0, want a refusal: a config field must not supply the model")
	}
	if code != 2 {
		t.Errorf("exit code = %d, want 2 for a startup error; stderr = %q", code, stderr2.String())
	}
	if got := oneErrorEvent(t, stdout2.String()); !strings.Contains(got, "model") {
		t.Errorf("error event text = %q, want it to name the offending model key", got)
	}
	assertNoStartEvent(t, stdout2.String())
}

// A field of the wrong JSON type is reported with the type that was found, not
// with the type the reader wished for: "got number" is what tells the operator
// to quote the value.
func TestRunDriverConfigWrongTypeFailsStartup(t *testing.T) {
	path := writeDriverConfig(t, `{"base_url": 8080}`)
	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m"}, strings.NewReader(""), &stdout, &stderr, driverConfigEnv(path))
	if code != 2 {
		t.Fatalf("exit code = %d, want 2 for a startup error; stderr = %q", code, stderr.String())
	}
	got := oneErrorEvent(t, stdout.String())
	if !strings.Contains(got, path) || !strings.Contains(got, "base_url") || !strings.Contains(got, "number") {
		t.Errorf("error event text = %q, want it to name %q, the field base_url and the kind found (number)", got, path)
	}
	assertNoStartEvent(t, stdout.String())
	if !strings.Contains(stderr.String(), "base_url") {
		t.Errorf("stderr = %q, want the operator copy to name the offending field", stderr.String())
	}
}

// The default file is allowed to be absent: a home directory without
// ~/.config/spindrift-driver/config.json is "the file said nothing", so the run
// proceeds on flags exactly as it would with no config feature at all.
func TestRunDefaultDriverConfigAbsentIsNotAnError(t *testing.T) {
	home := t.TempDir()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeSSE(w, sseChatCompletion(`{"model":"m","choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m", "--base-url", srv.URL}, strings.NewReader(""), &stdout, &stderr, func(name string) (string, bool) {
		switch name {
		case "HOME":
			return home, true
		default:
			return "", false
		}
	})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 with no default config file; stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "ok") {
		t.Errorf("stdout = %q, want the answer from the fake server", stdout.String())
	}
	resultEvent(t, stdout.String())
}

// --help is how an operator recovers from a file they just broke, so it must not
// touch the config: the usage still comes out on stderr at exit 0.
func TestRunHelpIgnoresBrokenDriverConfig(t *testing.T) {
	path := writeDriverConfig(t, `{"base_url": `)
	for _, arg := range []string{"--help", "-h"} {
		var stdout, stderr bytes.Buffer
		code := run([]string{arg}, strings.NewReader(""), &stdout, &stderr, driverConfigEnv(path))
		if code != 0 {
			t.Errorf("%s with a broken config: exit code = %d, want 0; stderr = %q", arg, code, stderr.String())
		}
		if stdout.Len() != 0 {
			t.Errorf("%s: stdout = %q, want nothing: stdout carries only JSONL events", arg, stdout.String())
		}
		if !strings.Contains(stderr.String(), "usage:") {
			t.Errorf("%s: stderr = %q, want usage text", arg, stderr.String())
		}
	}
}

// The Driver reads no config from the Target repo: nothing in the working
// directory is ever consulted, so a checkout cannot steer the endpoint, the
// secret or the timeout of a run started inside it.
func TestParseArgsReadsNoConfigFromWorkingDirectory(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	planted := map[string]string{
		"spindrift.json":         `{"base_url":"http://planted.invalid/spindrift/v1","api_key":"sk-planted-spindrift","timeout":"7s"}`,
		"config.json":            `{"base_url":"http://planted.invalid/config/v1","api_key":"sk-planted-config","timeout":"8s"}`,
		".spindrift-driver.json": `{"base_url":"http://planted.invalid/dotfile/v1","api_key":"sk-planted-dotfile","timeout":"9s"}`,
		".env":                   "SPINDRIFT_BASE_URL=http://planted.invalid/dotenv/v1\nSPINDRIFT_API_KEY=sk-planted-dotenv\nSPINDRIFT_TIMEOUT=10s\n",
	}
	for name, body := range planted {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	// The fixtures have to be real and have to disagree with the defaults, or
	// this test could pass by writing nothing at all.
	for name, body := range planted {
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("planted %s is not readable after WriteFile: %v", name, err)
		}
		if string(raw) != body {
			t.Fatalf("planted %s = %q, want the bytes written to it", name, string(raw))
		}
		if !strings.Contains(body, "planted.invalid") {
			t.Fatalf("planted %s carries no non-default base URL: %q", name, body)
		}
	}

	cfg, err := parseArgs([]string{"--model", "m", "p"}, nil, func(string) (string, bool) { return "", false })
	if err != nil {
		t.Fatalf("parseArgs returned error: %v", err)
	}
	if cfg.baseURL != "http://localhost:8080/v1" {
		t.Errorf("baseURL = %q, want the built-in default: something in the working directory was read", cfg.baseURL)
	}
	if cfg.apiKey != "" {
		t.Errorf("apiKey = %q, want empty: something in the working directory was read", cfg.apiKey)
	}
	if cfg.timeout != chatAttemptTimeout {
		t.Errorf("timeout = %v, want the built-in %v: something in the working directory was read", cfg.timeout, chatAttemptTimeout)
	}
}
