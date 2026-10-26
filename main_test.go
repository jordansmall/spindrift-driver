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

	if stdout.String() != "here is the fix\n" {
		t.Errorf("stdout = %q, want exactly the response text", stdout.String())
	}
	stderrLines := strings.Count(strings.TrimSuffix(stderr.String(), "\n"), "\n") + 1
	if stderrLines != 1 {
		t.Errorf("stderr has %d lines, want 1 progress line: %q", stderrLines, stderr.String())
	}
	if !strings.Contains(stderr.String(), "/chat/completions") {
		t.Errorf("stderr = %q, want a progress line naming the request target", stderr.String())
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
	if stdout.String() != "all done\n" {
		t.Errorf("stdout = %q, want final answer with trailing newline", stdout.String())
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
	if stdout.String() != "gave up gracefully\n" {
		t.Errorf("stdout = %q, want the final answer", stdout.String())
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
	if stdout.String() != "truncation acknowledged\n" {
		t.Errorf("stdout = %q, want the final answer after the flooded turn", stdout.String())
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
	if stdout.String() != "recovered from timeout\n" {
		t.Errorf("stdout = %q, want the final answer after the timed-out turn", stdout.String())
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
	if stdout.Len() != 0 {
		t.Errorf("stdout = %q, want empty (no final answer was produced)", stdout.String())
	}
	if bodies := srv.requestBodies(t); len(bodies) != 3 {
		t.Errorf("POSTs = %d, want exactly max-turns 3", len(bodies))
	}
	if !strings.Contains(stderr.String(), "turn") {
		t.Errorf("stderr = %q, want a message about exhausting the turns", stderr.String())
	}
}

func TestRunServerErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not found", http.StatusInternalServerError)
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := run([]string{"--model", "m", "--base-url", srv.URL, "p"}, nil, &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code == 0 {
		t.Fatal("non-200 response should exit non-zero")
	}
	if stdout.Len() != 0 {
		t.Errorf("stdout = %q, want empty on server error", stdout.String())
	}
	if !strings.Contains(stderr.String(), "500") {
		t.Errorf("stderr = %q, want the failing status", stderr.String())
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
	if stdout.Len() != 0 {
		t.Errorf("stdout = %q, want empty on empty response", stdout.String())
	}
	if !strings.Contains(stderr.String(), "choices") {
		t.Errorf("stderr = %q, want a message about the empty response", stderr.String())
	}
}

func TestRunMissingModel(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run([]string{"say hi"}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code == 0 {
		t.Fatal("run with missing --model should exit non-zero")
	}
	if stdout.Len() != 0 {
		t.Errorf("stdout = %q, want empty on usage error", stdout.String())
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
	if stdout.Len() != 0 {
		t.Errorf("stdout = %q, want empty on usage error", stdout.String())
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
	if stdout.String() != "ends with newline\n" {
		t.Errorf("stdout = %q, want single trailing newline (no doubling)", stdout.String())
	}
}

func TestRunHelpExitsZeroOnStdout(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run([]string{"-h"}, strings.NewReader(""), &stdout, &stderr, func(string) (string, bool) { return "", false })
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for -h", code)
	}
	if !strings.Contains(stdout.String(), "usage:") {
		t.Errorf("stdout = %q, want usage text", stdout.String())
	}
	if stderr.Len() != 0 {
		t.Errorf("stderr = %q, want empty for -h", stderr.String())
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

func TestParseArgsExtraPositional(t *testing.T) {
	_, err := parseArgs([]string{"--model", "m", "p1", "p2"}, nil, func(string) (string, bool) { return "", false })
	if err == nil {
		t.Fatal("parseArgs should reject more than one positional prompt")
	}
	if !strings.Contains(err.Error(), "p2") {
		t.Errorf("err = %v, want it to name the extra argument", err)
	}
}
