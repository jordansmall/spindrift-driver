package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
)

// subagentHelperEnv names the test binary's helper-process branch in TestMain:
// set, the binary IS the Driver, which is what lets a spawn test run a real
// child without building a second binary.
const subagentHelperEnv = "SPINDRIFT_SUBAGENT_HELPER"

// subagentEnv points a run and every child it spawns at one roster and one fake
// endpoint. A child inherits the environment rather than a config struct, so
// these variables are the only channel to it — and they only reach the parent's
// own parseArgs if the test passes os.LookupEnv, which is also what lets parent
// and child resolve the same roster.
func subagentEnv(t *testing.T, cfgPath, baseURL string) {
	t.Helper()
	t.Setenv(envConfigPath, cfgPath)
	t.Setenv(envBaseURL, baseURL)
	t.Setenv(subagentHelperEnv, "1")
}

// parentToolCallResponse scripts the parent's first turn: one Subagent Tool call
// and no prose. The call's arguments are escaped by hand because the wire nests
// them — a JSON object inside a JSON string inside the completion body.
const parentToolCallResponse = `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_s","type":"function","function":{"name":"subagent","arguments":"{\"agent\":\"reviewer\",\"task\":\"review the diff\"}"}}]}}]}`

// completion is the shorthand for a scripted final answer.
func completion(text string) string {
	return `{"choices":[{"message":{"role":"assistant","content":"` + text + `"}}]}`
}

// postedSpawnRequest is the part of a recorded POST body a spawn test reads: the
// model id sent verbatim, the messages that opened the conversation, and the
// tool offer attached to it.
type postedSpawnRequest struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	Tools []toolSpec `json:"tools"`
}

func postedSpawn(t *testing.T, raw []byte) postedSpawnRequest {
	t.Helper()
	var body postedSpawnRequest
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("POST body is not valid JSON: %v; body = %q", err, raw)
	}
	return body
}

// relayedEvents returns the lines a child produced, told apart from the parent's
// own only by in_subagent — the relay reuses every kind the child emits.
func relayedEvents(t *testing.T, raw string) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, ev := range stdoutEvents(t, raw) {
		if ev["in_subagent"] == true {
			out = append(out, ev)
		}
	}
	return out
}

// parentResult returns the run's own closing result. resultEvent cannot be used
// on a relaying run: the child's result line shares the kind, and in_subagent is
// the only thing that separates the two.
func parentResult(t *testing.T, raw string) map[string]any {
	t.Helper()
	events := stdoutEvents(t, raw)
	var found map[string]any
	for _, ev := range events {
		if ev["kind"] == "result" && ev["in_subagent"] != true {
			if found != nil {
				t.Fatalf("stdout carries more than one parent result event: %q", raw)
			}
			found = ev
		}
	}
	if found == nil {
		t.Fatalf("stdout carries no parent result event: %q", raw)
	}
	if events[len(events)-1]["kind"] != "result" || events[len(events)-1]["in_subagent"] == true {
		t.Fatalf("the last stdout line is not the parent's own result: %q", raw)
	}
	return found
}

// spawnPair finds the parent's subagent tool_call/tool_result pair with the
// relayed child lines allowed between them: toolPair pins adjacency, and a
// relaying run puts the child's whole stream in the gap by design.
func spawnPair(t *testing.T, raw string) (callEv, resultEv map[string]any) {
	t.Helper()
	events := stdoutEvents(t, raw)
	callIdx, resultIdx := -1, -1
	for i, ev := range events {
		if ev["in_subagent"] == true || ev["tool_name"] != "subagent" {
			continue
		}
		switch ev["kind"] {
		case "tool_call":
			if callIdx >= 0 {
				t.Fatalf("stdout carries more than one subagent tool_call: %q", raw)
			}
			callIdx = i
		case "tool_result":
			if resultIdx >= 0 {
				t.Fatalf("stdout carries more than one subagent tool_result: %q", raw)
			}
			resultIdx = i
		}
	}
	if callIdx < 0 || resultIdx < 0 || resultIdx < callIdx {
		t.Fatalf("stdout carries no subagent tool_call/tool_result pair (call %d, result %d) in %q", callIdx, resultIdx, raw)
	}
	return events[callIdx], events[resultIdx]
}

// subagentReplyServer answers one model id with a 500 and every other POST with
// the next scripted completion. The POST-counting scriptedChatServer cannot fail
// only the child: the child retries, and both runs share one endpoint, so the
// routing key has to be the model in the body.
type subagentReplyServer struct {
	mu        sync.Mutex
	failModel string
	replies   []string
	answered  int
	bodies    [][]byte
	srv       *httptest.Server
}

func newSubagentReplyServer(t *testing.T, failModel string, replies ...string) *subagentReplyServer {
	s := &subagentReplyServer{failModel: failModel, replies: replies}
	s.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		s.bodies = append(s.bodies, body)
		var req struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(body, &req)
		fail := req.Model == failModel
		if !fail {
			if s.answered < len(s.replies) {
				reply := s.replies[s.answered]
				s.answered++
				s.mu.Unlock()
				writeSSE(w, sseChatCompletion(reply))
				return
			}
			fail = true
		}
		s.mu.Unlock()
		http.Error(w, "scripted failure", http.StatusInternalServerError)
	}))
	t.Cleanup(s.srv.Close)
	return s
}

func (s *subagentReplyServer) requestBodies(t *testing.T) [][]byte {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([][]byte, len(s.bodies))
	copy(out, s.bodies)
	return out
}

// TestSubagentSpawnRunsTheAgentFileEntry is the issue's core criterion: the child
// is a whole Driver run started from the roster entry, so the model id goes out
// verbatim and the Agent File body is what its conversation opens with.
func TestSubagentSpawnRunsTheAgentFileEntry(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-reviewer", "review things")
	srv := newScriptedChatServer(t,
		parentToolCallResponse,
		completion("child answer"),
		completion("parent answer"),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"top-level prompt", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}

	sent := srv.requestBodies(t)
	if len(sent) != 3 {
		t.Fatalf("POSTs = %d, want parent, child, parent: %q", len(sent), stdout.String())
	}
	if got := postedSpawn(t, sent[0]).Model; got != "m-parent" {
		t.Errorf("parent sent model %q, want the --model value", got)
	}
	child := postedSpawn(t, sent[1])
	if child.Model != "m-reviewer" {
		t.Errorf("child sent model %q, want the Agent File model id verbatim", child.Model)
	}
	if len(child.Messages) != 2 {
		t.Fatalf("child sent %d messages, want the system prompt and the task: %+v", len(child.Messages), child.Messages)
	}
	if got := child.Messages[0]; got.Role != "system" || got.Content != "review things" {
		t.Errorf("child system message = %+v, want role system with the Agent File body", got)
	}
	if got := child.Messages[1]; got.Role != "user" || got.Content != "review the diff" {
		t.Errorf("child user message = %+v, want role user with the task", got)
	}
}

// TestSubagentRelaysChildEventsIntoParentStream pins where a child's events go:
// inside the parent's tool_call/tool_result pair, attributed by role, and never
// stamped as the parent's own.
func TestSubagentRelaysChildEventsIntoParentStream(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-reviewer", "review things")
	srv := newScriptedChatServer(t,
		parentToolCallResponse,
		completion("child answer"),
		completion("parent answer"),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}

	events := stdoutEvents(t, stdout.String())
	callIdx, resultIdx := -1, -1
	for i, ev := range events {
		if ev["tool_name"] == "subagent" && ev["in_subagent"] != true {
			switch ev["kind"] {
			case "tool_call":
				callIdx = i
			case "tool_result":
				resultIdx = i
			}
		}
	}
	if callIdx < 0 || resultIdx < 0 {
		t.Fatalf("no subagent tool_call/tool_result pair in %q", stdout.String())
	}

	relayed := relayedEvents(t, stdout.String())
	if len(relayed) == 0 {
		t.Fatalf("no relayed child events in %q", stdout.String())
	}
	// The whole child stream has to come across, closing result included: it is
	// where a reader sums the child's usage from, and in_subagent is what keeps it
	// distinct from the parent's own closing result.
	missing := map[string]bool{"start": true, "text": true, "usage": true, "result": true}
	for _, ev := range relayed {
		if got := ev["role"]; got != "reviewer" {
			t.Errorf("relayed %v event role = %v, want the agent name", ev["kind"], got)
		}
		if k, ok := ev["kind"].(string); ok {
			delete(missing, k)
		}
	}
	for kind := range missing {
		t.Errorf("relayed stream carries no %q event; child events = %v", kind, relayed)
	}
	// The child's lines have to sit in the gap: a relay that ran ahead of the
	// call, or trailed the result, would attribute its prose to the wrong turn.
	for i, ev := range events {
		if ev["in_subagent"] == true && (i <= callIdx || i >= resultIdx) {
			t.Errorf("relayed %v event sits outside the tool_call/tool_result pair (call %d, result %d, event %d)", ev["kind"], callIdx, resultIdx, i)
		}
	}
}

// TestSubagentToolResultNamesTheAgent: subagent and tool_target are what let a
// reader see which roster entry produced a Tool Result.
func TestSubagentToolResultNamesTheAgent(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-reviewer", "review things")
	srv := newScriptedChatServer(t,
		parentToolCallResponse,
		completion("child answer"),
		completion("parent answer"),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}

	callEv, resultEv := spawnPair(t, stdout.String())
	if got := callEv["tool_name"]; got != "subagent" {
		t.Errorf("tool_call tool_name = %v, want subagent", got)
	}
	if want := map[string]any{"agent": "reviewer", "task": "review the diff"}; !reflect.DeepEqual(callEv["tool_input"], want) {
		t.Errorf("tool_call tool_input = %#v, want the decoded %+v", callEv["tool_input"], want)
	}
	for _, field := range []string{"subagent", "tool_target"} {
		if got := resultEv[field]; got != "reviewer" {
			t.Errorf("tool_result %s = %v, want the agent name", field, got)
		}
	}
	if got := resultEv["tool_output"]; got != "child answer" {
		t.Errorf("tool_result tool_output = %v, want the child's answer verbatim", got)
	}
	if v, ok := resultEv["is_error"]; ok {
		t.Errorf("tool_result carries is_error = %v on a succeeding subagent run", v)
	}
}

// TestSubagentAnswerDoesNotBecomeParentResult keeps the two answers apart: the
// child's answer is a Tool Result, the parent's is the run's result.
func TestSubagentAnswerDoesNotBecomeParentResult(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-reviewer", "review things")
	srv := newScriptedChatServer(t,
		parentToolCallResponse,
		completion("child answer"),
		completion("parent answer"),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}
	if got := parentResult(t, stdout.String())["text"]; got != "parent answer" {
		t.Errorf("parent result text = %v, want the parent's own final answer", got)
	}
	var childResultText any
	for _, ev := range relayedEvents(t, stdout.String()) {
		if ev["kind"] == "result" {
			childResultText = ev["text"]
		}
	}
	if childResultText != "child answer" {
		t.Errorf("relayed result text = %v, want the child's own answer relayed as its own result event", childResultText)
	}
	// The child's answer reached the parent twice over: as the relayed prose and
	// as the Tool Result the next turn reads.
	if _, resultEv := spawnPair(t, stdout.String()); resultEv["tool_output"] != "child answer" {
		t.Errorf("tool_output = %v, want the child's answer", resultEv["tool_output"])
	}
}

// TestSubagentFailureIsAnErrorToolResult: a child that dies is an ordinary error
// shaped Tool Result, and the parent still finishes its own run.
func TestSubagentFailureIsAnErrorToolResult(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-reviewer", "review things")
	srv := newSubagentReplyServer(t, "m-reviewer",
		parentToolCallResponse,
		completion("parent answer"),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 — a failed child is a Tool Result, not a failed run; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}
	if got := parentResult(t, stdout.String())["text"]; got != "parent answer" {
		t.Errorf("parent result text = %v, want the run to continue past a dead child", got)
	}
	_, resultEv := spawnPair(t, stdout.String())
	out, _ := resultEv["tool_output"].(string)
	if !strings.HasPrefix(out, "error:") {
		t.Errorf("tool_output = %q, want an error:-prefixed result", out)
	}
	if resultEv["is_error"] != true {
		t.Errorf("is_error = %v, want true for a failed subagent", resultEv["is_error"])
	}
	if resultEv["subagent"] != "reviewer" {
		t.Errorf("subagent = %v, want the agent named on the call", resultEv["subagent"])
	}
	// The conversation has to carry the same text the event reported, or the
	// parent's next turn would be answering something the reader never saw.
	bodies := srv.requestBodies(t)
	var replayed string
	var req struct {
		Messages []recordedMessage `json:"messages"`
	}
	if err := json.Unmarshal(bodies[len(bodies)-1], &req); err != nil {
		t.Fatalf("last POST body is not valid JSON: %v", err)
	}
	for _, msg := range req.Messages {
		if msg.Role == "tool" {
			replayed = msg.Content
		}
	}
	if replayed != out {
		t.Errorf("replayed tool message = %q, want the same error text the event carried %q", replayed, out)
	}
}

// TestSubagentChildIsNotOfferedTheSubagentTool: recursion is stopped at the
// offer, so the child must be told about bash only.
func TestSubagentChildIsNotOfferedTheSubagentTool(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-reviewer", "review things")
	srv := newScriptedChatServer(t,
		parentToolCallResponse,
		completion("child answer"),
		completion("parent answer"),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}
	sent := srv.requestBodies(t)
	if len(sent) != 3 {
		t.Fatalf("POSTs = %d, want parent, child, parent", len(sent))
	}
	if got := postedToolNames(t, sent[0]); !reflect.DeepEqual(got, []string{"bash", "subagent"}) {
		t.Errorf("parent offer = %v, want [bash subagent]", got)
	}
	if got := postedToolNames(t, sent[1]); !reflect.DeepEqual(got, []string{"bash"}) {
		t.Errorf("child offer = %v, want [bash] only — a subagent that can spawn is unbounded recursion", got)
	}
}

// TestSubagentUnknownAgentIsAnOrdinaryResult: a name off the roster is an answer
// for the Model, never a crash and never a spawn.
func TestSubagentUnknownAgentIsAnOrdinaryResult(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-reviewer", "review things")
	srv := newScriptedChatServer(t,
		`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_s","type":"function","function":{"name":"subagent","arguments":"{\"agent\":\"ghost\",\"task\":\"do the impossible\"}"}}]}}]}`,
		completion("parent answer"),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}
	sent := srv.requestBodies(t)
	if len(sent) != 2 {
		t.Fatalf("POSTs = %d, want the parent's two turns only — no spawn may be attempted", len(sent))
	}
	for i, raw := range sent {
		if got := postedSpawn(t, raw).Model; got != "m-parent" {
			t.Errorf("POST %d sent model %q, want only the parent's model", i, got)
		}
	}
	if relayed := relayedEvents(t, stdout.String()); len(relayed) != 0 {
		t.Errorf("stdout carries relayed child events for an unknown agent: %v", relayed)
	}
	_, resultEv := spawnPair(t, stdout.String())
	out, _ := resultEv["tool_output"].(string)
	if !strings.HasPrefix(out, "error:") || !strings.Contains(out, "ghost") {
		t.Errorf("tool_output = %q, want an error:-prefixed result naming the unknown agent", out)
	}
	if resultEv["is_error"] != true {
		t.Errorf("is_error = %v, want true", resultEv["is_error"])
	}
	if got := parentResult(t, stdout.String())["text"]; got != "parent answer" {
		t.Errorf("parent result text = %v, want the run to continue", got)
	}
}

// TestSubagentTaskStartingWithADashStaysPositional pins the "--" in the child's
// argv: without it reorderFlagsFirst would hand a dash-leading task to the last
// flag as its value and the child would run the wrong prompt.
func TestSubagentTaskStartingWithADashStaysPositional(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-reviewer", "review things")
	call := `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_s","type":"function","function":{"name":"subagent","arguments":"{\"agent\":\"reviewer\",\"task\":\"-not a flag- just a task\"}"}}]}}]}`
	srv := newScriptedChatServer(t, call, completion("child answer"), completion("parent answer"))
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}
	sent := srv.requestBodies(t)
	if len(sent) != 3 {
		t.Fatalf("POSTs = %d, want parent, child, parent: stdout = %q stderr = %q", len(sent), stdout.String(), stderr.String())
	}
	child := postedSpawn(t, sent[1])
	if len(child.Messages) != 2 || child.Messages[1].Content != "-not a flag- just a task" {
		t.Errorf("child messages = %+v, want the dash-leading task as the user prompt verbatim", child.Messages)
	}
}

// TestSubagentRelayDropsNonObjectLines covers what a real child cannot produce on
// demand: a stray non-JSONL line on the child's stdout must never reach the
// parent's stdout, which carries only events, and must never be swallowed
// silently either.
func TestSubagentRelayDropsNonObjectLines(t *testing.T) {
	var out, errOut bytes.Buffer
	stream := "not an event at all\nnull\n" +
		`{"kind":"result","timestamp":1234567890123,"text":"child answer","num_turns":1}` + "\n"

	text, sawResult := relaySubagentOutput(strings.NewReader(stream), "reviewer", eventWriter{w: &out}, &errOut)
	if !sawResult || text != "child answer" {
		t.Errorf("relay returned text %q sawResult %v, want the result event's text", text, sawResult)
	}
	events := stdoutEvents(t, out.String())
	if len(events) != 1 {
		t.Fatalf("parent stdout got %d lines, want only the one JSON object: %q", len(events), out.String())
	}
	if events[0]["kind"] != "result" || events[0]["in_subagent"] != true || events[0]["role"] != "reviewer" {
		t.Errorf("relayed line = %v, want the child's result stamped with role and in_subagent", events[0])
	}
	// A map re-encode keeps the child's own timestamp, and it must survive as the
	// integer the child wrote rather than a float64 rendering of it.
	if !strings.Contains(out.String(), `"timestamp":1234567890123`) {
		t.Errorf("relayed line lost or rewrote the child timestamp: %q", out.String())
	}
	dropped := errOut.String()
	if !strings.Contains(dropped, "not an event at all") {
		t.Errorf("stderr does not name the discarded line: %q", dropped)
	}
}
