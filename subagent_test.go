package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
	// passes counts how many of failModel's POSTs are still answered from the
	// script before the refusals start, so a test can hand a child one Model call
	// that succeeds — and one that fails afterwards.
	passes   int
	replies  []string
	answered int
	bodies   [][]byte
	srv      *httptest.Server
}

func newSubagentReplyServer(t *testing.T, failModel string, replies ...string) *subagentReplyServer {
	return newSubagentReplyServerPassing(t, failModel, 0, replies...)
}

// newSubagentReplyServerPassing refuses failModel only after answering its first
// passes POSTs from the script.
func newSubagentReplyServerPassing(t *testing.T, failModel string, passes int, replies ...string) *subagentReplyServer {
	s := &subagentReplyServer{failModel: failModel, passes: passes, replies: replies}
	s.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		s.bodies = append(s.bodies, body)
		var req struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(body, &req)
		fail := req.Model == failModel && s.passes <= 0
		if !fail {
			if s.answered < len(s.replies) {
				if req.Model == failModel {
					s.passes--
				}
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

	text, sawResult := relaySubagentOutput(strings.NewReader(stream), "reviewer", relaySession(&out, &errOut))
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

// TestSubagentChildTurnDoesNotBumpParentNumTurns keeps num_turns what it has
// always been: this process's own Model calls, however many a child made.
func TestSubagentChildTurnDoesNotBumpParentNumTurns(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-reviewer", "review things")
	srv := newScriptedChatServer(t,
		withUsage(parentToolCallResponse, "m-parent", 100, 20),
		usageCompletion("m-reviewer", "child answer", 7, 1),
		usageCompletion("m-parent", "parent answer", 30, 5),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}
	// The child ran a whole turn of its own inside the pair; the parent still
	// counts only its two Model calls.
	if got, _ := parentResult(t, stdout.String())["num_turns"].(float64); int(got) != 2 {
		t.Errorf("parent result num_turns = %v, want the parent's own two Model calls", got)
	}
}

// TestSubagentRollUpAddsNoUsageEventOfTheParents: the child's spend reaches the
// result through the tally, never through a usage event the parent invents — a
// reader that sums the usage events it sees must still get one per parent call.
func TestSubagentRollUpAddsNoUsageEventOfTheParents(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-reviewer", "review things")
	srv := newScriptedChatServer(t,
		withUsage(parentToolCallResponse, "m-parent", 100, 20),
		usageCompletion("m-reviewer", "child answer", 7, 1),
		usageCompletion("m-parent", "parent answer", 30, 5),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}

	own, relayedUsage := 0, 0
	for _, ev := range stdoutEvents(t, stdout.String()) {
		if ev["kind"] != "usage" {
			continue
		}
		if ev["in_subagent"] == true {
			relayedUsage++
			// The relayed line stays attributed to the child even though its spend is
			// now counted: attribution and accounting are separate facts.
			if ev["role"] != "reviewer" {
				t.Errorf("relayed usage role = %v, want the agent name", ev["role"])
			}
			continue
		}
		own++
		if ev["model"] != "m-parent" {
			t.Errorf("parent usage model = %v, want the parent's own model", ev["model"])
		}
	}
	if own != 2 {
		t.Errorf("stdout carries %d non-relayed usage events, want exactly one per parent Model call: %q", own, stdout.String())
	}
	if relayedUsage != 1 {
		t.Errorf("stdout carries %d relayed usage events, want the child's one: %q", relayedUsage, stdout.String())
	}
}

// childBashTurnResponse is a child turn that keeps the child running: one bash
// Tool Call, so its next Model call is the one the fake refuses.
const childBashTurnResponse = `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_b","type":"function","function":{"name":"bash","arguments":"{\"command\":\"true\"}"}}]}}]}`

// TestSubagentFailingChildStillContributesItsSpend is why the roll-up folds usage
// events rather than the child's result totals: a child that dies emits no result
// event at all, and the tokens it already spent are still real.
func TestSubagentFailingChildStillContributesItsSpend(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-reviewer", "review things")
	srv := newSubagentReplyServerPassing(t, "m-reviewer", 1,
		withUsage(parentToolCallResponse, "m-parent", 100, 20),
		withUsage(childBashTurnResponse, "m-reviewer", 7, 1),
		usageCompletion("m-parent", "parent answer", 30, 5),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 — a dead child is still a Tool Result; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}

	result := parentResult(t, stdout.String())
	got := resultTokenFields(t, result)
	if got["input"] != 137 || got["output"] != 26 {
		t.Errorf("parent result tokens = %v, want parent 130/25 plus the dead child's 7/1", got)
	}
	// The premise of the assertion above: the child got one usage event relayed,
	// and died before it could report a result of its own.
	var relayedResult, relayedUsage int
	for _, ev := range relayedEvents(t, stdout.String()) {
		switch ev["kind"] {
		case "result":
			relayedResult++
		case "usage":
			relayedUsage++
		}
	}
	if relayedUsage != 1 || relayedResult != 0 {
		t.Errorf("relayed stream has %d usage and %d result events, want the child's one usage and no result: %q", relayedUsage, relayedResult, stdout.String())
	}
	_, resultEv := spawnPair(t, stdout.String())
	out, _ := resultEv["tool_output"].(string)
	if !strings.HasPrefix(out, "error:") {
		t.Errorf("tool_output = %q, want an error:-prefixed result", out)
	}
	if got, _ := result["num_turns"].(float64); int(got) != 2 {
		t.Errorf("parent result num_turns = %v, want the parent's own two Model calls", got)
	}
}

// TestSubagentRelayIgnoresUnusableUsageTokens: one child usage line with no
// tokens, or tokens of the wrong shape, contributes nothing and must not corrupt
// the parent's totals — and it still gets relayed, because relaying and
// accounting are separate jobs.
func TestSubagentRelayIgnoresUnusableUsageTokens(t *testing.T) {
	var out, errOut bytes.Buffer
	var spent tokenTotals
	s := session{events: eventWriter{w: &out}, stderr: &errOut, delegated: &spent, models: map[string]struct{}{}}
	stream := strings.Join([]string{
		`{"kind":"usage","timestamp":1,"model":"m-reviewer"}`,
		`{"kind":"usage","timestamp":2,"tokens":"not an object"}`,
		`{"kind":"usage","timestamp":3,"tokens":{"input":"seven"}}`,
		`{"kind":"usage","timestamp":4,"tokens":null}`,
	}, "\n") + "\n"

	if _, sawResult := relaySubagentOutput(strings.NewReader(stream), "reviewer", s); sawResult {
		t.Errorf("sawResult = true, want false: the stream carries no result event")
	}
	if spent != (tokenTotals{}) {
		t.Errorf("delegated tally = %+v, want unusable usage lines to contribute nothing", spent)
	}
	// A model id is a separate fact from the tokens, so the one line that carried
	// it still counts.
	if _, ok := s.models["m-reviewer"]; !ok {
		t.Errorf("model set = %v, want the usage event's model id recorded", s.models)
	}
	if events := stdoutEvents(t, out.String()); len(events) != 4 {
		t.Errorf("parent stdout got %d lines, want all four usage events relayed: %q", len(events), out.String())
	}
}

// TestSubagentRelayResultWithoutText: a result event is how the child hands over
// its answer, so one carrying no usable text is a failure rather than an empty
// success — the error: prefix is the only failure signal the Tool Result can
// carry, and it still gets relayed because relaying and answering are separate.
func TestSubagentRelayResultWithoutText(t *testing.T) {
	for _, stream := range []string{
		`{"kind":"result","timestamp":1,"num_turns":1}` + "\n",
		`{"kind":"result","timestamp":2,"text":42,"num_turns":1}` + "\n",
	} {
		var out, errOut bytes.Buffer
		text, sawResult := relaySubagentOutput(strings.NewReader(stream), "reviewer", relaySession(&out, &errOut))
		if !sawResult {
			t.Errorf("sawResult = false, want true: the child did finish with a result event (%s)", stream)
		}
		if !strings.HasPrefix(text, "error:") {
			t.Errorf("relay returned text %q, want an error:-prefixed message (%s)", text, stream)
		}
		if events := stdoutEvents(t, out.String()); len(events) != 1 || events[0]["kind"] != "result" {
			t.Errorf("parent stdout = %q, want the child's result event relayed", out.String())
		}
	}
}

// TestSubagentRelayDrainsAfterOverLongLine: the scan stops at a line larger than
// sseMaxLineBytes, but the child is still writing into a pipe nobody reads and
// there is deliberately no child-side timeout, so the relay has to drain what is
// left or runSubagentTool's cmd.Wait can block forever. The lines after the break
// are the accepted loss: the run ends as "emitted no result event".
func TestSubagentRelayDrainsAfterOverLongLine(t *testing.T) {
	chunk := strings.Repeat("x", 64*1024)
	// Assembled from repeats rather than one oversized literal, so the test stays
	// quick and never holds the whole over-long line in memory at once.
	var parts []io.Reader
	for i := 0; i < sseMaxLineBytes/len(chunk)+16; i++ {
		parts = append(parts, strings.NewReader(chunk))
	}
	stream := io.MultiReader(append(parts, strings.NewReader("\n"+`{"kind":"result","timestamp":1,"text":"late answer"}`+"\n"))...)

	var out, errOut bytes.Buffer
	text, sawResult := relaySubagentOutput(stream, "reviewer", relaySession(&out, &errOut))
	if sawResult || text != "" {
		t.Errorf("relay returned text %q sawResult %v, want the result after the broken line to be lost", text, sawResult)
	}
	if !strings.Contains(errOut.String(), fmt.Sprint(sseMaxLineBytes)) {
		t.Errorf("stderr does not name the byte bound: %q", errOut.String())
	}
	if out.Len() != 0 || strings.Contains(out.String(), "xxxx") {
		t.Errorf("over-long line reached the parent's stdout: %q", boundBody(out.Bytes()))
	}
	// The reader sitting at EOF is the whole point: a real pipe left half-read is a
	// child blocked in Write.
	if _, err := stream.Read(make([]byte, 1)); err != io.EOF {
		t.Errorf("read after the relay returned %v, want io.EOF: the relay left output undrained", err)
	}
}

// relaySession builds the session a test needs when it calls the relay directly
// rather than through a whole run.
func relaySession(out, errOut io.Writer) session {
	return session{events: eventWriter{w: out}, stderr: errOut, delegated: &tokenTotals{}, models: map[string]struct{}{}}
}

// withUsage attaches a model id and one Model call's usage to a scripted
// completion that carries neither: a roll-up test can only tell the parent's
// tokens from the child's if each is stamped with its own numbers.
func withUsage(response, model string, input, output int) string {
	return strings.Replace(response, `{"choices"`, fmt.Sprintf(`{"model":%q,"usage":{"prompt_tokens":%d,"completion_tokens":%d},"choices"`, model, input, output), 1)
}

// usageCompletion scripts a final answer that answers as a given model id and
// spends given tokens.
func usageCompletion(model, text string, input, output int) string {
	return withUsage(completion(text), model, input, output)
}

// resultTokenFields reads a result event's token object as numbers. A value that
// is not a number fails the test rather than reading as zero, so a rewritten
// field cannot hide as a missing one.
func resultTokenFields(t *testing.T, ev map[string]any) map[string]float64 {
	t.Helper()
	raw, ok := ev["tokens"].(map[string]any)
	if !ok {
		t.Fatalf("result event carries no token object: %#v", ev["tokens"])
	}
	out := make(map[string]float64, len(raw))
	for field, v := range raw {
		n, ok := v.(float64)
		if !ok {
			t.Fatalf("result token %s = %#v, want a JSON number", field, v)
		}
		out[field] = n
	}
	return out
}

// resultStrings reads one of a result event's string-array fields as strings.
func resultStrings(t *testing.T, ev map[string]any, field string) []string {
	t.Helper()
	raw, ok := ev[field].([]any)
	if !ok {
		t.Fatalf("result event field %s = %#v, want a JSON array", field, ev[field])
	}
	out := make([]string, 0, len(raw))
	for _, v := range raw {
		s, ok := v.(string)
		if !ok {
			t.Fatalf("result event %s entry = %#v, want a string", field, v)
		}
		out = append(out, s)
	}
	return out
}

// TestSubagentChildUsageRollsIntoParentResultTotals is the issue's accounting
// criterion: the parent's result states what the whole session cost, child
// Model calls included.
func TestSubagentChildUsageRollsIntoParentResultTotals(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-reviewer", "review things")
	srv := newScriptedChatServer(t,
		withUsage(parentToolCallResponse, "m-parent", 100, 20),
		usageCompletion("m-reviewer", "child answer", 7, 1),
		usageCompletion("m-parent", "parent answer", 30, 5),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}

	got := resultTokenFields(t, parentResult(t, stdout.String()))
	want := map[string]float64{"input": 137, "output": 26, "cache_read": 0, "cache_write_5m": 0, "cache_write_1h": 0, "cost_usd": 0}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("parent result tokens = %v, want parent 130/25 plus child 7/1 = %v", got, want)
	}
	if got["input"] == 130 && got["output"] == 25 {
		t.Errorf("parent result tokens are the parent's own %v; the child's spend is missing", got)
	}
}

// TestSubagentChildModelRollsIntoParentResultModels: a model that answered the
// session answered it, whoever delegated the call to it.
func TestSubagentChildModelRollsIntoParentResultModels(t *testing.T) {
	cfgPath := agentFixture(t, "reviewer", "m-reviewer", "review things")
	srv := newScriptedChatServer(t,
		withUsage(parentToolCallResponse, "m-parent", 100, 20),
		usageCompletion("m-reviewer", "child answer", 7, 1),
		usageCompletion("m-parent", "parent answer", 30, 5),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}

	// Sorted, because sortedModelIDs is the only order the result contract has.
	if got := resultStrings(t, parentResult(t, stdout.String()), "models"); !reflect.DeepEqual(got, []string{"m-parent", "m-reviewer"}) {
		t.Errorf("parent result models = %v, want both model ids sorted", got)
	}
}

// denyHookRosterFixture writes the three files a guarded roster needs — one Agent
// File for reviewer, a real Guard Hook script that denies only a command
// containing the word "denied" for a reason naming a credential path, and the
// Hook Manifest naming that script for Bash — and hands back the paths so each
// test composes its own Driver Config. Only the config joins them, which is what
// lets the contrast test below drop one key and nothing else.
func denyHookRosterFixture(t *testing.T, reason string) (agentsDir, hookPath, manifestPath string) {
	t.Helper()
	agentsDir = t.TempDir()
	writeAgentFile(t, agentsDir, "reviewer.md", "---\nname: reviewer\nmodel: m-reviewer\n---\nreview things\n")
	// The hook reads the protocol payload on stdin, so the match is on the command
	// text the child asked for and not on anything the shell would expand.
	hookPath = writeGuardHookScript(t, "deny-credential-path.sh",
		`input=$(cat)
	case "$input" in
	  *denied*) echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"`+reason+`"}}' ;;
	esac`)
	manifestPath = writeHookManifest(t, `{"pre":[{"path":"`+hookPath+`","tools":["Bash"]}]}`)
	return agentsDir, hookPath, manifestPath
}

// childGuardTurnResponse scripts the child's one turn as a single assistant
// message carrying two bash Tool Calls — the two-calls-in-one-message shape
// TestRunStreamReassemblesInterleavedToolCallFragments pins — so the Guard chain
// has to judge each call on its own and the refusal of one cannot spill onto the
// other. The arguments are escaped by hand for the same reason
// parentToolCallResponse is: a JSON object nested inside a JSON string.
func childGuardTurnResponse(denied, clean string) string {
	bashCall := func(id, command string) string {
		return `{"id":"` + id + `","type":"function","function":{"name":"bash","arguments":"{\"command\":\"` + command + `\"}"}}`
	}
	return `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[` +
		bashCall("call_denied", denied) + "," + bashCall("call_clean", clean) + `]}}]}`
}

// relayedToolResults indexes a relaying run's Tool Result lines by tool_id, the
// only handle a test has on which of a child's calls a line reports.
func relayedToolResults(t *testing.T, raw string) map[string]map[string]any {
	t.Helper()
	out := map[string]map[string]any{}
	for _, ev := range relayedEvents(t, raw) {
		if ev["kind"] == "tool_result" {
			out[fmt.Sprint(ev["tool_id"])] = ev
		}
	}
	return out
}

// TestSubagentChildHonoursTheConfiguredDenyHook is the criterion issue #25 was
// blocked on: the child inherits the Driver Config, and a configured deny hook
// blocks a denied command inside the child. The child resolves the config through
// the environment it inherits, so nothing here hands the manifest to the child by
// any channel of its own.
func TestSubagentChildHonoursTheConfiguredDenyHook(t *testing.T) {
	agentsDir, hookPath, manifestPath := denyHookRosterFixture(t, "reads the credential path /run/secrets/spindrift/token")
	cfgPath := writeDriverConfig(t, `{"agents_dir":"`+agentsDir+`","hook_manifest":"`+manifestPath+`"}`)

	dir := t.TempDir()
	deniedMarker := filepath.Join(dir, "denied-credential-read.marker")
	cleanMarker := filepath.Join(dir, "clean-ran.marker")
	deniedCmd := "printf nope > " + deniedMarker
	cleanCmd := "printf clean-ran | tee " + cleanMarker

	srv := newScriptedChatServer(t,
		parentToolCallResponse,
		childGuardTurnResponse(deniedCmd, cleanCmd),
		completion("child answer"),
		completion("parent answer"),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 — a refusal inside a child is a Tool Result, not a failed run; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}
	raw := stdout.String()

	// The side effects are the verdict that matters: the guard stopped the denied
	// command and only the denied command.
	if hookMarkerExists(t, deniedMarker) {
		t.Errorf("marker %s exists: the child ran the command its configured hook denies", deniedMarker)
	}
	if !hookMarkerExists(t, cleanMarker) {
		t.Errorf("marker %s does not exist: the child's clean call was blocked too", cleanMarker)
	}

	results := relayedToolResults(t, raw)
	denied, ok := results["call_denied"]
	if !ok {
		t.Fatalf("relayed stream carries no tool_result for the denied call: %q", raw)
	}
	if denied["role"] != "reviewer" {
		t.Errorf("denied call tool_result role = %v, want the agent name", denied["role"])
	}
	if denied["is_error"] != true {
		t.Errorf("denied call tool_result is_error = %v, want true: %v", denied["is_error"], denied)
	}
	// The call event still names the command, and the refusal names the script to
	// go ask and why — the same contract a refusal in the parent has.
	if denied["tool_target"] != deniedCmd {
		t.Errorf("denied call tool_target = %v, want %q", denied["tool_target"], deniedCmd)
	}
	out, _ := denied["tool_output"].(string)
	for _, want := range []string{"error: ", hookPath, "credential path /run/secrets/spindrift/token"} {
		if !strings.Contains(out, want) {
			t.Errorf("denied call tool_output = %q, want it to contain %q", out, want)
		}
	}

	clean, ok := results["call_clean"]
	if !ok {
		t.Fatalf("relayed stream carries no tool_result for the clean call: %q", raw)
	}
	if v, present := clean["is_error"]; present {
		t.Errorf("clean call tool_result carries is_error = %v, want a command that ran and succeeded", v)
	}
	cleanOut, _ := clean["tool_output"].(string)
	if !strings.Contains(cleanOut, "clean-ran") {
		t.Errorf("clean call tool_output = %q, want the command's own output", cleanOut)
	}

	// The child's run is otherwise an ordinary one: the refusal is a Tool Result it
	// answers past, its answer still reaches the parent, and the parent's result is
	// its own.
	if len(results) != 2 {
		t.Errorf("relayed stream carries %d tool_result events, want one per child call", len(results))
	}
	_, subagentResult := spawnPair(t, raw)
	if got := subagentResult["tool_output"]; got != "child answer" {
		t.Errorf("Subagent Tool Result tool_output = %v, want the child's final answer past the refusal", got)
	}
	if got := parentResult(t, raw)["text"]; got != "parent answer" {
		t.Errorf("parent result text = %v, want the parent's own final answer", got)
	}
}

// TestSubagentChildGuardComesFromTheConfiguredManifest is the config half of the
// criterion: with the same roster, the same hook script and manifest still on
// disk, and the same scripted child turn, a Driver Config that names no
// hook_manifest leaves the child unguarded. Since the parent hands the child
// nothing but argv and the environment, the key the two runs differ by is the
// only thing that can have guarded the child in the test above.
func TestSubagentChildGuardComesFromTheConfiguredManifest(t *testing.T) {
	agentsDir, hookPath, _ := denyHookRosterFixture(t, "credential path")
	cfgPath := writeDriverConfig(t, `{"agents_dir":"`+agentsDir+`"}`)

	marker := filepath.Join(t.TempDir(), "unguarded-denied-ran.marker")
	deniedCmd := "printf ran > " + marker

	srv := newScriptedChatServer(t,
		parentToolCallResponse,
		childGuardTurnResponse(deniedCmd, deniedCmd),
		completion("child answer"),
		completion("parent answer"),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}
	raw := stdout.String()

	if !hookMarkerExists(t, marker) {
		t.Errorf("marker %s does not exist: a child guarded with no hook_manifest configured is a guard that does not come from the config", marker)
	}
	for id, ev := range relayedToolResults(t, raw) {
		if v, present := ev["is_error"]; present {
			t.Errorf("%s tool_result is_error = %v, want no refusal without a manifest: %v", id, v, ev)
		}
		if out, _ := ev["tool_output"].(string); strings.Contains(out, hookPath) {
			t.Errorf("%s tool_output = %q, want no refusal by the unconfigured hook %q", id, out, hookPath)
		}
	}
}
