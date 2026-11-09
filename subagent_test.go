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

// TestSubagentSpawnServesTwoCallsInOneTurnSequentially pins the documented claim
// that spawns run one at a time: one assistant message asking for two subagents is
// served them in call order. The ordering is read off the recorded request bodies
// because the model id goes out verbatim and so names which child sent each
// request, while usage and result events look alike on both children. The roster
// file names sort opposite to the call order, so walking the roster cannot produce
// the recorded sequence; and the first child is scripted to take two Model calls,
// so a pair that ran side by side could not hide behind one request each.
func TestSubagentSpawnServesTwoCallsInOneTurnSequentially(t *testing.T) {
	agentsDir := t.TempDir()
	writeAgentFile(t, agentsDir, "a-planner.md", "---\nname: planner\nmodel: m-planner\n---\nplan things\n")
	writeAgentFile(t, agentsDir, "z-reviewer.md", "---\nname: reviewer\nmodel: m-reviewer\n---\nreview things\n")
	cfgPath := writeDriverConfig(t, `{"agents_dir":"`+agentsDir+`"}`)

	spawnCall := func(id, agent, task string) string {
		return `{"id":"` + id + `","type":"function","function":{"name":"subagent","arguments":"{\"agent\":\"` + agent + `\",\"task\":\"` + task + `\"}"}}`
	}
	childShellTurn := `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_shell","type":"function","function":{"name":"bash","arguments":"{\"command\":\"printf reviewer-step\"}"}}]}}]}`

	srv := newScriptedChatServer(t,
		`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[`+
			spawnCall("call_first", "reviewer", "review the diff")+","+
			spawnCall("call_second", "planner", "plan the release")+`]}}]}`,
		childShellTurn,
		completion("first child answer"),
		completion("second child answer"),
		completion("parent answer"),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}
	raw := stdout.String()

	bodies := srv.requestBodies(t)
	var order []string
	for _, body := range bodies {
		order = append(order, postedSpawn(t, body).Model)
	}
	if want := []string{"m-parent", "m-reviewer", "m-reviewer", "m-planner", "m-parent"}; !reflect.DeepEqual(order, want) {
		t.Fatalf("POST order = %v, want the parent, then every request of the first child, then every request of the second, then the parent's final turn; stdout = %q", order, raw)
	}
	for i, body := range bodies {
		req := postedSpawn(t, body)
		entry, named := map[string]string{"m-reviewer": "review things", "m-planner": "plan things"}[req.Model]
		if !named {
			continue
		}
		if got := req.Messages[0]; got.Role != "system" || got.Content != entry {
			t.Errorf("POST %d sent model %q opens with %+v, want that entry's Agent File body", i, req.Model, got)
		}
	}

	var seq []string
	callAgent := map[string]string{}
	type answered struct{ agent, output string }
	answers := map[string]answered{}
	for _, ev := range stdoutEvents(t, raw) {
		if ev["in_subagent"] == true || ev["tool_name"] != "subagent" {
			continue
		}
		id, _ := ev["tool_id"].(string)
		switch ev["kind"] {
		case "tool_call":
			seq = append(seq, "call "+id)
			if input, ok := ev["tool_input"].(map[string]any); ok {
				callAgent[id], _ = input["agent"].(string)
			}
		case "tool_result":
			seq = append(seq, "result "+id)
			agent, _ := ev["subagent"].(string)
			output, _ := ev["tool_output"].(string)
			answers[id] = answered{agent: agent, output: output}
		}
	}
	if want := []string{"call call_first", "result call_first", "call call_second", "result call_second"}; !reflect.DeepEqual(seq, want) {
		t.Errorf("parent subagent events = %v, want both pairs in the order the calls were asked for; stdout = %q", seq, raw)
	}
	if want := map[string]string{"call_first": "reviewer", "call_second": "planner"}; !reflect.DeepEqual(callAgent, want) {
		t.Errorf("subagent tool_call agents = %v, want the roster entry each call named: %v", callAgent, want)
	}
	for _, want := range []struct{ id, agent, output string }{
		{"call_first", "reviewer", "first child answer"},
		{"call_second", "planner", "second child answer"},
	} {
		if got := answers[want.id]; got.agent != want.agent || got.output != want.output {
			t.Errorf("subagent tool_result %s = %+v, want %q", want.id, got, want.agent+"/"+want.output)
		}
	}

	lastReviewer, firstPlanner := -1, -1
	for i, ev := range relayedEvents(t, raw) {
		switch ev["role"] {
		case "reviewer":
			lastReviewer = i
		case "planner":
			if firstPlanner < 0 {
				firstPlanner = i
			}
		}
	}
	if lastReviewer < 0 || firstPlanner < 0 || lastReviewer > firstPlanner {
		t.Errorf("relayed child lines interleave (last reviewer line %d, first planner line %d) in %q", lastReviewer, firstPlanner, raw)
	}

	if got := parentResult(t, raw)["text"]; got != "parent answer" {
		t.Errorf("parent result text = %v, want the parent's own final answer after both children", got)
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

// TestSubagentChildIsNotOfferedTheSubagentTool: the child's Agent File grants no
// Tools, so its offer is the bash floor — the grant is what limits it, and the
// Subagent Tool is nothing it was given, which is where recursion stops.
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
	if got := postedToolNames(t, sent[0]); !reflect.DeepEqual(got, []string{"bash", "edit", "write", "subagent"}) {
		t.Errorf("parent offer = %v, want [bash edit write subagent]", got)
	}
	if got := postedToolNames(t, sent[1]); !reflect.DeepEqual(got, []string{"bash"}) {
		t.Errorf("child offer = %v, want [bash] — an entry granting nothing leaves only the floor", got)
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

// spawnAgentCall scripts one turn that asks for one subagent by name. The
// scripted server answers whatever the Model asked for, so a name the caller may
// not spawn still reaches the runtime check the offer cannot express.
func spawnAgentCall(agent string) string {
	return `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_s","type":"function","function":{"name":"subagent","arguments":"{\"agent\":\"` + agent + `\",\"task\":\"do the thing\"}"}}]}}]}`
}

// TestSubagentChildSpawnsOnlyItsOwn: an entry's spawns bounds what its run may
// hand work to at the moment of the call, not merely what the offer lists — so a
// name the entry does not name is refused as an ordinary Tool Result and starts
// no process.
func TestSubagentChildSpawnsOnlyItsOwn(t *testing.T) {
	cfgPath := agentFixtureEntries(t,
		agentSpec{name: "planner", model: "m-planner", tools: "bash, subagent", spawns: "worker"},
		agentSpec{name: "scout", model: "m-scout"},
		agentSpec{name: "reviewer", model: "m-reviewer"},
		agentSpec{name: "worker", model: "m-worker"},
	)
	callerArgs := func() []string {
		return []string{"p", "--agent", "planner"}
	}

	t.Run("spawns names the child it may run", func(t *testing.T) {
		srv := newScriptedChatServer(t, spawnAgentCall("worker"), completion("child answer"), completion("parent answer"))
		subagentEnv(t, cfgPath, srv.srv.URL)

		var stdout, stderr bytes.Buffer
		args := append(callerArgs(), "--base-url", srv.srv.URL)
		code := run(args, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
		if code != 0 {
			t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
		}
		raw := stdout.String()
		if sent := srv.requestBodies(t); len(sent) != 3 {
			t.Fatalf("POSTs = %d, want the caller's turn, the child's turn, the caller's final turn: stdout = %q", len(sent), raw)
		}
		relayed := relayedEvents(t, raw)
		if len(relayed) == 0 {
			t.Fatalf("stdout carries no relayed child events: %q", raw)
		}
		for _, ev := range relayed {
			if ev["role"] != "worker" {
				t.Errorf("relayed event role = %v, want the child's entry name", ev["role"])
			}
		}
		_, resultEv := spawnPair(t, raw)
		if out, _ := resultEv["tool_output"].(string); out != "child answer" {
			t.Errorf("tool_output = %q, want the child's answer", out)
		}
		if resultEv["is_error"] == true {
			t.Errorf("is_error = true, want a permitted spawn reported as a success: %v", resultEv)
		}
		if got := parentResult(t, raw)["text"]; got != "parent answer" {
			t.Errorf("parent result text = %v, want the run to continue", got)
		}
	})

	t.Run("a name spawns does not name is refused without spawning", func(t *testing.T) {
		srv := newScriptedChatServer(t, spawnAgentCall("scout"), completion("parent answer"))
		subagentEnv(t, cfgPath, srv.srv.URL)

		var stdout, stderr bytes.Buffer
		args := append(callerArgs(), "--base-url", srv.srv.URL)
		code := run(args, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
		if code != 0 {
			t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
		}
		raw := stdout.String()
		sent := srv.requestBodies(t)
		if len(sent) != 2 {
			t.Fatalf("POSTs = %d, want the caller's two turns only — no spawn may be attempted: stdout = %q", len(sent), raw)
		}
		for i, body := range sent {
			if got := postedSpawn(t, body).Model; got != "m-planner" {
				t.Errorf("POST %d sent model %q, want only the caller's model", i, got)
			}
		}
		if relayed := relayedEvents(t, raw); len(relayed) != 0 {
			t.Errorf("stdout carries relayed child events for a spawn the caller may not make: %v", relayed)
		}
		callEv, resultEv := spawnPair(t, raw)
		out, _ := resultEv["tool_output"].(string)
		if !strings.HasPrefix(out, "error:") || resultEv["is_error"] != true {
			t.Errorf("tool_output = %q is_error = %v, want an error:-prefixed result flagged as one", out, resultEv["is_error"])
		}
		if !strings.Contains(out, `cannot spawn agent "scout"`) {
			t.Errorf("tool_output = %q, want it to name the agent the call asked for", out)
		}
		if !strings.Contains(out, "worker") {
			t.Errorf("tool_output = %q, want it to name the one agent this run may spawn", out)
		}
		for _, other := range []string{"planner", "reviewer"} {
			if strings.Contains(out, other) {
				t.Errorf("refusal %q names %q, which this run may not spawn", out, other)
			}
		}
		// A refusal implies no relay, so neither half of the pairing fields is set.
		if callEv["tool_target"] != nil || resultEv["subagent"] != nil || resultEv["tool_target"] != nil {
			t.Errorf("refusal events carry subagent = %v tool_target = %v, want both unset", resultEv["subagent"], resultEv["tool_target"])
		}
		if got := parentResult(t, raw)["text"]; got != "parent answer" {
			t.Errorf("parent result text = %v, want the run to go on after the refusal", got)
		}
	})
}

// TestSubagentSpawnOfAgentWithNoFileIsRefused: spawns is a grant over the Agent
// Files that exist, so an entry whose spawns names only a missing file may spawn
// nobody — and the refusal says so rather than naming an empty list.
func TestSubagentSpawnOfAgentWithNoFileIsRefused(t *testing.T) {
	cfgPath := agentFixtureEntries(t,
		agentSpec{name: "planner", model: "m-planner", tools: "bash, subagent", spawns: "helper"},
		agentSpec{name: "worker", model: "m-worker"},
	)
	srv := newScriptedChatServer(t, spawnAgentCall("helper"), completion("parent answer"))
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--agent", "planner", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}
	raw := stdout.String()
	if sent := srv.requestBodies(t); len(sent) != 2 {
		t.Fatalf("POSTs = %d, want the caller's two turns only — no spawn may be attempted: stdout = %q", len(sent), raw)
	}
	_, resultEv := spawnPair(t, raw)
	out, _ := resultEv["tool_output"].(string)
	if !strings.HasPrefix(out, "error:") || resultEv["is_error"] != true {
		t.Errorf("tool_output = %q is_error = %v, want an error:-prefixed result flagged as one", out, resultEv["is_error"])
	}
	if !strings.Contains(out, "this run may spawn no agents") {
		t.Errorf("tool_output = %q, want the refusal to say this run may spawn nobody", out)
	}
	if strings.Contains(out, "worker") {
		t.Errorf("tool_output = %q names worker, which planner's spawns does not grant", out)
	}
	if got := parentResult(t, raw)["text"]; got != "parent answer" {
		t.Errorf("parent result text = %v, want the run to go on after the refusal", got)
	}
}

// TestSubagentTopLevelIsUnrestricted: no Agent File selects the top-level
// process, so no spawns field bounds it — it may name an entry another entry
// cannot.
func TestSubagentTopLevelIsUnrestricted(t *testing.T) {
	cfgPath := agentFixtureEntries(t,
		agentSpec{name: "planner", model: "m-planner", tools: "bash, subagent", spawns: "worker"},
		agentSpec{name: "worker", model: "m-worker"},
	)
	srv := newScriptedChatServer(t, spawnAgentCall("worker"), completion("child answer"), completion("parent answer"))
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}
	raw := stdout.String()
	if sent := srv.requestBodies(t); len(sent) != 3 {
		t.Fatalf("POSTs = %d, want parent, child, parent: stdout = %q", len(sent), raw)
	}
	callEv, resultEv := spawnPair(t, raw)
	if resultEv["subagent"] != "worker" || callEv["tool_target"] != "worker" {
		t.Errorf("naming fields = %v / %v, want the entry the top-level run named", callEv["tool_target"], resultEv["subagent"])
	}
	if out, _ := resultEv["tool_output"].(string); out != "child answer" {
		t.Errorf("tool_output = %q, want the child's answer", out)
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

// bashCallTurn scripts one turn that asks for exactly one bash command, with the
// arguments escaped by hand as parentToolCallResponse does: a JSON object nested
// inside a JSON string. The id is a parameter so a script can repeat the very
// same call without repeating its event-pair handle.
func bashCallTurn(id, command string) string {
	return `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"` + id +
		`","type":"function","function":{"name":"bash","arguments":"{\"command\":\"` + command + `\"}"}}]}}]}`
}

// repeatedCallBreakerText is the third-identical-call refusal main.go's
// repeatedCallRefusal synthesises — written out rather than called so a child
// inventing its own wording, rather than the inherited text, is what fails.
const repeatedCallBreakerText = "error: bash was already called 2 times in a row with these exact arguments, so it was not run again; take a different action"

// TestSubagentChildInheritsRepeatedCallBreaker: --agent selects the same kind of
// run a spawn starts, and the breaker is compiled into run() rather than handed
// down — so a child refuses its own third identical call with the very text the
// top-level golden stream pins, as a synthetic Tool Result the run answers past.
func TestSubagentChildInheritsRepeatedCallBreaker(t *testing.T) {
	cfgPath := agentFixtureEntries(t, agentSpec{name: "worker", model: "m-worker", tools: "bash"})
	srv := newScriptedChatServer(t,
		bashCallTurn("call_1", "echo one"),
		bashCallTurn("call_2", "echo one"),
		bashCallTurn("call_3", "echo one"),
		completion("child answer"),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"--agent", "worker", "p", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 — a refusal is a Tool Result the run answers past, not a failed run; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}
	raw := stdout.String()
	if sent := srv.requestBodies(t); len(sent) != 4 {
		t.Fatalf("POSTs = %d, want three call turns and the answer: the third call must not be sent again, but the refusal is only a Tool Result: stdout = %q", len(sent), raw)
	}

	var results, refused int
	for _, ev := range stdoutEvents(t, raw) {
		if ev["kind"] != "tool_result" {
			continue
		}
		results++
		// A run started with --agent is the top-level process here, so every line it
		// writes is its own: in_subagent would mean the refusal came from somewhere else.
		if v, present := ev["in_subagent"]; present && v == true {
			t.Errorf("tool_result carries in_subagent = %v, want the refusal among this run's own events: %v", v, ev)
		}
		out, _ := ev["tool_output"].(string)
		if !strings.Contains(out, "bash was already called 2 times in a row") {
			continue
		}
		refused++
		if ev["tool_name"] != "bash" || ev["tool_target"] != "echo one" {
			t.Errorf("refusing tool_result names %v %v, want the refused bash call and its command: %v", ev["tool_name"], ev["tool_target"], ev)
		}
		if ev["is_error"] != true {
			t.Errorf("refusing tool_result is_error = %v, want true: %v", ev["is_error"], ev)
		}
		if out != repeatedCallBreakerText {
			t.Errorf("refusing tool_output = %q, want the Driver's own synthetic refusal %q", out, repeatedCallBreakerText)
		}
	}
	if refused != 1 {
		t.Fatalf("%d tool_result event(s) carry the breaker refusal, want the third call refused once: %q", refused, raw)
	}
	if results != 3 {
		t.Errorf("%d tool_result events, want one per scripted call: %q", results, raw)
	}
	if got := resultText(t, raw); got != "child answer" {
		t.Errorf("result text = %q, want the child's answer past its own refusal", got)
	}
}

// TestSubagentChildRepeatedCallCounterIsItsOwn: by the time the parent makes its
// next identical call, the child has driven the "echo same" streak to a refusal,
// so one shared counter would refuse the parent here — only a counter of its own
// lets that call run. The intervening subagent call resets the parent's own
// streak, so what is pinned is that the child's counts never reach the parent.
func TestSubagentChildRepeatedCallCounterIsItsOwn(t *testing.T) {
	cfgPath := agentFixtureEntries(t, agentSpec{name: "worker", model: "m-worker", tools: "bash"})
	srv := newScriptedChatServer(t,
		bashCallTurn("call_p1", "echo same"),
		spawnAgentCall("worker"),
		bashCallTurn("call_c1", "echo same"),
		bashCallTurn("call_c2", "echo same"),
		bashCallTurn("call_c3", "echo same"),
		completion("child answer"),
		bashCallTurn("call_p2", "echo same"),
		completion("parent answer"),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}
	raw := stdout.String()
	if sent := srv.requestBodies(t); len(sent) != 8 {
		t.Fatalf("POSTs = %d, want the script of 8 in full (the refusal must not stop either run short): stdout = %q", len(sent), raw)
	}

	// The child did refuse its third identical call, and it did so inside itself —
	// relayed, so it belongs to the child's stream.
	relayed := relayedToolResults(t, raw)
	if n := len(relayed); n != 3 {
		t.Fatalf("%d relayed tool_result events, want the child's three scripted calls: %q", n, raw)
	}
	refused, ok := relayed["call_c3"]
	if !ok {
		t.Fatalf("relayed stream carries no tool_result for the child's third call: %q", raw)
	}
	if out, _ := refused["tool_output"].(string); out != repeatedCallBreakerText {
		t.Errorf("child's third call tool_output = %q, want the refusal %q", out, repeatedCallBreakerText)
	}
	if refused["is_error"] != true {
		t.Errorf("child's third call is_error = %v, want true: %v", refused["is_error"], refused)
	}

	// And the parent's identical call after the spawn ran anyway.
	var parentResults int
	for _, ev := range stdoutEvents(t, raw) {
		if ev["kind"] != "tool_result" || ev["in_subagent"] == true || ev["tool_name"] != "bash" {
			continue
		}
		parentResults++
		out, _ := ev["tool_output"].(string)
		if strings.Contains(out, "was already called") {
			t.Errorf("parent-side bash tool_output = %q, want no breaker refusal: the child's streak is not the parent's: %v", out, ev)
		}
		if v, present := ev["is_error"]; present {
			t.Errorf("parent-side bash tool_result is_error = %v, want the command to have run: %v", v, ev)
		}
		if !strings.Contains(out, "same\nexit code: 0") {
			t.Errorf("parent-side bash tool_output = %q, want the command's own output", out)
		}
	}
	if parentResults != 2 {
		t.Errorf("%d parent-side bash tool_result events, want the two identical calls the parent scripted: %q", parentResults, raw)
	}
	if got := parentResult(t, raw)["text"]; got != "parent answer" {
		t.Errorf("parent result text = %v, want the parent's own final answer", got)
	}
}

// TestSubagentChildEventClockIsPinnableFromEnv is the child half of a golden
// Pass log: the relay keeps the child's own timestamp, so the parent's stdout can
// only compare byte for byte if the child's clock is pinnable across the process
// boundary. The environment is the sole channel to a spawned child (see
// subagentEnv), so the pin rides there.
func TestSubagentChildEventClockIsPinnableFromEnv(t *testing.T) {
	const (
		pinnedStart = 1700000000000
		pinnedStep  = 1000
	)
	cfgPath := agentFixture(t, "reviewer", "m-reviewer", "review things")
	srv := newScriptedChatServer(t,
		parentToolCallResponse,
		completion("child answer"),
		completion("parent answer"),
	)
	subagentEnv(t, cfgPath, srv.srv.URL)
	t.Setenv(envTestNowMS, fmt.Sprintf("%d:%d", pinnedStart, pinnedStep))

	var stdout, stderr bytes.Buffer
	code := run([]string{"p", "--model", "m-parent", "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", code, stdout.String(), stderr.String())
	}
	raw := stdout.String()

	relayed := relayedEvents(t, raw)
	if len(relayed) == 0 {
		t.Fatalf("stdout carries no relayed child lines: %q", raw)
	}

	// run() reads the clock once for its own run-start bookkeeping before the
	// first event line goes out, so the child's start line carries the second tick
	// of the pinned clock — a wall-clock millisecond is what this pins out.
	starts := 0
	for _, ev := range relayed {
		if ev["kind"] != "start" {
			continue
		}
		starts++
		if ts, _ := ev["timestamp"].(float64); int64(ts) != pinnedStart+pinnedStep {
			t.Errorf("relayed child start timestamp = %v, want the pinned tick %d: %q", ev["timestamp"], pinnedStart+pinnedStep, raw)
		}
	}
	if starts != 1 {
		t.Fatalf("%d relayed child start lines, want exactly one: %q", starts, raw)
	}

	// Every relayed line has to sit on the pinned grid, not just the first: one
	// wall-clock stamp anywhere in the child's stream makes the whole block
	// uncomparable, which is the property a golden log needs rather than any
	// particular value.
	for i, ev := range relayed {
		ts, ok := ev["timestamp"].(float64)
		if !ok {
			t.Fatalf("relayed child line %d (%v) carries no numeric timestamp: %v", i, ev["kind"], ev)
		}
		if v := int64(ts); v < pinnedStart || (v-pinnedStart)%pinnedStep != 0 {
			t.Errorf("relayed child line %d (%v) timestamp = %d, want a tick of the pinned clock %d+%dn: %q", i, ev["kind"], v, pinnedStart, pinnedStep, raw)
		}
	}

	// The durations the child derives from that same clock are pinned with it: a
	// golden log that stamps identically every run but compares a drifting
	// duration_ms is no more reproducible than one that does not stamp at all.
	for _, ev := range relayed {
		if ev["kind"] != "result" {
			continue
		}
		d, ok := ev["duration_ms"].(float64)
		if !ok {
			t.Fatalf("relayed child result carries no numeric duration_ms: %v", ev)
		}
		if v := int64(d); v <= 0 || v%pinnedStep != 0 {
			t.Errorf("relayed child result duration_ms = %d, want a positive multiple of the pinned step %d: %q", v, pinnedStep, raw)
		}
	}
	if got := relayed[len(relayed)-1]["kind"]; got != "result" {
		t.Errorf("last relayed child line kind = %v, want the child's own result: %q", got, raw)
	}
}
