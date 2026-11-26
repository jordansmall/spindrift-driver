package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// subagentHelperEnv names the test binary's helper-process branch in TestMain:
// set, the binary IS the Driver, which is what lets a spawn test run a real
// child without building a second binary.
const subagentHelperEnv = "SPINDRIFT_SUBAGENT_HELPER"

// subagentEnv points a run and every child it spawns at one roster and one
// scripted Model endpoint. A child inherits the environment rather than a
// config struct, so
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

// TestSubagentSpawnRunsTheAgentFileEntry is the issue's core criterion: the child
// is a whole Driver run started from the roster entry, so the model id goes out
// verbatim and the Agent File body is what its conversation opens with.
func TestSubagentSpawnRunsTheAgentFileEntry(t *testing.T) {
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(parentToolCallResponse),
			replyCompletion(completion("parent answer")),
		},
		"m-reviewer": {replyCompletion(completion("child answer"))},
	}, "top-level prompt",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}

	if len(got.requests) != 3 {
		t.Fatalf("POSTs = %d, want parent, child, parent: %q", len(got.requests), got.stdout)
	}
	if model := postedSpawn(t, got.requests[0].body).Model; model != "m-parent" {
		t.Errorf("parent sent model %q, want the --model value", model)
	}
	child := postedSpawn(t, got.requests[1].body)
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
	// The specs are written planner-first so the index-prefixed Agent File names
	// still sort opposite to the order the two calls ask for them in.
	specs := []agentSpec{
		{name: "planner", model: "m-planner", body: "plan things"},
		{name: "reviewer", model: "m-reviewer", body: "review things"},
	}

	spawnCall := func(id, agent, task string) string {
		return `{"id":"` + id + `","type":"function","function":{"name":"subagent","arguments":"{\"agent\":\"` + agent + `\",\"task\":\"` + task + `\"}"}}`
	}
	childShellTurn := `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_shell","type":"function","function":{"name":"bash","arguments":"{\"command\":\"printf reviewer-step\"}"}}]}}]}`

	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[` +
				spawnCall("call_first", "reviewer", "review the diff") + "," +
				spawnCall("call_second", "planner", "plan the release") + `]}}]}`),
			replyCompletion(completion("parent answer")),
		},
		"m-reviewer": {replyCompletion(childShellTurn), replyCompletion(completion("first child answer"))},
		"m-planner":  {replyCompletion(completion("second child answer"))},
	}, "p",
		withRoster(specs...),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}
	raw := string(got.stdout)

	var order []string
	for _, req := range got.requests {
		order = append(order, postedSpawn(t, req.body).Model)
	}
	if want := []string{"m-parent", "m-reviewer", "m-reviewer", "m-planner", "m-parent"}; !reflect.DeepEqual(order, want) {
		t.Fatalf("POST order = %v, want the parent, then every request of the first child, then every request of the second, then the parent's final turn; stdout = %q", order, raw)
	}
	for i, req := range got.requests {
		posted := postedSpawn(t, req.body)
		entry, named := map[string]string{"m-reviewer": "review things", "m-planner": "plan things"}[posted.Model]
		if !named {
			continue
		}
		if msg := posted.Messages[0]; msg.Role != "system" || msg.Content != entry {
			t.Errorf("POST %d sent model %q opens with %+v, want that entry's Agent File body", i, posted.Model, msg)
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
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(parentToolCallResponse),
			replyCompletion(completion("parent answer")),
		},
		"m-reviewer": {replyCompletion(completion("child answer"))},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}

	events := stdoutEvents(t, string(got.stdout))
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
		t.Fatalf("no subagent tool_call/tool_result pair in %q", got.stdout)
	}

	relayed := relayedEvents(t, string(got.stdout))
	if len(relayed) == 0 {
		t.Fatalf("no relayed child events in %q", got.stdout)
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
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(parentToolCallResponse),
			replyCompletion(completion("parent answer")),
		},
		"m-reviewer": {replyCompletion(completion("child answer"))},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}

	callEv, resultEv := spawnPair(t, string(got.stdout))
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

// TestSubagentAnswerStartingLikeAFailureIsNotOne: the child's answer is the
// verdict's whole evidence, so prose that merely opens with the word "error:" is
// still a child that answered, and no is_error may ride on it.
func TestSubagentAnswerStartingLikeAFailureIsNotOne(t *testing.T) {
	answer := "error: the build is broken in the second file"
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(parentToolCallResponse),
			replyCompletion(completion("parent answer")),
		},
		"m-reviewer": {replyCompletion(completion(answer))},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}

	_, resultEv := spawnPair(t, string(got.stdout))
	if got := resultEv["tool_output"]; got != answer {
		t.Errorf("tool_result tool_output = %v, want the child's answer verbatim", got)
	}
	if v, ok := resultEv["is_error"]; ok {
		t.Errorf("tool_result carries is_error = %v for a child that answered; only the spawn's own failure may state one", v)
	}
}

// TestSubagentAnswerDoesNotBecomeParentResult keeps the two answers apart: the
// child's answer is a Tool Result, the parent's is the run's result.
func TestSubagentAnswerDoesNotBecomeParentResult(t *testing.T) {
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(parentToolCallResponse),
			replyCompletion(completion("parent answer")),
		},
		"m-reviewer": {replyCompletion(completion("child answer"))},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}
	if text := parentResult(t, string(got.stdout))["text"]; text != "parent answer" {
		t.Errorf("parent result text = %v, want the parent's own final answer", text)
	}
	var childResultText any
	for _, ev := range relayedEvents(t, string(got.stdout)) {
		if ev["kind"] == "result" {
			childResultText = ev["text"]
		}
	}
	if childResultText != "child answer" {
		t.Errorf("relayed result text = %v, want the child's own answer relayed as its own result event", childResultText)
	}
	// The child's answer reached the parent twice over: as the relayed prose and
	// as the Tool Result the next turn reads.
	if _, resultEv := spawnPair(t, string(got.stdout)); resultEv["tool_output"] != "child answer" {
		t.Errorf("tool_output = %v, want the child's answer", resultEv["tool_output"])
	}
}

// TestSubagentFailureIsAnErrorToolResult: a child that dies is an ordinary error
// shaped Tool Result, and the parent still finishes its own run.
func TestSubagentFailureIsAnErrorToolResult(t *testing.T) {
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(parentToolCallResponse),
			replyCompletion(completion("parent answer")),
		},
		// The child never gets an answer, and a 500 is retried, so its queue holds
		// one refusal per attempt: chatRetryAttempts (main.go:800) at its default
		// of 3, the only value a re-exec'd child can see. Should that default
		// change, the queue runs dry and checkNoUnscriptedRequests fails the test.
		"m-reviewer": {
			reject(http.StatusInternalServerError, nil, "scripted failure"),
			reject(http.StatusInternalServerError, nil, "scripted failure"),
			reject(http.StatusInternalServerError, nil, "scripted failure"),
		},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0 — a failed child is a Tool Result, not a failed run; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}
	if text := parentResult(t, string(got.stdout))["text"]; text != "parent answer" {
		t.Errorf("parent result text = %v, want the run to continue past a dead child", text)
	}
	_, resultEv := spawnPair(t, string(got.stdout))
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
	var replayed string
	var req struct {
		Messages []recordedMessage `json:"messages"`
	}
	last := got.requests[len(got.requests)-1].body
	if err := json.Unmarshal(last, &req); err != nil {
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
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(parentToolCallResponse),
			replyCompletion(completion("parent answer")),
		},
		"m-reviewer": {replyCompletion(completion("child answer"))},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}
	if len(got.requests) != 3 {
		t.Fatalf("POSTs = %d, want parent, child, parent", len(got.requests))
	}
	if names := postedToolNames(t, got.requests[0].body); !reflect.DeepEqual(names, []string{"bash", "edit", "write", "subagent"}) {
		t.Errorf("parent offer = %v, want [bash edit write subagent]", names)
	}
	if names := postedToolNames(t, got.requests[1].body); !reflect.DeepEqual(names, []string{"bash"}) {
		t.Errorf("child offer = %v, want [bash] — an entry granting nothing leaves only the floor", names)
	}
}

// TestSubagentUnknownAgentIsAnOrdinaryResult: a name off the roster is an answer
// for the Model, never a crash and never a spawn.
func TestSubagentUnknownAgentIsAnOrdinaryResult(t *testing.T) {
	// Only the parent's two turns are scripted: a queue for the roster's own model
	// would answer a spawn this test is asserting never happens.
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_s","type":"function","function":{"name":"subagent","arguments":"{\"agent\":\"ghost\",\"task\":\"do the impossible\"}"}}]}}]}`),
			replyCompletion(completion("parent answer")),
		},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}
	if len(got.requests) != 2 {
		t.Fatalf("POSTs = %d, want the parent's two turns only — no spawn may be attempted", len(got.requests))
	}
	for i, req := range got.requests {
		if model := postedSpawn(t, req.body).Model; model != "m-parent" {
			t.Errorf("POST %d sent model %q, want only the parent's model", i, model)
		}
	}
	if relayed := relayedEvents(t, string(got.stdout)); len(relayed) != 0 {
		t.Errorf("stdout carries relayed child events for an unknown agent: %v", relayed)
	}
	_, resultEv := spawnPair(t, string(got.stdout))
	out, _ := resultEv["tool_output"].(string)
	if !strings.HasPrefix(out, "error:") || !strings.Contains(out, "ghost") {
		t.Errorf("tool_output = %q, want an error:-prefixed result naming the unknown agent", out)
	}
	if resultEv["is_error"] != true {
		t.Errorf("is_error = %v, want true", resultEv["is_error"])
	}
	if text := parentResult(t, string(got.stdout))["text"]; text != "parent answer" {
		t.Errorf("parent result text = %v, want the run to continue", text)
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
	specs := []agentSpec{
		{name: "planner", model: "m-planner", tools: "bash, subagent", spawns: "worker"},
		{name: "scout", model: "m-scout"},
		{name: "reviewer", model: "m-reviewer"},
		{name: "worker", model: "m-worker"},
	}

	t.Run("spawns names the child it may run", func(t *testing.T) {
		got := runDriver(t, map[string][]modelEntry{
			"m-planner": {
				replyCompletion(spawnAgentCall("worker")),
				replyCompletion(completion("parent answer")),
			},
			"m-worker": {replyCompletion(completion("child answer"))},
		}, "p",
			withRoster(specs...),
			withFlags("--agent", "planner"),
		)
		if got.code != 0 {
			t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
		}
		raw := string(got.stdout)
		if len(got.requests) != 3 {
			t.Fatalf("POSTs = %d, want the caller's turn, the child's turn, the caller's final turn: stdout = %q", len(got.requests), raw)
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
		got := runDriver(t, map[string][]modelEntry{
			"m-planner": {
				replyCompletion(spawnAgentCall("scout")),
				replyCompletion(completion("parent answer")),
			},
		}, "p",
			withRoster(specs...),
			withFlags("--agent", "planner"),
		)
		if got.code != 0 {
			t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
		}
		raw := string(got.stdout)
		if len(got.requests) != 2 {
			t.Fatalf("POSTs = %d, want the caller's two turns only — no spawn may be attempted: stdout = %q", len(got.requests), raw)
		}
		for i, req := range got.requests {
			if model := postedSpawn(t, req.body).Model; model != "m-planner" {
				t.Errorf("POST %d sent model %q, want only the caller's model", i, model)
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
// nobody — and the refusal says so rather than naming an empty list. The
// set-level call refusal is the one that answers here, because a run with nobody
// it may name is never given the Subagent Tool in the first place.
func TestSubagentSpawnOfAgentWithNoFileIsRefused(t *testing.T) {
	got := runDriver(t, map[string][]modelEntry{
		"m-planner": {
			replyCompletion(spawnAgentCall("helper")),
			replyCompletion(completion("parent answer")),
		},
	}, "p",
		withRoster(
			agentSpec{name: "planner", model: "m-planner", tools: "bash, subagent", spawns: "helper"},
			agentSpec{name: "worker", model: "m-worker"},
		),
		withFlags("--agent", "planner"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}
	raw := string(got.stdout)
	if len(got.requests) != 2 {
		t.Fatalf("POSTs = %d, want the caller's two turns only — no spawn may be attempted: stdout = %q", len(got.requests), raw)
	}
	callEv, resultEv := spawnPair(t, raw)
	out, _ := resultEv["tool_output"].(string)
	if !strings.HasPrefix(out, "error:") || resultEv["is_error"] != true {
		t.Errorf("tool_output = %q is_error = %v, want an error:-prefixed result flagged as one", out, resultEv["is_error"])
	}
	if !strings.Contains(out, `cannot call tool "subagent"`) {
		t.Errorf("tool_output = %q, want the refusal to name the Tool the call asked for", out)
	}
	if !strings.Contains(out, "bash") {
		t.Errorf("tool_output = %q, want the refusal to name the one Tool this run may call", out)
	}
	if strings.Contains(out, "worker") {
		t.Errorf("tool_output = %q names worker, which planner's spawns does not grant", out)
	}
	if callEv["tool_target"] != nil || resultEv["subagent"] != nil {
		t.Errorf("refusal events carry subagent = %v tool_target = %v, want both unset", resultEv["subagent"], callEv["tool_target"])
	}
	if relayed := relayedEvents(t, raw); len(relayed) != 0 {
		t.Errorf("stdout carries relayed child events for a spawn this run may not make: %v", relayed)
	}
	if got := parentResult(t, raw)["text"]; got != "parent answer" {
		t.Errorf("parent result text = %v, want the run to go on after the refusal", got)
	}
}

// TestSubagentTopLevelIsUnrestricted: no Agent File selects the top-level
// process, so no spawns field bounds it — it may name an entry another entry
// cannot.
func TestSubagentTopLevelIsUnrestricted(t *testing.T) {
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(spawnAgentCall("worker")),
			replyCompletion(completion("parent answer")),
		},
		"m-worker": {replyCompletion(completion("child answer"))},
	}, "p",
		withRoster(
			agentSpec{name: "planner", model: "m-planner", tools: "bash, subagent", spawns: "worker"},
			agentSpec{name: "worker", model: "m-worker"},
		),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}
	raw := string(got.stdout)
	if len(got.requests) != 3 {
		t.Fatalf("POSTs = %d, want parent, child, parent: stdout = %q", len(got.requests), raw)
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
	call := `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_s","type":"function","function":{"name":"subagent","arguments":"{\"agent\":\"reviewer\",\"task\":\"-not a flag- just a task\"}"}}]}}]}`
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(call),
			replyCompletion(completion("parent answer")),
		},
		"m-reviewer": {replyCompletion(completion("child answer"))},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}
	if len(got.requests) != 3 {
		t.Fatalf("POSTs = %d, want parent, child, parent: stdout = %q stderr = %q", len(got.requests), got.stdout, got.stderr)
	}
	child := postedSpawn(t, got.requests[1].body)
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

	res, sawResult := relaySubagentOutput(strings.NewReader(stream), "reviewer", relaySession(&out, &errOut, newRunTally()))
	if !sawResult || res.text != "child answer" {
		t.Errorf("relay returned text %q sawResult %v, want the result event's text", res.text, sawResult)
	}
	if res.failed {
		t.Errorf("relay returned failed = true for a child's real answer: %q", res.text)
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
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(withUsage(parentToolCallResponse, "m-parent", 100, 20)),
			replyCompletion(usageCompletion("m-parent", "parent answer", 30, 5)),
		},
		"m-reviewer": {replyCompletion(usageCompletion("m-reviewer", "child answer", 7, 1))},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}
	// The child ran a whole turn of its own inside the pair; the parent still
	// counts only its two Model calls.
	if turns, _ := parentResult(t, string(got.stdout))["num_turns"].(float64); int(turns) != 2 {
		t.Errorf("parent result num_turns = %v, want the parent's own two Model calls", turns)
	}
}

// TestSubagentRollUpAddsNoUsageEventOfTheParents: the child's spend reaches the
// result through the tally, never through a usage event the parent invents — a
// reader that sums the usage events it sees must still get one per parent call.
func TestSubagentRollUpAddsNoUsageEventOfTheParents(t *testing.T) {
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(withUsage(parentToolCallResponse, "m-parent", 100, 20)),
			replyCompletion(usageCompletion("m-parent", "parent answer", 30, 5)),
		},
		"m-reviewer": {replyCompletion(usageCompletion("m-reviewer", "child answer", 7, 1))},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}

	own, relayedUsage := 0, 0
	for _, ev := range got.events {
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
		t.Errorf("stdout carries %d non-relayed usage events, want exactly one per parent Model call: %q", own, got.stdout)
	}
	if relayedUsage != 1 {
		t.Errorf("stdout carries %d relayed usage events, want the child's one: %q", relayedUsage, got.stdout)
	}
}

// childBashTurnResponse is a child turn that keeps the child running: one bash
// Tool Call, so its next Model call is the one the script refuses.
const childBashTurnResponse = `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_b","type":"function","function":{"name":"bash","arguments":"{\"command\":\"true\"}"}}]}}]}`

// TestSubagentFailingChildStillContributesItsSpend is why the roll-up folds usage
// events rather than the child's result totals: a child that dies emits no result
// event at all, and the tokens it already spent are still real.
func TestSubagentFailingChildStillContributesItsSpend(t *testing.T) {
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(withUsage(parentToolCallResponse, "m-parent", 100, 20)),
			replyCompletion(usageCompletion("m-parent", "parent answer", 30, 5)),
		},
		// The child's first Model call succeeds and spends its tokens; the call after
		// its bash turn is refused, retried chatRetryAttempts times (main.go:800, and
		// a re-exec'd child always runs that default of 3), so the tail holds one
		// rejection per attempt. A dry queue trips checkNoUnscriptedRequests.
		"m-reviewer": {
			replyCompletion(withUsage(childBashTurnResponse, "m-reviewer", 7, 1)),
			reject(http.StatusInternalServerError, nil, "scripted failure"),
			reject(http.StatusInternalServerError, nil, "scripted failure"),
			reject(http.StatusInternalServerError, nil, "scripted failure"),
		},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0 — a dead child is still a Tool Result; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}

	result := parentResult(t, string(got.stdout))
	tokens := resultTokenFields(t, result)
	if tokens["input"] != 137 || tokens["output"] != 26 {
		t.Errorf("parent result tokens = %v, want parent 130/25 plus the dead child's 7/1", tokens)
	}
	// The premise of the assertion above: the child got one usage event relayed,
	// and died before it could report a result of its own.
	var relayedResult, relayedUsage int
	for _, ev := range relayedEvents(t, string(got.stdout)) {
		switch ev["kind"] {
		case "result":
			relayedResult++
		case "usage":
			relayedUsage++
		}
	}
	if relayedUsage != 1 || relayedResult != 0 {
		t.Errorf("relayed stream has %d usage and %d result events, want the child's one usage and no result: %q", relayedUsage, relayedResult, got.stdout)
	}
	_, resultEv := spawnPair(t, string(got.stdout))
	out, _ := resultEv["tool_output"].(string)
	if !strings.HasPrefix(out, "error:") {
		t.Errorf("tool_output = %q, want an error:-prefixed result", out)
	}
	if resultEv["is_error"] != true {
		t.Errorf("is_error = %v, want a child that emitted no result event reported as failed", resultEv["is_error"])
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
	tally := newRunTally()
	s := relaySession(&out, &errOut, tally)
	stream := strings.Join([]string{
		`{"kind":"usage","timestamp":1,"model":"m-reviewer"}`,
		`{"kind":"usage","timestamp":2,"tokens":"not an object"}`,
		`{"kind":"usage","timestamp":3,"tokens":{"input":"seven"}}`,
		`{"kind":"usage","timestamp":4,"tokens":null}`,
	}, "\n") + "\n"

	if _, sawResult := relaySubagentOutput(strings.NewReader(stream), "reviewer", s); sawResult {
		t.Errorf("sawResult = true, want false: the stream carries no result event")
	}
	if got := tally.totals(); got != (tokenTotals{}) {
		t.Errorf("tally totals = %+v, want unusable usage lines to contribute nothing", got)
	}
	// A model id is a separate fact from the tokens, so the one line that carried
	// it still counts.
	if got, want := tally.modelIDs(), []string{"m-reviewer"}; !reflect.DeepEqual(got, want) {
		t.Errorf("model set = %v, want the usage event's model id recorded as %v", got, want)
	}
	if events := stdoutEvents(t, out.String()); len(events) != 4 {
		t.Errorf("parent stdout got %d lines, want all four usage events relayed: %q", len(events), out.String())
	}
}

// TestSubagentRelayChargesNothingForAKindThatIsNotAString: the kind the relay loop
// parses is its only read — a non-string kind charges nothing and still relays.
func TestSubagentRelayChargesNothingForAKindThatIsNotAString(t *testing.T) {
	var out, errOut bytes.Buffer
	tally := newRunTally()
	stream := `{"kind":42,"timestamp":1,"tokens":{"input":9,"output":9},"model":"m-reviewer"}` + "\n"

	if _, sawResult := relaySubagentOutput(strings.NewReader(stream), "reviewer", relaySession(&out, &errOut, tally)); sawResult {
		t.Errorf("sawResult = true, want false: a kind of 42 is no child's result event")
	}
	if got := tally.totals(); got != (tokenTotals{}) {
		t.Errorf("tally totals = %+v, want an unusable kind to charge nothing", got)
	}
	if events := stdoutEvents(t, out.String()); len(events) != 1 {
		t.Errorf("parent stdout got %d lines, want the line relayed: %q", len(events), out.String())
	}
}

// TestSubagentRelayResultWithoutText: a result event is how the child hands over
// its answer, so one carrying no usable text is a failure rather than an empty
// success — the relay states that verdict on the value it returns, because it is
// the only reader of the child's result event, and the answer still gets relayed
// because relaying and answering are separate.
func TestSubagentRelayResultWithoutText(t *testing.T) {
	for _, stream := range []string{
		`{"kind":"result","timestamp":1,"num_turns":1}` + "\n",
		`{"kind":"result","timestamp":2,"text":42,"num_turns":1}` + "\n",
	} {
		var out, errOut bytes.Buffer
		res, sawResult := relaySubagentOutput(strings.NewReader(stream), "reviewer", relaySession(&out, &errOut, newRunTally()))
		if !sawResult {
			t.Errorf("sawResult = false, want true: the child did finish with a result event (%s)", stream)
		}
		if !strings.HasPrefix(res.text, "error:") {
			t.Errorf("relay returned text %q, want an error:-prefixed message (%s)", res.text, stream)
		}
		if !res.failed {
			t.Errorf("relay returned failed = false, want the no-answer case stating itself failed (%s)", stream)
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
	res, sawResult := relaySubagentOutput(stream, "reviewer", relaySession(&out, &errOut, newRunTally()))
	if sawResult || res.text != "" {
		t.Errorf("relay returned text %q sawResult %v, want the result after the broken line to be lost", res.text, sawResult)
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

// TestSubagentRelayFunctionKeepsTheChildTimestamp pins the forwarding half of the
// one function a Subagent Tool gets from the run: the child's line goes back out
// whole, so the clock the child stamped survives (an emit would restamp it) and
// every field the child wrote survives (an event struct drops what it lacks).
func TestSubagentRelayFunctionKeepsTheChildTimestamp(t *testing.T) {
	var out bytes.Buffer
	relayChildEvent(eventWriter{w: &out}, newRunTally())("reviewer", "text", relayedEventFields(map[string]string{
		"kind":       `"text"`,
		"timestamp":  `1234567890123`,
		"message_id": `"msg-1"`,
		"text":       `"child prose"`,
	}))

	if line := out.String(); !strings.Contains(line, `"timestamp":1234567890123`) {
		t.Errorf("relayed line restamped or rewrote the child timestamp: %q", line)
	}
	events := stdoutEvents(t, out.String())
	if len(events) != 1 {
		t.Fatalf("relay wrote %d lines, want exactly one: %q", len(events), out.String())
	}
	if events[0]["role"] != "reviewer" || events[0]["in_subagent"] != true {
		t.Errorf("relayed line = %v, want role reviewer and in_subagent true", events[0])
	}
	if events[0]["message_id"] != "msg-1" || events[0]["text"] != "child prose" {
		t.Errorf("relayed line dropped a field the child wrote: %v", events[0])
	}
}

// TestSubagentRelayFunctionFoldsUsageIntoTheTally pins the accounting half: the
// relay is the only way a child's spend reaches the run, and what it charged is
// read back through the projections the result event reads.
func TestSubagentRelayFunctionFoldsUsageIntoTheTally(t *testing.T) {
	var out bytes.Buffer
	tally := newRunTally()

	relayChildEvent(eventWriter{w: &out}, tally)("reviewer", "usage", relayedUsageFields(`{"input":7,"output":1,"cost_usd":0.25}`, `"m-reviewer"`))

	if got, want := tally.totals(), (tokenTotals{Input: 7, Output: 1, CostUSD: 0.25}); got != want {
		t.Errorf("totals() = %+v, want the relayed usage projected as %+v", got, want)
	}
	if got, want := tally.modelIDs(), []string{"m-reviewer"}; !reflect.DeepEqual(got, want) {
		t.Errorf("modelIDs() = %v, want the child model that answered as %v", got, want)
	}
	if events := stdoutEvents(t, out.String()); len(events) != 1 {
		t.Errorf("relay wrote %d lines, want the usage line once: %q", len(events), out.String())
	}
}

// TestSubagentRelayFunctionFoldsNothingFromANonUsageLine: a child's result carries
// the same tokens its usage events already did, so folding anything but usage
// would bill them twice — and the line still gets relayed.
func TestSubagentRelayFunctionFoldsNothingFromANonUsageLine(t *testing.T) {
	var out bytes.Buffer
	tally := newRunTally()

	relayChildEvent(eventWriter{w: &out}, tally)("reviewer", "result", relayedEventFields(map[string]string{
		"kind":      `"result"`,
		"timestamp": `1`,
		"tokens":    `{"input":9000,"output":9}`,
		"model":     `"m-reviewer"`,
	}))

	if got := tally.totals(); got != (tokenTotals{}) {
		t.Errorf("totals() = %+v, want a non-usage line to fold nothing", got)
	}
	if got := tally.modelIDs(); len(got) != 0 {
		t.Errorf("modelIDs() = %v, want a non-usage line to record no model", got)
	}
	if events := stdoutEvents(t, out.String()); len(events) != 1 || events[0]["kind"] != "result" {
		t.Errorf("relay wrote %q, want the child's result line relayed", out.String())
	}
}

// TestSubagentRelayFunctionFoldsNothingFromUnusableUsageTokens: one odd child line
// is never a reason to corrupt the totals or to withhold a line — but a model id
// is its own fact and still counts when the tokens beside it are not.
func TestSubagentRelayFunctionFoldsNothingFromUnusableUsageTokens(t *testing.T) {
	for _, tokens := range []string{"", "null", `"not an object"`, `{"input":"seven"}`} {
		var out bytes.Buffer
		tally := newRunTally()

		relayChildEvent(eventWriter{w: &out}, tally)("reviewer", "usage", relayedUsageFields(tokens, `"m-reviewer"`))

		if got := tally.totals(); got != (tokenTotals{}) {
			t.Errorf("tokens %q: totals() = %+v, want nothing folded", tokens, got)
		}
		if got, want := tally.modelIDs(), []string{"m-reviewer"}; !reflect.DeepEqual(got, want) {
			t.Errorf("tokens %q: modelIDs() = %v, want the model id still recorded as %v", tokens, got, want)
		}
		if events := stdoutEvents(t, out.String()); len(events) != 1 {
			t.Errorf("tokens %q: relay wrote %d lines, want the usage line relayed: %q", tokens, len(events), out.String())
		}
	}
}

// TestSubagentRelayFunctionWritesOneLineInItsOwnKeyOrder pins the wire shape:
// re-encoding a map sorts its keys, so a relayed line carries the parent's key
// order and the child's values, one object per line.
func TestSubagentRelayFunctionWritesOneLineInItsOwnKeyOrder(t *testing.T) {
	var out bytes.Buffer
	relay := relayChildEvent(eventWriter{w: &out}, newRunTally())

	relay("reviewer", "usage", relayedUsageFields(`{"input":7,"output":1}`, `"m-reviewer"`))
	relay("planner", "text", relayedEventFields(map[string]string{"kind": `"text"`, "timestamp": `2`}))

	want := "{\"in_subagent\":true,\"kind\":\"usage\",\"model\":\"m-reviewer\",\"role\":\"reviewer\",\"tokens\":{\"input\":7,\"output\":1}}\n" +
		"{\"in_subagent\":true,\"kind\":\"text\",\"role\":\"planner\",\"timestamp\":2}\n"
	if got := out.String(); got != want {
		t.Errorf("relay wrote %q, want %q", got, want)
	}
}

// relaySession builds the session a test needs when it calls the relay directly
// rather than through a whole run: the tally it hands the relay is how such a test
// reads back what a child's lines were charged.
func relaySession(out, errOut io.Writer, tally *runTally) session {
	return session{stderr: errOut, relay: relayChildEvent(eventWriter{w: out}, tally)}
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
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(withUsage(parentToolCallResponse, "m-parent", 100, 20)),
			replyCompletion(usageCompletion("m-parent", "parent answer", 30, 5)),
		},
		"m-reviewer": {replyCompletion(usageCompletion("m-reviewer", "child answer", 7, 1))},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}

	tokens := resultTokenFields(t, parentResult(t, string(got.stdout)))
	want := map[string]float64{"input": 137, "output": 26, "cache_read": 0, "cache_write_5m": 0, "cache_write_1h": 0, "cost_usd": 0}
	if !reflect.DeepEqual(tokens, want) {
		t.Errorf("parent result tokens = %v, want parent 130/25 plus child 7/1 = %v", tokens, want)
	}
	if tokens["input"] == 130 && tokens["output"] == 25 {
		t.Errorf("parent result tokens are the parent's own %v; the child's spend is missing", tokens)
	}
}

// TestSubagentChildModelRollsIntoParentResultModels: a model that answered the
// session answered it, whoever delegated the call to it.
func TestSubagentChildModelRollsIntoParentResultModels(t *testing.T) {
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(withUsage(parentToolCallResponse, "m-parent", 100, 20)),
			replyCompletion(usageCompletion("m-parent", "parent answer", 30, 5)),
		},
		"m-reviewer": {replyCompletion(usageCompletion("m-reviewer", "child answer", 7, 1))},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}

	// Sorted, because sortedModelIDs is the only order the result contract has.
	if models := resultStrings(t, parentResult(t, string(got.stdout)), "models"); !reflect.DeepEqual(models, []string{"m-parent", "m-reviewer"}) {
		t.Errorf("parent result models = %v, want both model ids sorted", models)
	}
}

// denyHookRosterFixture writes the real Guard Hook script a guarded roster needs:
// one that denies only a command containing the word "denied", for a reason
// naming a credential path. The roster and the Hook Manifest naming this script
// are the runs' to compose — the first through withRoster, the second through
// withHookManifest — so the contrast test below passes no manifest at all and
// nothing else changes.
func denyHookRosterFixture(t *testing.T, reason string) (hookPath string) {
	t.Helper()
	// The hook reads the protocol payload on stdin, so the match is on the command
	// text the child asked for and not on anything the shell would expand.
	hookPath = writeGuardHookScript(t, "deny-credential-path.sh",
		`input=$(cat)
	case "$input" in
	  *denied*) echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"`+reason+`"}}' ;;
	esac`)
	return hookPath
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
	hookPath := denyHookRosterFixture(t, "reads the credential path /run/secrets/spindrift/token")

	dir := t.TempDir()
	deniedMarker := filepath.Join(dir, "denied-credential-read.marker")
	cleanMarker := filepath.Join(dir, "clean-ran.marker")
	deniedCmd := "printf nope > " + deniedMarker
	cleanCmd := "printf clean-ran | tee " + cleanMarker

	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(parentToolCallResponse),
			replyCompletion(completion("parent answer")),
		},
		"m-reviewer": {
			replyCompletion(childGuardTurnResponse(deniedCmd, cleanCmd)),
			replyCompletion(completion("child answer")),
		},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withHookManifest(`{"pre":[{"path":"`+hookPath+`","tools":["Bash"]}]}`),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0 — a refusal inside a child is a Tool Result, not a failed run; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}
	raw := string(got.stdout)

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

// TestSubagentCallIsShownToNoHook is the ADR's claim for the Subagent Tool made
// testable: a spawn declares no shape of its own, so no hook is shown it — not
// even a pre entry naming no tools, which toolMatches reads as every tool. The
// script denies whatever it is shown and touches a marker on every run, and the
// child here makes no Tool Call of its own, so a marker can only mean the
// parent's spawn was put in front of the chain. That a denied call made *by the
// child* is refused is TestSubagentChildHonoursTheConfiguredDenyHook, which also
// proves the child's clean call runs; nothing is repeated here.
func TestSubagentCallIsShownToNoHook(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "hook-ran.marker")
	hookPath := writeGuardHookScript(t, "deny-everything.sh",
		`: > `+marker+`
echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"no call is allowed here"}}'`)

	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(parentToolCallResponse),
			replyCompletion(completion("parent answer")),
		},
		"m-reviewer": {
			replyCompletion(completion("child answer")),
		},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		// No "tools" key at all: this entry is asked about every tool the chain is
		// asked about, which is exactly why the spawn must not be one of them.
		withHookManifest(`{"pre":[{"path":"`+hookPath+`"}]}`),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0 — a spawn no hook was shown is an ordinary run; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}
	raw := string(got.stdout)

	if hookMarkerExists(t, marker) {
		t.Errorf("marker %s exists: the hook ran, so the parent's Subagent call was shown to it", marker)
	}

	_, subagentResult := spawnPair(t, raw)
	if v, present := subagentResult["is_error"]; present {
		t.Errorf("Subagent Tool Result is_error = %v, want a spawn that ran: %v", v, subagentResult)
	}
	out, _ := subagentResult["tool_output"].(string)
	if strings.Contains(out, hookPath) || strings.Contains(out, "no call is allowed here") {
		t.Errorf("Subagent Tool Result tool_output = %q, want no guard refusal for a call no hook is shown", out)
	}
	if got := subagentResult["tool_output"]; got != "child answer" {
		t.Errorf("Subagent Tool Result tool_output = %v, want the child's answer: the spawn has to run", got)
	}
	if got := parentResult(t, raw)["text"]; got != "parent answer" {
		t.Errorf("parent result text = %v, want the parent's own final answer", got)
	}
}

// TestSubagentChildGuardComesFromTheConfiguredManifest is the config half of the
// criterion: with the same roster, the same hook script still on disk, and the
// same scripted child turn, a Driver Config that names no hook_manifest leaves
// the child unguarded. Since the parent hands the child
// nothing but argv and the environment, the key the two runs differ by is the
// only thing that can have guarded the child in the test above.
func TestSubagentChildGuardComesFromTheConfiguredManifest(t *testing.T) {
	hookPath := denyHookRosterFixture(t, "credential path")

	marker := filepath.Join(t.TempDir(), "unguarded-denied-ran.marker")
	deniedCmd := "printf ran > " + marker

	// No withHookManifest: the missing hook_manifest key is all that separates this
	// run from the guarded one above.
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(parentToolCallResponse),
			replyCompletion(completion("parent answer")),
		},
		"m-reviewer": {
			replyCompletion(childGuardTurnResponse(deniedCmd, deniedCmd)),
			replyCompletion(completion("child answer")),
		},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}
	raw := string(got.stdout)

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
	got := runDriver(t, map[string][]modelEntry{
		"m-worker": {
			replyCompletion(bashCallTurn("call_1", "echo one")),
			replyCompletion(bashCallTurn("call_2", "echo one")),
			replyCompletion(bashCallTurn("call_3", "echo one")),
			replyCompletion(completion("child answer")),
		},
	}, "p",
		withRoster(agentSpec{name: "worker", model: "m-worker", tools: "bash"}),
		withFlags("--agent", "worker"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0 — a refusal is a Tool Result the run answers past, not a failed run; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}
	raw := string(got.stdout)
	if len(got.requests) != 4 {
		t.Fatalf("POSTs = %d, want three call turns and the answer: the third call must not be sent again, but the refusal is only a Tool Result: stdout = %q", len(got.requests), raw)
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
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(bashCallTurn("call_p1", "echo same")),
			replyCompletion(spawnAgentCall("worker")),
			replyCompletion(bashCallTurn("call_p2", "echo same")),
			replyCompletion(completion("parent answer")),
		},
		"m-worker": {
			replyCompletion(bashCallTurn("call_c1", "echo same")),
			replyCompletion(bashCallTurn("call_c2", "echo same")),
			replyCompletion(bashCallTurn("call_c3", "echo same")),
			replyCompletion(completion("child answer")),
		},
	}, "p",
		withRoster(agentSpec{name: "worker", model: "m-worker", tools: "bash"}),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}
	raw := string(got.stdout)
	if len(got.requests) != 8 {
		t.Fatalf("POSTs = %d, want the script of 8 in full (the refusal must not stop either run short): stdout = %q", len(got.requests), raw)
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
	got := runDriver(t, map[string][]modelEntry{
		"m-parent": {
			replyCompletion(parentToolCallResponse),
			replyCompletion(completion("parent answer")),
		},
		"m-reviewer": {replyCompletion(completion("child answer"))},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", body: "review things"}),
		withEnv(envTestNowMS, fmt.Sprintf("%d:%d", pinnedStart, pinnedStep)),
		withFlags("--model", "m-parent"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stdout = %q stderr = %q", got.code, got.stdout, got.stderr)
	}
	raw := string(got.stdout)

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
