package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
)

// Subagent Tool: one roster entry runs as a child Driver process of this one, so
// the child gets its own context window and its own conversation, and the parent
// sees only the answer. The child is this same binary re-invoked with --agent,
// which is what makes the model id and the system prompt come from the Agent File
// rather than from anything the parent composed.

// planSubagentCall reads one Subagent Tool call. The roster is the only place an
// agent comes from, so a name that is not in it is an ordinary Tool Result the
// Model can correct — the same standing an unknown tool has.
func planSubagentCall(call toolCall, input any, cfg config, events eventWriter, stderr io.Writer) toolPlan {
	var args struct {
		Agent string `json:"agent"`
		Task  string `json:"task"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return toolPlan{input: input, run: func() string { return fmt.Sprintf("error: cannot parse tool arguments: %v", err) }}
	}
	entry, ok := findAgent(cfg.agents, args.Agent)
	if !ok {
		return toolPlan{input: input, run: func() string { return fmt.Sprintf("error: unknown agent %q", args.Agent) }}
	}
	return toolPlan{
		input:    input,
		target:   entry.Name,
		subagent: entry.Name,
		isError:  subagentResultIsError,
		run:      func() string { return runSubagentTool(entry, args.Task, events, stderr) },
	}
}

// subagentResultIsError is the is_error rule for a Subagent Tool Result: the
// child's answer carries no exit-code line for toolResultIsError to read, so the
// error: prefix a failed spawn returns with is the only failure signal the result
// can carry. A child's answer that merely begins with the word "error:" is a
// false positive this accepts — the alternative is reporting every failure as a
// success.
func subagentResultIsError(output string) bool {
	return strings.HasPrefix(output, "error:")
}

// runSubagentTool starts the child Driver and returns its answer as the Tool
// Result. A child that dies is an error-shaped result, never a failed run: the
// Model decides what to do about it, so the parent's exit code stays 0.
func runSubagentTool(entry agentEntry, task string, events eventWriter, stderr io.Writer) string {
	exe, err := os.Executable()
	if err != nil {
		return fmt.Sprintf("error: cannot locate the Driver binary: %v", err)
	}
	// No cmd.Dir and no cmd.Env: inheriting the working directory and the
	// environment is the whole point of the spawn — it is how the child resolves
	// the same Driver Config, and therefore the same roster, the parent did.
	// "--agent" alone names the entry; --model stays out of argv so the model id
	// can only come from the Agent File, and "--" keeps a dash-leading task from
	// being read as the value of the last flag.
	cmd := exec.Command(exe, "--agent", entry.Name, "--", task)
	// Stdin is deliberately nil rather than inherited: a child that read the
	// parent's stdin would consume the operator's prompt and then block on input
	// nobody is going to type.
	cmd.Stdin = nil
	// The child's stderr is the operator's progress copy and goes straight
	// through; its stdout is JSONL, so it is piped and re-emitted, never shared.
	cmd.Stderr = stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Sprintf("error: cannot start subagent %q: %v", entry.Name, err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Sprintf("error: cannot start subagent %q: %v", entry.Name, err)
	}
	// The pipe has to be drained before Wait, which closes it: the child's result
	// line is the last thing it writes and would be lost otherwise.
	text, sawResult := relaySubagentOutput(stdout, entry.Name, events, stderr)
	if err := cmd.Wait(); err != nil {
		return fmt.Sprintf("error: subagent %q failed: %v", entry.Name, err)
	}
	if !sawResult {
		return fmt.Sprintf("error: subagent %q emitted no result event", entry.Name)
	}
	return text
}

// relaySubagentOutput forwards every event line the child wrote onto the parent's
// stdout as it arrives, attributed to the agent, and reports the text of the
// child's own result event.
//
// Lines go through eventWriter.encode and never through emit: emit restamps the
// clock, and the child's own timestamp is the fact worth keeping. Re-encoding a
// map sorts its keys, so a relayed line is not byte-identical to the child's own
// — the values are, the key order is not.
//
// A child line larger than sseMaxLineBytes, and anything that is not a JSON
// object, is reported on stderr rather than dropped quietly: stdout carries only
// JSONL, so a stray line cannot be forwarded, and forwarding it raw would put a
// malformed event on the wire.
func relaySubagentOutput(r io.Reader, agentName string, events eventWriter, stderr io.Writer) (string, bool) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), sseMaxLineBytes)
	text, sawResult := "", false
	for scanner.Scan() {
		line := scanner.Bytes()
		// RawMessage rather than any: the child's numbers round-trip byte-exactly
		// instead of being re-rendered out of float64.
		var fields map[string]json.RawMessage
		// A bare "null" decodes into a nil map with no error at all, and there is
		// nothing in it to attribute, so only a populated object is an event.
		if err := json.Unmarshal(line, &fields); err != nil || fields == nil {
			fmt.Fprintf(stderr, "spindrift-driver: subagent %q: dropped non-event output line: %s\n", agentName, boundBody(line))
			continue
		}
		relayed := make(map[string]any, len(fields)+2)
		for k, v := range fields {
			relayed[k] = v
		}
		relayed["role"] = agentName
		relayed["in_subagent"] = true
		_ = events.encode(relayed)

		var kind string
		_ = json.Unmarshal(fields["kind"], &kind)
		if kind == "result" {
			sawResult = true
			var answer string
			if err := json.Unmarshal(fields["text"], &answer); err == nil {
				text = answer
			}
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(stderr, "spindrift-driver: subagent %q: cannot read its output: %v\n", agentName, err)
	}
	return text, sawResult
}
