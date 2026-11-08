package main

import (
	"bufio"
	"encoding/json"
	"errors"
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

// planSubagentCall reads one Subagent Tool call. Agents come only from the
// roster and a run may only hand work to the entries its own entry names, so a
// name outside that set is an ordinary Tool Result the Model can correct — the
// same standing an unknown tool has.
func planSubagentCall(call toolCall, input any, s session) toolPlan {
	var args struct {
		Agent string `json:"agent"`
		Task  string `json:"task"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return toolPlan{input: input, run: func() string { return fmt.Sprintf("error: cannot parse tool arguments: %v", err) }}
	}
	// Looking up against the caller's permitted set covers both refusals the issue
	// unifies: a name with no Agent File, and one its spawns does not permit. The
	// top-level process names no entry, so nothing bounds it.
	permitted := spawnable(s.cfg.agent, s.cfg.agents)
	entry, ok := findAgent(permitted, args.Agent)
	if !ok {
		return toolPlan{input: input, run: func() string { return spawnRefusal(args.Agent, permitted) }}
	}
	return toolPlan{
		input:    input,
		target:   entry.Name,
		subagent: entry.Name,
		isError:  subagentResultIsError,
		run:      func() string { return runSubagentTool(entry, args.Task, s) },
	}
}

// spawnRefusal is the answer to a spawn this run may not make; naming the agents
// it may spawn is what makes it correctable in one step rather than a guess.
func spawnRefusal(name string, permitted []agentEntry) string {
	if len(permitted) == 0 {
		return fmt.Sprintf("error: cannot spawn agent %q: this run may spawn no agents", name)
	}
	names := make([]string, 0, len(permitted))
	for _, entry := range permitted {
		names = append(names, entry.Name)
	}
	return fmt.Sprintf("error: cannot spawn agent %q: this run may spawn %s", name, strings.Join(names, ", "))
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
func runSubagentTool(entry agentEntry, task string, s session) string {
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
	cmd.Stderr = s.stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Sprintf("error: cannot start subagent %q: %v", entry.Name, err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Sprintf("error: cannot start subagent %q: %v", entry.Name, err)
	}
	// The pipe has to be drained before Wait, which closes it: the child's result
	// line is the last thing it writes and would be lost otherwise.
	text, sawResult := relaySubagentOutput(stdout, entry.Name, s)
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
// child's own result event — or, when that event carries no text, an error: message,
// because a child that cannot hand over an answer has not succeeded.
//
// Lines go through eventWriter.encode and never through emit: emit restamps the
// clock, and the child's own timestamp is the fact worth keeping. Re-encoding a
// map sorts its keys, so a relayed line is not byte-identical to the child's own
// — the values are, the key order is not.
//
// A child line larger than sseMaxLineBytes, and anything that is not a JSON
// object, is reported on stderr rather than dropped quietly: stdout carries only
// JSONL, so a stray line cannot be forwarded, and forwarding it raw would put a
// malformed event on the wire. An over-long line ends the scan — the stream broke
// the protocol — and the rest of the child's output is drained and lost, which is
// why such a run reports no result event at all.
func relaySubagentOutput(r io.Reader, agentName string, s session) (string, bool) {
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
			fmt.Fprintf(s.stderr, "spindrift-driver: subagent %q: dropped non-event output line: %s\n", agentName, boundBody(line))
			continue
		}
		relayed := make(map[string]any, len(fields)+2)
		for k, v := range fields {
			relayed[k] = v
		}
		relayed["role"] = agentName
		relayed["in_subagent"] = true
		_ = s.events.encode(relayed)

		var kind string
		_ = json.Unmarshal(fields["kind"], &kind)
		switch kind {
		case "usage":
			foldChildUsage(fields, s)
		case "result":
			// sawResult reports that the child finished, which is a separate fact from
			// whether it had an answer: a result with no text, or text that is not a
			// string, would otherwise come back as an empty successful delegation.
			sawResult = true
			var answer string
			if err := json.Unmarshal(fields["text"], &answer); err != nil {
				text = fmt.Sprintf("error: subagent %q emitted a result event with no text", agentName)
			} else {
				text = answer
			}
		}
	}
	if err := scanner.Err(); err != nil {
		note := ""
		if errors.Is(err, bufio.ErrTooLong) {
			note = fmt.Sprintf(" (a line exceeded the %d byte bound)", sseMaxLineBytes)
		}
		fmt.Fprintf(s.stderr, "spindrift-driver: subagent %q: cannot read its output%s: %v\n", agentName, note, err)
		// The child is still writing into a pipe this process owns and there is no
		// child-side timeout to free it, so the leftovers have to be discarded here
		// or cmd.Wait blocks on a full pipe. Everything buffered past the break is
		// the accepted loss: the run ends as "emitted no result event".
		_, _ = io.Copy(io.Discard, r)
	}
	return text, sawResult
}

// foldChildUsage charges one relayed usage event to the parent: its tokens into
// the delegated tally and its model id into the run's model set.
//
// Usage events are folded rather than the child's result totals because a child
// that fails never emits a result event while the spend it made up to that point
// is real, and because folding per usage event is exactly how the parent builds
// its own tally — parent and child are then accounted the same way. Folding the
// child's result totals on top of this would count the same tokens twice.
//
// A line whose tokens key is missing or is not a token object contributes
// nothing: one odd child line must not corrupt the parent's totals.
func foldChildUsage(fields map[string]json.RawMessage, s session) {
	if raw, ok := fields["tokens"]; ok {
		var spent tokenTotals
		if err := json.Unmarshal(raw, &spent); err == nil {
			*s.delegated = s.delegated.add(spent)
		}
	}
	if raw, ok := fields["model"]; ok {
		var model string
		if err := json.Unmarshal(raw, &model); err == nil && model != "" {
			s.models[model] = struct{}{}
		}
	}
}
