package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// planWriteCall reads one Write Tool call and plans the write of its whole file.
//
// The path is resolved here at plan time rather than inside run: the tool_call
// event carries plan.target and goes out on the wire before plan.run() is
// called, so resolving separately for the event and for the write would let a
// run report one path and touch another. filepath.Abs resolves against the
// process cwd, which is where a file Tool acts — there is no per-run working
// directory to consult.
//
// s is unused so far: the Guard Hook chain this call will answer to is wired in
// separately, and until then nothing about the run changes what a write does.
func planWriteCall(call toolCall, input any, s session) toolPlan {
	var args struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return writeErrorPlan(input, "error: cannot parse tool arguments: %v", err)
	}
	// Checked before resolving: filepath.Abs("") succeeds by naming the cwd, so a
	// call that omitted path would otherwise be planned as a write onto the run's
	// own working directory instead of an error the Model can correct.
	if args.Path == "" {
		return writeErrorPlan(input, "error: path is required")
	}
	resolved, err := filepath.Abs(args.Path)
	if err != nil {
		return writeErrorPlan(input, "error: cannot resolve path %q: %v", args.Path, err)
	}
	return toolPlan{
		input:   input,
		target:  resolved,
		isError: fileToolResultIsError,
		run:     func() string { return writeFileTool(resolved, args.Content) },
	}
}

// writeErrorPlan is a Write Tool plan that never touches the filesystem. It
// carries fileToolResultIsError rather than leaving isError nil: an inherited
// toolResultIsError happens to answer true for these messages, but only because
// they lack an exit-code line, and a rule that works by accident is one a reword
// can silently turn into a reported success.
func writeErrorPlan(input any, format string, a ...any) toolPlan {
	msg := fmt.Sprintf(format, a...)
	return toolPlan{input: input, isError: fileToolResultIsError, run: func() string { return msg }}
}

// fileToolResultIsError is the is_error rule for the file Tools' results. A
// write reports a sentence rather than a command's output, so it has no trailing
// exit-code line for toolResultIsError's suffix test to find — inheriting that
// rule would report every successful write as a failure. The "error:" prefix a
// failed write returns with is the only failure signal such a result can carry.
func fileToolResultIsError(output string) bool {
	return strings.HasPrefix(output, "error:")
}

// writeFileTool writes content over path, creating the parents the Model did not
// mention: making it spend a turn pre-creating a directory before it can name
// the file it wants is a turn wasted for a fact the host can supply.
func writeFileTool(path, content string) string {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Sprintf("error: cannot write %s: %v", path, err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return fmt.Sprintf("error: cannot write %s: %v", path, err)
	}
	return fmt.Sprintf("wrote %d bytes to %s", len(content), path)
}
