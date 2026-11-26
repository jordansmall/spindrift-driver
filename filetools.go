package main

import (
	"encoding/json"
	"errors"
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
// The Guard Hook chain is not asked here: the plan declares its Read
// presentation and guardRun — the one sequence dispatch runs every plan through
// — asks the chain when the call runs rather than while it is planned, so the
// tool_call event is already on the wire before any hook fires, exactly as it is
// for bash. A refusal therefore arrives before this closure is ever called: it
// states its own failure, because nothing reached the disk and there is no write
// verdict to read. The subject handed back is the declared path — a path has
// nothing for a hook to rewrite — so run acts on the one resolved value the
// event reported.
func planWriteCall(call toolCall, input any, s session) toolPlan {
	var args struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return fileErrorPlan(input, unparseableArgs, err)
	}
	resolved, err := resolveFilePath(args.Path)
	if err != nil {
		return fileErrorPlan(input, "error: %v", err)
	}
	return toolPlan{
		input:        input,
		target:       resolved,
		presentation: readPresentation(resolved),
		run: func(subject string) toolResult {
			return writeFileTool(subject, args.Content)
		},
	}
}

// resolveFilePath turns a file Tool call's path argument into the absolute path
// its plan reports and its run touches. Both file Tools share it so the refusal
// wording cannot drift between them.
func resolveFilePath(path string) (string, error) {
	// Checked before resolving: filepath.Abs("") succeeds by naming the cwd, so a
	// call that omitted path would otherwise be planned as a write onto the run's
	// own working directory instead of an error the Model can correct.
	if path == "" {
		return "", errors.New("path is required")
	}
	resolved, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("cannot resolve path %q: %v", path, err)
	}
	return resolved, nil
}

// planEditCall reads one Edit Tool call and plans the exact-text replacement in
// its file. It shares planWriteCall's reason for resolving the path here rather
// than inside run, for declaring the Read presentation rather than guarding in
// run, and that one resolved value is both plan.target and the file the edit
// rewrites.
func planEditCall(call toolCall, input any, s session) toolPlan {
	var args struct {
		Path       string `json:"path"`
		OldString  string `json:"old_string"`
		NewString  string `json:"new_string"`
		ReplaceAll bool   `json:"replace_all"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return fileErrorPlan(input, unparseableArgs, err)
	}
	// Refused rather than counted: strings.Count(s, "") is len(s)+1, so an empty
	// old string reads as a file full of matches and a replace_all would scatter
	// new_string between every byte of it.
	if args.OldString == "" {
		return fileErrorPlan(input, "error: old_string is required")
	}
	resolved, err := resolveFilePath(args.Path)
	if err != nil {
		return fileErrorPlan(input, "error: %v", err)
	}
	return toolPlan{
		input:        input,
		target:       resolved,
		presentation: readPresentation(resolved),
		run: func(subject string) toolResult {
			return editFileTool(subject, args.OldString, args.NewString, args.ReplaceAll)
		},
	}
}

// editFileTool rewrites path with the exact text oldString swapped for newString.
func editFileTool(path, oldString, newString string, replaceAll bool) toolResult {
	body, err := os.ReadFile(path)
	if err != nil {
		return toolResult{text: fmt.Sprintf("error: cannot read %s: %v", path, err), failed: true}
	}
	// Counted before anything is written: an edit that guessed which occurrence was
	// meant changes the file in a way the Model cannot spot, so an ambiguous call
	// comes back as a result naming the count instead.
	count := strings.Count(string(body), oldString)
	switch {
	case count == 0:
		return toolResult{text: fmt.Sprintf("error: old_string not found in %s", path), failed: true}
	case count > 1 && !replaceAll:
		return toolResult{text: fmt.Sprintf("error: %d matches of old_string in %s; set replace_all to replace every match", count, path), failed: true}
	}
	// A unique match and an accepted replace_all differ only in count by now.
	out := strings.Replace(string(body), oldString, newString, count)
	if err := os.WriteFile(path, []byte(out), 0o644); err != nil {
		return toolResult{text: fmt.Sprintf("error: cannot write %s: %v", path, err), failed: true}
	}
	noun := "match"
	if count != 1 {
		noun = "matches"
	}
	return toolResult{text: fmt.Sprintf("edited %s: replaced %d %s", path, count, noun)}
}

// fileErrorPlan is a file Tool plan that never touches the filesystem: nothing
// ran, so the plan states the failure the call text reports rather than leaving a
// verdict to be inferred from how that text reads. It declares no presentation
// on purpose: a call that never resolved a path gives the Guard chain no subject
// to rule on, so no hook is ever asked for it.
func fileErrorPlan(input any, format string, a ...any) toolPlan {
	return toolPlan{
		input:        input,
		presentation: nonePresentation(),
		run:          func(string) toolResult { return failedResult(format, a...) },
	}
}

// writeFileTool writes content over path, creating the parents the Model did not
// mention: making it spend a turn pre-creating a directory before it can name
// the file it wants is a turn wasted for a fact the host can supply.
func writeFileTool(path, content string) toolResult {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return toolResult{text: fmt.Sprintf("error: cannot write %s: %v", path, err), failed: true}
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return toolResult{text: fmt.Sprintf("error: cannot write %s: %v", path, err), failed: true}
	}
	return toolResult{text: fmt.Sprintf("wrote %d bytes to %s", len(content), path)}
}
