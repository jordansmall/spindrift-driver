package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeCall builds a Write Tool call the way the wire delivers one: the arguments
// already a JSON string, so the plan is what has to decode them.
func writeCall(args string) toolCall {
	return toolCall{ID: "call_w", Type: "function", Function: toolCallFunction{Name: "write", Arguments: args}}
}

// editCall builds an Edit Tool call with the arguments left as the JSON string
// the wire delivers, so the plan is what has to decode them.
func editCall(args string) toolCall {
	return toolCall{ID: "call_e", Type: "function", Function: toolCallFunction{Name: "edit", Arguments: args}}
}

// runFilePlan runs a file Tool plan the way dispatch runs it, through guardRun.
// The closure acts on the subject the Guard chain hands back rather than on one
// a test invents, so a hook-free run acts on the resolved path the chain answers
// with — the same value plan.target reports.
func runFilePlan(plan toolPlan) toolResult {
	return session{}.guardRun(plan)
}

func TestPlanWriteCallCreatesFile(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	plan := planWriteCall(writeCall(`{"path":"out.txt","content":"hello, world"}`), nil, session{})
	if res := runFilePlan(plan); res.failed {
		t.Fatalf("plan reported an error, want a created file: %s", res.text)
	}

	want := filepath.Join(dir, "out.txt")
	if plan.target != want {
		t.Errorf("plan.target = %q, want %q", plan.target, want)
	}
	b, err := os.ReadFile(want)
	if err != nil {
		t.Fatalf("read created file: %v", err)
	}
	if string(b) != "hello, world" {
		t.Errorf("file content = %q, want %q", b, "hello, world")
	}
}

// TestWriteFileToolStatesFailureFromAFailedWrite pins the verdict at the leaf that
// can actually know it: a write that never landed states its own failure as a
// value, so nothing downstream has to read its wording to decide.
func TestWriteFileToolStatesFailureFromAFailedWrite(t *testing.T) {
	dir := t.TempDir()
	blocker := filepath.Join(dir, "blocker")
	if err := os.WriteFile(blocker, []byte("not a directory"), 0o644); err != nil {
		t.Fatalf("pre-write blocker: %v", err)
	}

	blocked := writeFileTool(filepath.Join(blocker, "child.txt"), "nope")
	if !strings.HasPrefix(blocked.text, "error:") {
		t.Errorf("result = %q, want the error: wording the Model already sees", blocked.text)
	}
	if !blocked.failed {
		t.Errorf("failed = false for %q, want the failed write to say so", blocked.text)
	}

	written := writeFileTool(filepath.Join(dir, "ok.txt"), "written")
	if written.failed {
		t.Errorf("failed = true for a successful write: %q", written.text)
	}
	if want := "wrote 7 bytes to " + filepath.Join(dir, "ok.txt"); written.text != want {
		t.Errorf("result = %q, want %q", written.text, want)
	}
}

func TestPlanWriteCallMissingPath(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	plan := planWriteCall(writeCall(`{"content":"orphan"}`), nil, session{})
	res := runFilePlan(plan)
	if !strings.HasPrefix(res.text, "error:") {
		t.Errorf("result = %q, want an error: prefix", res.text)
	}
	if !strings.Contains(res.text, "path") {
		t.Errorf("result = %q, want it to name the missing path argument", res.text)
	}
	if !res.failed {
		t.Errorf("is_error = false for %q, want true", res.text)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read temp dir: %v", err)
	}
	if len(entries) != 0 {
		t.Errorf("temp dir contains %v, want nothing written", entries)
	}
}

func TestPlanWriteCallResultNamesBytesAndPath(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	plan := planWriteCall(writeCall(`{"path":"a/b.txt","content":"12345"}`), nil, session{})
	res := runFilePlan(plan)
	if res.failed {
		t.Fatalf("is_error = true for a successful write: %q", res.text)
	}
	want := filepath.Join(dir, "a", "b.txt")
	if !strings.Contains(res.text, "5") || !strings.Contains(res.text, want) {
		t.Errorf("result = %q, want it to name the byte count and %q", res.text, want)
	}
}

func TestPlanWriteCallCreatesMissingParents(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	plan := planWriteCall(writeCall(`{"path":"nested/deep/file.txt","content":"deep"}`), nil, session{})
	if res := runFilePlan(plan); res.failed {
		t.Fatalf("plan reported an error, want the parents created: %s", res.text)
	}
	want := filepath.Join(dir, "nested", "deep", "file.txt")
	if plan.target != want {
		t.Errorf("plan.target = %q, want %q", plan.target, want)
	}
	b, err := os.ReadFile(want)
	if err != nil {
		t.Fatalf("read created file: %v", err)
	}
	if string(b) != "deep" {
		t.Errorf("file content = %q, want %q", b, "deep")
	}
}

func TestPlanWriteCallOverwritesCompletely(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	path := filepath.Join(dir, "both.txt")
	if err := os.WriteFile(path, []byte("the old and much longer body"), 0o644); err != nil {
		t.Fatalf("pre-write: %v", err)
	}
	plan := planWriteCall(writeCall(`{"path":"both.txt","content":"new"}`), nil, session{})
	if res := runFilePlan(plan); res.failed {
		t.Fatalf("plan reported an error, want an overwrite: %s", res.text)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read overwritten file: %v", err)
	}
	if string(b) != "new" {
		t.Errorf("file content = %q, want %q with no leftover tail", b, "new")
	}
}

func TestPlanWriteCallRelativePathResolvesAgainstCwd(t *testing.T) {
	dir := t.TempDir()
	other := filepath.Join(dir, "other")
	if err := os.Mkdir(other, 0o755); err != nil {
		t.Fatalf("mkdir cwd: %v", err)
	}
	t.Chdir(other)

	plan := planWriteCall(writeCall(`{"path":"x.txt","content":"rel"}`), nil, session{})
	if res := runFilePlan(plan); res.failed {
		t.Fatalf("plan reported an error, want a write under the cwd: %s", res.text)
	}
	want := filepath.Join(other, "x.txt")
	if plan.target != want {
		t.Errorf("plan.target = %q, want %q", plan.target, want)
	}
	if _, err := os.Stat(want); err != nil {
		t.Errorf("file did not land at %q: %v", want, err)
	}
	if _, err := os.Stat(filepath.Join(dir, "x.txt")); err == nil {
		t.Errorf("a file landed in %q as well, want only the cwd written", dir)
	}
}

func TestPlanWriteCallAbsolutePathIsUsedVerbatim(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	outside := filepath.Join(t.TempDir(), "elsewhere.txt")
	plan := planWriteCall(writeCall(`{"path":"`+outside+`","content":"abs"}`), nil, session{})
	if res := runFilePlan(plan); res.failed {
		t.Fatalf("plan reported an error, want the absolute path written: %s", res.text)
	}
	if plan.target != outside {
		t.Errorf("plan.target = %q, want %q", plan.target, outside)
	}
	b, err := os.ReadFile(outside)
	if err != nil {
		t.Fatalf("read created file: %v", err)
	}
	if string(b) != "abs" {
		t.Errorf("file content = %q, want %q", b, "abs")
	}
}

func TestPlanWriteCallUnwriteableTarget(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	blocker := filepath.Join(dir, "blocker")
	if err := os.WriteFile(blocker, []byte("not a directory"), 0o644); err != nil {
		t.Fatalf("pre-write blocker: %v", err)
	}
	plan := planWriteCall(writeCall(`{"path":"blocker/child.txt","content":"nope"}`), nil, session{})
	res := runFilePlan(plan)
	if !strings.HasPrefix(res.text, "error:") {
		t.Errorf("result = %q, want an error: prefix", res.text)
	}
	if !res.failed {
		t.Errorf("is_error = false for %q, want true", res.text)
	}
	if _, err := os.Stat(filepath.Join(blocker, "child.txt")); err == nil {
		t.Error("child.txt exists, want no write through a parent that is a file")
	}
}

func TestPlanWriteCallUnparseableArguments(t *testing.T) {
	plan := planWriteCall(writeCall(`{"path":`), nil, session{})
	res := runFilePlan(plan)
	if !strings.HasPrefix(res.text, "error: cannot parse tool arguments") {
		t.Errorf("result = %q, want the shared unparseable-arguments error", res.text)
	}
	if !res.failed {
		t.Errorf("is_error = false for %q, want true", res.text)
	}
	if plan.target != "" {
		t.Errorf("plan.target = %q, want the empty target an unparseable call reports", plan.target)
	}
}

func TestPlanEditCallReplacesUniqueMatch(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	path := filepath.Join(dir, "a.txt")
	if err := os.WriteFile(path, []byte("alpha\nbeta\nalpha-tail\n"), 0o644); err != nil {
		t.Fatalf("pre-write: %v", err)
	}

	plan := planEditCall(editCall(`{"path":"a.txt","old_string":"beta","new_string":"BETA!"}`), nil, session{})
	want := filepath.Join(dir, "a.txt")
	if plan.target != want {
		t.Errorf("plan.target = %q, want %q", plan.target, want)
	}
	res := runFilePlan(plan)
	if res.failed {
		t.Fatalf("is_error = true for a unique match: %q", res.text)
	}

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read edited file: %v", err)
	}
	if got := string(b); got != "alpha\nBETA!\nalpha-tail\n" {
		t.Errorf("file content = %q, want %q", got, "alpha\nBETA!\nalpha-tail\n")
	}
}

func TestPlanEditCallOldStringNotFound(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	path := filepath.Join(dir, "a.txt")
	body := "alpha\nbeta\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("pre-write: %v", err)
	}

	plan := planEditCall(editCall(`{"path":"a.txt","old_string":"gamma","new_string":"GAMMA"}`), nil, session{})
	res := runFilePlan(plan)
	if !strings.Contains(res.text, "not found") {
		t.Errorf("result = %q, want it to say the old string was not found", res.text)
	}
	if !res.failed {
		t.Errorf("is_error = false for %q, want true", res.text)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file after a failed edit: %v", err)
	}
	if string(b) != body {
		t.Errorf("file content = %q, want the untouched %q", b, body)
	}
}

// TestEditFileToolStatesFailureWhenOldStringNotFound pins the miss as a stated
// verdict: the edit read the file and wrote nothing, and only that branch can say
// so without reading its own sentence for a prefix.
func TestEditFileToolStatesFailureWhenOldStringNotFound(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "a.txt")
	if err := os.WriteFile(path, []byte("alpha\nbeta\n"), 0o644); err != nil {
		t.Fatalf("pre-write: %v", err)
	}

	res := editFileTool(path, "gamma", "GAMMA", false)
	if want := "error: old_string not found in " + path; res.text != want {
		t.Errorf("result = %q, want %q", res.text, want)
	}
	if !res.failed {
		t.Errorf("failed = false for %q, want the miss to say so", res.text)
	}
}

func TestPlanEditCallAmbiguousWithoutReplaceAll(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	path := filepath.Join(dir, "a.txt")
	body := "tick\ntock\ntick\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("pre-write: %v", err)
	}

	plan := planEditCall(editCall(`{"path":"a.txt","old_string":"tick","new_string":"tock"}`), nil, session{})
	res := runFilePlan(plan)
	if !strings.Contains(res.text, "2 matches") {
		t.Errorf("result = %q, want it to report 2 matches", res.text)
	}
	if !res.failed {
		t.Errorf("is_error = false for %q, want true", res.text)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file after an ambiguous edit: %v", err)
	}
	if string(b) != body {
		t.Errorf("file content = %q, want the untouched %q", b, body)
	}
}

// TestEditFileToolStatesFailureOnAmbiguousMatchesWithoutReplaceAll is the
// judgement branch: the file was read, the replacement was declined, and the text
// naming the count is only as trustworthy as the verdict stated beside it.
func TestEditFileToolStatesFailureOnAmbiguousMatchesWithoutReplaceAll(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "a.txt")
	if err := os.WriteFile(path, []byte("tick\ntock\ntick\n"), 0o644); err != nil {
		t.Fatalf("pre-write: %v", err)
	}

	res := editFileTool(path, "tick", "tock", false)
	if want := "error: 2 matches of old_string in " + path + "; set replace_all to replace every match"; res.text != want {
		t.Errorf("result = %q, want %q", res.text, want)
	}
	if !res.failed {
		t.Errorf("failed = false for %q, want the declined edit to say so", res.text)
	}
}

func TestPlanEditCallReplaceAll(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	path := filepath.Join(dir, "a.txt")
	if err := os.WriteFile(path, []byte("tick\ntock\ntick\n"), 0o644); err != nil {
		t.Fatalf("pre-write: %v", err)
	}

	plan := planEditCall(editCall(`{"path":"a.txt","old_string":"tick","new_string":"tock","replace_all":true}`), nil, session{})
	res := runFilePlan(plan)
	if res.failed {
		t.Fatalf("is_error = true for replace_all: %q", res.text)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read edited file: %v", err)
	}
	if got := string(b); got != "tock\ntock\ntock\n" {
		t.Errorf("file content = %q, want every occurrence replaced", got)
	}
}

// TestEditFileToolStatesFailureFromAnUnreadableFile covers the branch that fails
// before it has anything to count: the path is a directory, so there is no body to
// inspect and the read refusal is the whole verdict.
func TestEditFileToolStatesFailureFromAnUnreadableFile(t *testing.T) {
	dir := t.TempDir()

	res := editFileTool(dir, "x", "y", false)
	if !strings.HasPrefix(res.text, "error: cannot read "+dir) {
		t.Errorf("result = %q, want the cannot-read wording the Model already sees", res.text)
	}
	if !res.failed {
		t.Errorf("failed = false for %q, want the unreadable file to say so", res.text)
	}
}

// TestEditFileToolStaysCleanOnACompletedEdit is the other half of the stated
// verdict: the two success wordings carry no "error:" prefix today, and a clean
// result must stay clean for reasons that have nothing to do with how they read.
func TestEditFileToolStaysCleanOnACompletedEdit(t *testing.T) {
	t.Run("unique match", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "a.txt")
		if err := os.WriteFile(path, []byte("alpha\n"), 0o644); err != nil {
			t.Fatalf("pre-write: %v", err)
		}
		res := editFileTool(path, "alpha", "BETA", false)
		if res.failed {
			t.Errorf("failed = true for a completed edit: %q", res.text)
		}
		if want := "edited " + path + ": replaced 1 match"; res.text != want {
			t.Errorf("result = %q, want %q", res.text, want)
		}
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read edited file: %v", err)
		}
		if string(body) != "BETA\n" {
			t.Errorf("file content = %q, want %q", body, "BETA\n")
		}
	})

	t.Run("replace_all", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "a.txt")
		if err := os.WriteFile(path, []byte("tick\ntock\ntick\n"), 0o644); err != nil {
			t.Fatalf("pre-write: %v", err)
		}
		res := editFileTool(path, "tick", "tock", true)
		if res.failed {
			t.Errorf("failed = true for a completed replace_all: %q", res.text)
		}
		if want := "edited " + path + ": replaced 2 matches"; res.text != want {
			t.Errorf("result = %q, want %q", res.text, want)
		}
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read edited file: %v", err)
		}
		if string(body) != "tock\ntock\ntock\n" {
			t.Errorf("file content = %q, want every occurrence replaced", body)
		}
	})
}

func TestPlanEditCallReplaceAllOnUniqueMatch(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	path := filepath.Join(dir, "a.txt")
	if err := os.WriteFile(path, []byte("only one here\n"), 0o644); err != nil {
		t.Fatalf("pre-write: %v", err)
	}

	plan := planEditCall(editCall(`{"path":"a.txt","old_string":"one","new_string":"two","replace_all":true}`), nil, session{})
	res := runFilePlan(plan)
	if res.failed {
		t.Fatalf("replace_all refused a unique match: %q", res.text)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read edited file: %v", err)
	}
	if got := string(b); got != "only two here\n" {
		t.Errorf("file content = %q, want %q", got, "only two here\n")
	}
}

func TestPlanEditCallRelativePathResolvesAgainstCwd(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("rel"), 0o644); err != nil {
		t.Fatalf("pre-write: %v", err)
	}
	plan := planEditCall(editCall(`{"path":"a.txt","old_string":"rel","new_string":"done"}`), nil, session{})
	want := filepath.Join(dir, "a.txt")
	if plan.target != want {
		t.Errorf("plan.target = %q, want %q", plan.target, want)
	}
	if res := runFilePlan(plan); res.failed {
		t.Fatalf("plan reported an error, want the edit under the cwd: %s", res.text)
	}
	b, err := os.ReadFile(want)
	if err != nil {
		t.Fatalf("read edited file: %v", err)
	}
	if string(b) != "done" {
		t.Errorf("file content = %q, want %q", b, "done")
	}
}

func TestPlanEditCallMissingFile(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	plan := planEditCall(editCall(`{"path":"gone.txt","old_string":"x","new_string":"y"}`), nil, session{})
	res := runFilePlan(plan)
	if !strings.HasPrefix(res.text, "error:") {
		t.Errorf("result = %q, want an error: prefix", res.text)
	}
	if !strings.Contains(res.text, filepath.Join(dir, "gone.txt")) {
		t.Errorf("result = %q, want it to name the file it could not read", res.text)
	}
	if !res.failed {
		t.Errorf("is_error = false for %q, want true", res.text)
	}
}

func TestPlanEditCallEmptyOldString(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	path := filepath.Join(dir, "a.txt")
	body := "hundred characters of body text here, and more besides."
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("pre-write: %v", err)
	}

	plan := planEditCall(editCall(`{"path":"a.txt","old_string":"","new_string":"oops"}`), nil, session{})
	res := runFilePlan(plan)
	if !strings.HasPrefix(res.text, "error: old_string") {
		t.Errorf("result = %q, want an error naming the old_string argument", res.text)
	}
	if !res.failed {
		t.Errorf("is_error = false for %q, want true", res.text)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file after a refused edit: %v", err)
	}
	if string(b) != body {
		t.Errorf("file content = %q, want the untouched %q", b, body)
	}
}

func TestPlanEditCallMissingPath(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	plan := planEditCall(editCall(`{"old_string":"x","new_string":"y"}`), nil, session{})
	res := runFilePlan(plan)
	if !strings.HasPrefix(res.text, "error: path is required") {
		t.Errorf("result = %q, want the shared missing-path error", res.text)
	}
	if !res.failed {
		t.Errorf("is_error = false for %q, want true", res.text)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read temp dir: %v", err)
	}
	if len(entries) != 0 {
		t.Errorf("temp dir contains %v, want nothing written", entries)
	}
}

func TestPlanEditCallUnparseableArguments(t *testing.T) {
	plan := planEditCall(editCall(`{"path":`), nil, session{})
	res := runFilePlan(plan)
	if !strings.HasPrefix(res.text, "error: cannot parse tool arguments") {
		t.Errorf("result = %q, want the shared unparseable-arguments error", res.text)
	}
	if !res.failed {
		t.Errorf("is_error = false for %q, want true", res.text)
	}
}

// TestFileErrorPlanStatesItsOwnVerdict is why no file verdict may be read off text
// any more: a plan that never touches the filesystem already knows it failed, so
// the value it hands back states that whatever the sentence reads like — even
// prose the old prefix rule would have called clean.
func TestFileErrorPlanStatesItsOwnVerdict(t *testing.T) {
	plan := fileErrorPlan(nil, "error: %s", "path is required")
	if res := runFilePlan(plan); !res.failed {
		t.Errorf("failed = false for %q, want a plan that never ran to state failure", res.text)
	}
	if res := fileErrorPlan(nil, "a result the prefix rule would have read as clean").run(""); !res.failed {
		t.Error("failed = false for text the rule would have read as clean, want the plan's own stated failure")
	}
}

func TestFileToolDispatchesEdit(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("before"), 0o644); err != nil {
		t.Fatalf("pre-write: %v", err)
	}
	// Dispatch reads the run's resolved set, so the session has to carry one: the
	// top-level set over no roster is the set that holds edit and write.
	cfg := completeConfig("")
	cfg.tools = resolveTools(agentEntry{}, nil)
	s := session{cfg: cfg}
	plan := planToolCall(editCall(`{"path":"a.txt","old_string":"before","new_string":"after"}`), s)
	want := filepath.Join(dir, "a.txt")
	if plan.target != want {
		t.Errorf("plan.target = %q, want the resolved path %q", plan.target, want)
	}
	res := runFilePlan(plan)
	if res.failed {
		t.Fatalf("is_error = true for a dispatched edit: %q", res.text)
	}
	b, err := os.ReadFile(want)
	if err != nil {
		t.Fatalf("read edited file: %v", err)
	}
	if string(b) != "after" {
		t.Errorf("file content = %q, want %q", b, "after")
	}
}

func TestFileToolDispatchesWrite(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	cfg := completeConfig("")
	cfg.tools = resolveTools(agentEntry{}, nil)
	plan := planToolCall(
		writeCall(`{"path":"dispatched.txt","content":"via planToolCall"}`), session{cfg: cfg})
	want := filepath.Join(dir, "dispatched.txt")
	if plan.target != want {
		t.Errorf("plan.target = %q, want %q", plan.target, want)
	}
	res := runFilePlan(plan)
	if res.failed {
		t.Fatalf("is_error = true for a dispatched write: %q", res.text)
	}
	b, err := os.ReadFile(want)
	if err != nil {
		t.Fatalf("read created file: %v", err)
	}
	if string(b) != "via planToolCall" {
		t.Errorf("file content = %q, want %q", b, "via planToolCall")
	}
}

// TestGuardedWriteCallRefusesAndCreatesNoFile is the denied Write as the Model and
// the disk see it: the refusal is the Tool Result verbatim and states failure,
// and neither the file nor the directories leading to it exist because nothing
// ran. The chain is asked once for the whole call — dispatch asks from the
// presentation the plan declares, so a Tool that also asked inside its own run
// would fork the hook script twice for one call. What the chain was handed is
// read off the hook's own stdin rather than off the plan's fields: a declaration
// only matters as the payload a host rule gets to rule on.
func TestGuardedWriteCallRefusesAndCreatesNoFile(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	asked := filepath.Join(dir, "asks.log")
	stdin := filepath.Join(dir, "ask-stdin.json")
	script := writeGuardHookScript(t, "deny.sh",
		`cat > `+stdin+`
printf x >> `+asked+`
echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"credential path"}}'`)

	cfg := completeConfig("")
	cfg.hooks = chainOf(guardHook{path: script})
	s := session{cfg: cfg}
	plan := planWriteCall(writeCall(`{"path":"out/secret.txt","content":"hunter2"}`), nil, s)

	res := s.guardRun(plan)

	wantPath := filepath.Join(dir, "out", "secret.txt")
	want := "error: guard hook " + script + " denied this command: credential path"
	if res.text != want {
		t.Errorf("Tool Result = %q, want the refusal verbatim %q", res.text, want)
	}
	if !res.failed {
		t.Errorf("is_error = false for %q, want a refusal that never reached the disk to state failure", res.text)
	}
	if _, err := os.Stat(wantPath); !os.IsNotExist(err) {
		t.Errorf("denied write created out/secret.txt: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "out")); err == nil {
		t.Errorf("denied write created the parent directory out anyway")
	}
	if got := hookAskCount(t, asked); got != 1 {
		t.Errorf("chain was asked %d times, want exactly once for one call", got)
	}
	handed, err := os.ReadFile(stdin)
	if err != nil {
		t.Fatalf("read the deny hook's stdin: %v", err)
	}
	if wantPayload := `{"tool_name":"Read","tool_input":{"file_path":"` + wantPath + `"}}`; string(handed) != wantPayload {
		t.Errorf("Tool Call was presented to the hook as %q, want the Read shape of %s", handed, wantPath)
	}
}

// TestGuardedEditCallRefusesAndLeavesFileUntouched is the same judgement on the
// Tool that rewrites: the denial arrives before the file is read, so the bytes on
// disk are the ones that were there before the call was made. The Edit reaches the
// hook under the Read shape too, which is what lets one host rule cover every
// file Tool.
func TestGuardedEditCallRefusesAndLeavesFileUntouched(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	target := filepath.Join(dir, "notes.txt")
	if err := os.WriteFile(target, []byte("hunter2"), 0o644); err != nil {
		t.Fatalf("seed target file: %v", err)
	}
	asked := filepath.Join(dir, "asks.log")
	stdin := filepath.Join(dir, "ask-stdin.json")
	script := writeGuardHookScript(t, "deny.sh",
		`cat > `+stdin+`
printf x >> `+asked+`
echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"credential path"}}'`)

	cfg := completeConfig("")
	cfg.hooks = chainOf(guardHook{path: script})
	s := session{cfg: cfg}
	plan := planEditCall(editCall(`{"path":"notes.txt","old_string":"hunter2","new_string":"redacted"}`), nil, s)

	res := s.guardRun(plan)

	if want := "error: guard hook " + script + " denied this command: credential path"; res.text != want {
		t.Errorf("Tool Result = %q, want the refusal verbatim %q", res.text, want)
	}
	if !res.failed {
		t.Errorf("is_error = false for %q, want a refusal that never touched the file to state failure", res.text)
	}
	body, err := os.ReadFile(target)
	if err != nil {
		t.Fatalf("read target file: %v", err)
	}
	if string(body) != "hunter2" {
		t.Errorf("file content = %q, want the denied edit to leave %q untouched", body, "hunter2")
	}
	if got := hookAskCount(t, asked); got != 1 {
		t.Errorf("chain was asked %d times, want exactly once for one call", got)
	}
	handed, err := os.ReadFile(stdin)
	if err != nil {
		t.Fatalf("read the deny hook's stdin: %v", err)
	}
	if wantPayload := `{"tool_name":"Read","tool_input":{"file_path":"` + target + `"}}`; string(handed) != wantPayload {
		t.Errorf("Tool Call was presented to the hook as %q, want the Read shape of %s", handed, target)
	}
}

// TestFileCallWithoutAPathAsksNoHook is the other half of the declaration: a
// Write or Edit that never resolved a path has no subject for a host rule to
// rule on, so even a chain that denies everything is never asked about it and the
// plan's own failure is the Tool Result.
func TestFileCallWithoutAPathAsksNoHook(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	asked := filepath.Join(dir, "asks.log")
	script := writeGuardHookScript(t, "deny.sh",
		`printf x >> `+asked+`
echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"credential path"}}'`)

	cfg := completeConfig("")
	cfg.tools = resolveTools(agentEntry{}, nil)
	cfg.hooks = chainOf(guardHook{path: script})
	s := session{cfg: cfg}

	for _, tc := range []struct {
		name string
		call toolCall
	}{
		{"write with no path", writeCall(`{"content":"orphan"}`)},
		{"write with unparseable arguments", writeCall(`{`)},
		{"edit with no path", editCall(`{"old_string":"a","new_string":"b"}`)},
		{"edit with unparseable arguments", editCall(`{`)},
	} {
		res := s.guardRun(planToolCall(tc.call, s))
		if !res.failed || !strings.HasPrefix(res.text, "error:") {
			t.Errorf("%s: Tool Result = (%q, failed=%v), want the plan's own error", tc.name, res.text, res.failed)
		}
		if got := hookAskCount(t, asked); got != 0 {
			t.Errorf("%s: chain was asked %d times, want no hook asked for a call with no subject", tc.name, got)
		}
	}
}

// hookAskCount is how many times the deny script ran, counted from the byte it
// appends per invocation.
func hookAskCount(t *testing.T, path string) int {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		t.Fatalf("read hook ask log: %v", err)
	}
	return len(b)
}
