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

func TestPlanWriteCallCreatesFile(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	plan := planWriteCall(writeCall(`{"path":"out.txt","content":"hello, world"}`), nil, session{})
	if out := plan.run(); plan.isError(out) {
		t.Fatalf("plan reported an error, want a created file: %s", out)
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

func TestPlanWriteCallMissingPath(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	plan := planWriteCall(writeCall(`{"content":"orphan"}`), nil, session{})
	out := plan.run()
	if !strings.HasPrefix(out, "error:") {
		t.Errorf("result = %q, want an error: prefix", out)
	}
	if !strings.Contains(out, "path") {
		t.Errorf("result = %q, want it to name the missing path argument", out)
	}
	if !plan.isError(out) {
		t.Errorf("is_error = false for %q, want true", out)
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
	out := plan.run()
	if plan.isError(out) {
		t.Fatalf("is_error = true for a successful write: %q", out)
	}
	want := filepath.Join(dir, "a", "b.txt")
	if !strings.Contains(out, "5") || !strings.Contains(out, want) {
		t.Errorf("result = %q, want it to name the byte count and %q", out, want)
	}
}

func TestPlanWriteCallCreatesMissingParents(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	plan := planWriteCall(writeCall(`{"path":"nested/deep/file.txt","content":"deep"}`), nil, session{})
	if out := plan.run(); plan.isError(out) {
		t.Fatalf("plan reported an error, want the parents created: %s", out)
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
	if out := plan.run(); plan.isError(out) {
		t.Fatalf("plan reported an error, want an overwrite: %s", out)
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
	if out := plan.run(); plan.isError(out) {
		t.Fatalf("plan reported an error, want a write under the cwd: %s", out)
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
	if res := plan.run(); plan.isError(res) {
		t.Fatalf("plan reported an error, want the absolute path written: %s", res)
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
	out := plan.run()
	if !strings.HasPrefix(out, "error:") {
		t.Errorf("result = %q, want an error: prefix", out)
	}
	if !plan.isError(out) {
		t.Errorf("is_error = false for %q, want true", out)
	}
	if _, err := os.Stat(filepath.Join(blocker, "child.txt")); err == nil {
		t.Error("child.txt exists, want no write through a parent that is a file")
	}
}

func TestPlanWriteCallUnparseableArguments(t *testing.T) {
	plan := planWriteCall(writeCall(`{"path":`), nil, session{})
	out := plan.run()
	if !strings.HasPrefix(out, "error: cannot parse tool arguments") {
		t.Errorf("result = %q, want the shared unparseable-arguments error", out)
	}
	if !plan.isError(out) {
		t.Errorf("is_error = false for %q, want true", out)
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
	out := plan.run()
	if plan.isError(out) {
		t.Fatalf("is_error = true for a unique match: %q", out)
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
	out := plan.run()
	if !strings.Contains(out, "not found") {
		t.Errorf("result = %q, want it to say the old string was not found", out)
	}
	if !plan.isError(out) {
		t.Errorf("is_error = false for %q, want true", out)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file after a failed edit: %v", err)
	}
	if string(b) != body {
		t.Errorf("file content = %q, want the untouched %q", b, body)
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
	out := plan.run()
	if !strings.Contains(out, "2 matches") {
		t.Errorf("result = %q, want it to report 2 matches", out)
	}
	if !plan.isError(out) {
		t.Errorf("is_error = false for %q, want true", out)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file after an ambiguous edit: %v", err)
	}
	if string(b) != body {
		t.Errorf("file content = %q, want the untouched %q", b, body)
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
	out := plan.run()
	if plan.isError(out) {
		t.Fatalf("is_error = true for replace_all: %q", out)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read edited file: %v", err)
	}
	if got := string(b); got != "tock\ntock\ntock\n" {
		t.Errorf("file content = %q, want every occurrence replaced", got)
	}
}

func TestPlanEditCallReplaceAllOnUniqueMatch(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	path := filepath.Join(dir, "a.txt")
	if err := os.WriteFile(path, []byte("only one here\n"), 0o644); err != nil {
		t.Fatalf("pre-write: %v", err)
	}

	plan := planEditCall(editCall(`{"path":"a.txt","old_string":"one","new_string":"two","replace_all":true}`), nil, session{})
	out := plan.run()
	if plan.isError(out) {
		t.Fatalf("replace_all refused a unique match: %q", out)
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
	if out := plan.run(); plan.isError(out) {
		t.Fatalf("plan reported an error, want the edit under the cwd: %s", out)
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
	out := plan.run()
	if !strings.HasPrefix(out, "error:") {
		t.Errorf("result = %q, want an error: prefix", out)
	}
	if !strings.Contains(out, filepath.Join(dir, "gone.txt")) {
		t.Errorf("result = %q, want it to name the file it could not read", out)
	}
	if !plan.isError(out) {
		t.Errorf("is_error = false for %q, want true", out)
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
	out := plan.run()
	if !strings.HasPrefix(out, "error: old_string") {
		t.Errorf("result = %q, want an error naming the old_string argument", out)
	}
	if !plan.isError(out) {
		t.Errorf("is_error = false for %q, want true", out)
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
	out := plan.run()
	if !strings.HasPrefix(out, "error: path is required") {
		t.Errorf("result = %q, want the shared missing-path error", out)
	}
	if !plan.isError(out) {
		t.Errorf("is_error = false for %q, want true", out)
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
	out := plan.run()
	if !strings.HasPrefix(out, "error: cannot parse tool arguments") {
		t.Errorf("result = %q, want the shared unparseable-arguments error", out)
	}
	if !plan.isError(out) {
		t.Errorf("is_error = false for %q, want true", out)
	}
}

func TestFileToolDispatchesEdit(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("before"), 0o644); err != nil {
		t.Fatalf("pre-write: %v", err)
	}
	plan := planToolCall(editCall(`{"path":"a.txt","old_string":"before","new_string":"after"}`), session{})
	want := filepath.Join(dir, "a.txt")
	if plan.target != want {
		t.Errorf("plan.target = %q, want the resolved path %q", plan.target, want)
	}
	out := plan.run()
	if plan.isError(out) {
		t.Fatalf("is_error = true for a dispatched edit: %q", out)
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

	plan := planToolCall(writeCall(`{"path":"dispatched.txt","content":"via planToolCall"}`), session{})
	want := filepath.Join(dir, "dispatched.txt")
	if plan.target != want {
		t.Errorf("plan.target = %q, want %q", plan.target, want)
	}
	out := plan.run()
	if plan.isError(out) {
		t.Fatalf("is_error = true for a dispatched write: %q", out)
	}
	b, err := os.ReadFile(want)
	if err != nil {
		t.Fatalf("read created file: %v", err)
	}
	if string(b) != "via planToolCall" {
		t.Errorf("file content = %q, want %q", b, "via planToolCall")
	}
}

func TestPlanWriteCallGuardHookDenies(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	script := writeGuardHookScript(t, "deny.sh",
		`echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"credential path"}}'`)

	plan := planWriteCall(writeCall(`{"path":"secret.txt","content":"hunter2"}`), nil,
		session{cfg: config{hooks: []guardHook{{path: script}}}})
	out := plan.run()

	want := "error: guard hook " + script + " denied this command: credential path"
	if out != want {
		t.Errorf("plan.run() = %q, want %q", out, want)
	}
	if !plan.isError(out) {
		t.Errorf("is_error = false for %q, want true", out)
	}
	if _, err := os.Stat(filepath.Join(dir, "secret.txt")); !os.IsNotExist(err) {
		t.Errorf("denied write created secret.txt: %v", err)
	}
}

func TestPlanEditCallGuardHookDenies(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	target := filepath.Join(dir, "notes.txt")
	if err := os.WriteFile(target, []byte("hunter2"), 0o644); err != nil {
		t.Fatalf("seed target file: %v", err)
	}
	script := writeGuardHookScript(t, "deny.sh",
		`echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"credential path"}}'`)

	plan := planEditCall(editCall(`{"path":"notes.txt","old_string":"hunter2","new_string":"redacted"}`), nil,
		session{cfg: config{hooks: []guardHook{{path: script}}}})
	out := plan.run()

	if want := "error: guard hook " + script + " denied this command: credential path"; out != want {
		t.Errorf("plan.run() = %q, want %q", out, want)
	}
	body, err := os.ReadFile(target)
	if err != nil {
		t.Fatalf("read target file: %v", err)
	}
	if string(body) != "hunter2" {
		t.Errorf("file content = %q, want the denied edit to leave %q untouched", body, "hunter2")
	}
}
