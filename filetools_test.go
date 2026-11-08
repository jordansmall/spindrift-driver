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
