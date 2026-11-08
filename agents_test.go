package main

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// writeAgentFile puts contents in a named file inside dir and returns the path,
// the one thing both parseAgentFile and loadAgentDir are given.
func writeAgentFile(t *testing.T, dir, name, contents string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("write agent file %s: %v", path, err)
	}
	return path
}

func TestLoadAgentDirOrdersEntriesByFileName(t *testing.T) {
	// Written in scrambled order so the result can only come out sorted if the
	// loader sorts, and not because of the order the test happened to write in.
	dir := t.TempDir()
	for _, name := range []string{"zeta.md", "alpha.md", "mid.md"} {
		writeAgentFile(t, dir, name, "---\nname: "+name[:len(name)-3]+"\nmodel: m-"+name+"\n---\nbody\n")
	}

	entries, err := loadAgentDir(dir)
	if err != nil {
		t.Fatalf("loadAgentDir(%q) error = %v, want nil", dir, err)
	}
	want := []string{"alpha", "mid", "zeta"}
	if len(entries) != len(want) {
		t.Fatalf("len(entries) = %d, want %d (%+v)", len(entries), len(want), entries)
	}
	for i, name := range want {
		if entries[i].Name != name {
			t.Errorf("entries[%d].Name = %q, want %q", i, entries[i].Name, name)
		}
	}
}

func TestLoadAgentDirSkipsNonMarkdown(t *testing.T) {
	dir := t.TempDir()
	writeAgentFile(t, dir, "real.md", "---\nname: real\nmodel: m\n---\nbody\n")
	writeAgentFile(t, dir, "README.txt", "not an agent\n")
	writeAgentFile(t, dir, "notes.md.bak", "---\nname: bak\nmodel: m\n---\nbody\n")
	// a directory whose name ends in .md is not an Agent File either
	if err := os.Mkdir(filepath.Join(dir, "nested.md"), 0o755); err != nil {
		t.Fatalf("mkdir nested.md: %v", err)
	}

	entries, err := loadAgentDir(dir)
	if err != nil {
		t.Fatalf("loadAgentDir(%q) error = %v, want nil", dir, err)
	}
	if len(entries) != 1 || entries[0].Name != "real" {
		t.Fatalf("entries = %+v, want exactly the one .md entry", entries)
	}
}

func TestParseAgentFileSingleEntry(t *testing.T) {
	path := writeAgentFile(t, t.TempDir(), "reviewer.md", `---
name: reviewer
description: reads the diff and reports blockers
model: gpt-4o-mini
---
You review diffs. Be brief.
`)

	entry, err := parseAgentFile(path)
	if err != nil {
		t.Fatalf("parseAgentFile(%q) error = %v, want nil", path, err)
	}
	if entry.Name != "reviewer" {
		t.Errorf("Name = %q, want %q", entry.Name, "reviewer")
	}
	if entry.Description != "reads the diff and reports blockers" {
		t.Errorf("Description = %q, want %q", entry.Description, "reads the diff and reports blockers")
	}
	if entry.Model != "gpt-4o-mini" {
		t.Errorf("Model = %q, want %q", entry.Model, "gpt-4o-mini")
	}
	if entry.SystemPrompt != "You review diffs. Be brief." {
		t.Errorf("SystemPrompt = %q, want %q", entry.SystemPrompt, "You review diffs. Be brief.")
	}
	if len(entry.Tools) != 0 || len(entry.Spawns) != 0 {
		t.Errorf("Tools = %v, Spawns = %v, want both empty when the file says nothing", entry.Tools, entry.Spawns)
	}
}

func TestParseAgentFileListFields(t *testing.T) {
	for _, tc := range []struct {
		name       string
		contents   string
		wantTools  []string
		wantSpawns []string
	}{
		{
			name:       "inline comma",
			contents:   "---\nname: a\nmodel: m\ntools: bash, read\nspawns: scout, worker\n---\nbody\n",
			wantTools:  []string{"bash", "read"},
			wantSpawns: []string{"scout", "worker"},
		},
		{
			name:       "block list",
			contents:   "---\nname: a\nmodel: m\ntools:\n- bash\n- read\nspawns:\n- scout\n---\nbody\n",
			wantTools:  []string{"bash", "read"},
			wantSpawns: []string{"scout"},
		},
		{
			name:       "one of each form in one file",
			contents:   "---\nname: a\nmodel: m\ntools: bash\nspawns:\n- scout\n- worker\n---\nbody\n",
			wantTools:  []string{"bash"},
			wantSpawns: []string{"scout", "worker"},
		},
		{
			name:       "single inline item",
			contents:   "---\nname: a\nmodel: m\ntools: read\n---\nbody\n",
			wantTools:  []string{"read"},
			wantSpawns: nil,
		},
		{
			name:       "empty value reads as no items",
			contents:   "---\nname: a\nmodel: m\ntools:\ndescription: just a scalar after it\n---\nbody\n",
			wantTools:  nil,
			wantSpawns: nil,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeAgentFile(t, t.TempDir(), "a.md", tc.contents)

			entry, err := parseAgentFile(path)
			if err != nil {
				t.Fatalf("parseAgentFile(%q) error = %v, want nil", path, err)
			}
			if !reflect.DeepEqual(entry.Tools, tc.wantTools) {
				t.Errorf("Tools = %v, want %v", entry.Tools, tc.wantTools)
			}
			if !reflect.DeepEqual(entry.Spawns, tc.wantSpawns) {
				t.Errorf("Spawns = %v, want %v", entry.Spawns, tc.wantSpawns)
			}
			if entry.SystemPrompt != "body" {
				t.Errorf("SystemPrompt = %q, want %q", entry.SystemPrompt, "body")
			}
		})
	}
}

func TestParseAgentFileMissingRequiredField(t *testing.T) {
	for _, tc := range []struct{ name, contents, want string }{
		{"no name", "---\nmodel: gpt-4o-mini\n---\nbody\n", "name"},
		{"name with no value", "---\nname:\nmodel: gpt-4o-mini\n---\nbody\n", "name"},
		{"no model", "---\nname: reviewer\n---\nbody\n", "model"},
		{"model with no value", "---\nname: reviewer\nmodel: \n---\nbody\n", "model"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeAgentFile(t, t.TempDir(), "a.md", tc.contents)

			entry, err := parseAgentFile(path)
			assertErrMessage(t, err, path, "is required", tc.want)
			if !reflect.DeepEqual(entry, agentEntry{}) {
				t.Errorf("entry = %+v, want zero value on error", entry)
			}
		})
	}
}

func TestParseAgentFileFrontmatterFence(t *testing.T) {
	for _, tc := range []struct{ name, contents, want string }{
		{"unclosed fence", "---\nname: a\nmodel: m\nbody without a closing fence\n", "unclosed"},
		{"no fence at all", "name: a\nmodel: m\njust prose\n", "missing opening"},
		{"empty file", "", "missing opening"},
		{"leading blank line is not a fence", "\n---\nname: a\nmodel: m\n---\nbody\n", "missing opening"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeAgentFile(t, t.TempDir(), "a.md", tc.contents)

			_, err := parseAgentFile(path)
			assertErrMessage(t, err, path, "frontmatter")
			if errors.Is(err, errNoAgentDir) {
				t.Errorf("errors.Is(err, errNoAgentDir) = true, want false; err = %v", err)
			}
		})
	}
}

func TestParseAgentFileUnknownField(t *testing.T) {
	for _, tc := range []struct{ name, contents, want string }{
		{"stray key", "---\nname: a\nmodel: m\ncolor: blue\n---\nbody\n", "color"},
		{"near-miss typo", "---\nname: a\nmodel: m\ndecription: nope\n---\nbody\n", "decription"},
		{"temperature is not an agent field", "---\nname: a\nmodel: m\ntemperature: 0.2\n---\nbody\n", "temperature"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeAgentFile(t, t.TempDir(), "a.md", tc.contents)

			_, err := parseAgentFile(path)
			assertErrMessage(t, err, path, "unknown field", tc.want)
		})
	}
}

func TestParseAgentFileFieldGivenTwice(t *testing.T) {
	// A key the file does not know is fatal, so a key that overwrites one of its
	// own kind is the same mistake in the other direction: last-one-wins would
	// run the model the operator did not mean to name.
	for _, tc := range []struct{ name, contents, want string }{
		{"model twice", "---\nname: a\nmodel: m\nmodel: m2\n---\nbody\n", `"model"`},
		{"tools twice", "---\nname: a\nmodel: m\ntools: bash\ntools: read\n---\nbody\n", `"tools"`},
		{"name twice", "---\nname: a\nname: b\nmodel: m\n---\nbody\n", `"name"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeAgentFile(t, t.TempDir(), "a.md", tc.contents)

			_, err := parseAgentFile(path)
			assertErrMessage(t, err, path, "given twice", tc.want)
		})
	}
}

func TestParseAgentFileUnknownFieldReportsFirstInFileOrder(t *testing.T) {
	// Lines are scanned in file order, so the same broken file always names the
	// same key instead of complaining about whichever one came last.
	path := writeAgentFile(t, t.TempDir(), "a.md", "---\nname: a\nzulu: 1\nalpha: 2\nmodel: m\n---\nbody\n")

	_, err := parseAgentFile(path)
	assertErrMessage(t, err, path, "unknown field", `"zulu"`)
	if strings.Contains(err.Error(), "alpha") {
		t.Errorf("error %q names a key other than the first in file order", err.Error())
	}
}

func TestParseAgentFileMalformedFieldLine(t *testing.T) {
	for _, tc := range []struct{ name, contents, want string }{
		{"line with no colon", "---\nname: a\nmodel m\n---\nbody\n", "malformed frontmatter line"},
		{"list items under a scalar", "---\nname:\n- a\nmodel: m\n---\nbody\n", `field "name" does not accept a list`},
		{"list item before any field", "---\n- a\nname: b\nmodel: m\n---\nbody\n", "list item with no field"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeAgentFile(t, t.TempDir(), "a.md", tc.contents)

			_, err := parseAgentFile(path)
			assertErrMessage(t, err, path, tc.want)
		})
	}
}

func TestLoadAgentDirMalformedFileIsNamed(t *testing.T) {
	dir := t.TempDir()
	writeAgentFile(t, dir, "good.md", "---\nname: good\nmodel: m\n---\nbody\n")
	bad := writeAgentFile(t, dir, "bad.md", "---\nname: bad\n---\nbody\n")

	entries, err := loadAgentDir(dir)
	assertErrMessage(t, err, bad, "is required", "model")
	if strings.Contains(err.Error(), "good.md") {
		t.Errorf("error %q names a file that is not the broken one", err.Error())
	}
	if entries != nil {
		t.Errorf("entries = %+v, want nil on error", entries)
	}
	if errors.Is(err, errNoAgentDir) {
		t.Errorf("errors.Is(err, errNoAgentDir) = true, want false; err = %v", err)
	}
}

func TestLoadAgentDirDuplicateNameNamesBothFiles(t *testing.T) {
	// findAgent answers with the first entry it matches, so a roster that lets
	// two files claim one name would run whichever sorts first and never say so.
	dir := t.TempDir()
	first := writeAgentFile(t, dir, "a-reviewer.md", "---\nname: reviewer\nmodel: m-a\n---\nbody\n")
	second := writeAgentFile(t, dir, "b-reviewer.md", "---\nname: reviewer\nmodel: m-b\n---\nbody\n")

	entries, err := loadAgentDir(dir)
	assertErrMessage(t, err, first, `"reviewer"`, "agent dir", "defined by both")
	if !strings.Contains(err.Error(), second) {
		t.Errorf("error %q does not name the second file %q", err.Error(), second)
	}
	if entries != nil {
		t.Errorf("entries = %+v, want nil on error", entries)
	}
	if errors.Is(err, errNoAgentDir) {
		t.Errorf("errors.Is(err, errNoAgentDir) = true, want false; err = %v", err)
	}
}

// TestAgentsDirPath: the Agent Files directory is home-relative and resolved
// through the injected lookupEnv only — os.UserHomeDir would bypass this seam and
// leak a developer's own ~/.config into whatever run a test drives.
func TestAgentsDirPath(t *testing.T) {
	if filepath.IsAbs(defaultAgentsDirPath) {
		t.Errorf("defaultAgentsDirPath %q is absolute, want home-relative", defaultAgentsDirPath)
	}
	if strings.HasPrefix(defaultAgentsDirPath, string(os.PathSeparator)) {
		t.Errorf("defaultAgentsDirPath %q is cwd-relative, want home-relative", defaultAgentsDirPath)
	}
	for _, tc := range []struct {
		name      string
		lookupEnv func(string) (string, bool)
		want      string
	}{
		{
			// Spelled out rather than built from defaultAgentsDirPath, so
			// renaming the default has to fail here.
			name: "HOME only",
			lookupEnv: func(name string) (string, bool) {
				if name == "HOME" {
					return "/home/dev", true
				}
				return "", false
			},
			want: "/home/dev/.config/spindrift-driver/agents",
		},
		{
			name:      "no HOME is no Agent Files, not an error",
			lookupEnv: func(string) (string, bool) { return "", false },
			want:      "",
		},
		{
			name: "set-but-empty HOME reads as unset",
			lookupEnv: func(name string) (string, bool) {
				if name == "HOME" {
					return "", true
				}
				return "", false
			},
			want: "",
		},
		{
			// The home-relative spelling is joined, not expanded: nothing here
			// knows what a leading ~ means, so a literal home stays literal.
			name: "tilde is not expanded",
			lookupEnv: func(name string) (string, bool) {
				if name == "HOME" {
					return "~/dev", true
				}
				return "", false
			},
			want: filepath.Join("~/dev", ".config/spindrift-driver/agents"),
		},
		{
			// An environment variable of its own would make the golden log
			// depend on the machine running the suite, so there is none.
			name: "no env override of its own",
			lookupEnv: func(name string) (string, bool) {
				if name == "SPINDRIFT_AGENTS" {
					return "/from/env", true
				}
				return "", false
			},
			want: "",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := agentsDirPath(tc.lookupEnv); got != tc.want {
				t.Errorf("agentsDirPath = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestLoadAgentDirMissingDirIsDistinct: an absent directory is the sentinel, and
// an empty one is not — "no Agent Files because there is nothing there to read"
// has to stay separable from "the directory is broken" for the caller.
func TestLoadAgentDirMissingDirIsDistinct(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "agents")

	entries, err := loadAgentDir(missing)
	assertErrMessage(t, err, missing, "directory not found")
	if !errors.Is(err, errNoAgentDir) {
		t.Errorf("errors.Is(err, errNoAgentDir) = false, want true; err = %v", err)
	}
	if entries != nil {
		t.Errorf("entries = %+v, want nil", entries)
	}

	empty, err := loadAgentDir(t.TempDir())
	if err != nil {
		t.Fatalf("loadAgentDir on an existing empty dir error = %v, want nil", err)
	}
	if len(empty) != 0 {
		t.Errorf("entries from an existing empty dir = %+v, want none", empty)
	}
}

func TestParseAgentFileBodyIsVerbatim(t *testing.T) {
	path := writeAgentFile(t, t.TempDir(), "a.md", "---\nname: a\nmodel: m\n---\n\nFirst line.\n\n  indented\n\nLast.\n\n")

	entry, err := parseAgentFile(path)
	if err != nil {
		t.Fatalf("parseAgentFile(%q) error = %v, want nil", path, err)
	}
	if want := "First line.\n\n  indented\n\nLast."; entry.SystemPrompt != want {
		t.Errorf("SystemPrompt = %q, want %q", entry.SystemPrompt, want)
	}
}

func TestParseAgentFileDescriptionKeepsColons(t *testing.T) {
	path := writeAgentFile(t, t.TempDir(), "a.md", "---\nname: a\ndescription: reads: the diff, then reports\nmodel: m\n---\nbody\n")

	entry, err := parseAgentFile(path)
	if err != nil {
		t.Fatalf("parseAgentFile(%q) error = %v, want nil", path, err)
	}
	if want := "reads: the diff, then reports"; entry.Description != want {
		t.Errorf("Description = %q, want %q", entry.Description, want)
	}
}

func TestParseAgentFileCRLFLines(t *testing.T) {
	path := writeAgentFile(t, t.TempDir(), "a.md", "---\r\nname: a\r\nmodel: m\r\n---\r\nbody\r\n")

	entry, err := parseAgentFile(path)
	if err != nil {
		t.Fatalf("parseAgentFile(%q) error = %v, want nil", path, err)
	}
	if entry.Name != "a" || entry.SystemPrompt != "body" {
		t.Errorf("entry = %+v, want the CRLF file read as LF", entry)
	}
}

// offeredNames reduces a tool list to what a failure message can show without
// dumping four structs.
func offeredNames(tools []toolSpec) []string {
	var names []string
	for _, tool := range tools {
		names = append(names, tool.Function.Name)
	}
	return names
}

// TestOfferedTools: the Subagent Tool exists in the offer only when there is a
// roster to delegate to — an empty roster must never offer a call that has no
// subagent to answer it.
func TestOfferedTools(t *testing.T) {
	if got := offeredNames(offeredTools(nil)); !reflect.DeepEqual(got, []string{"bash", "edit", "write"}) {
		t.Errorf("offeredTools(nil) = %v, want [bash edit write]", got)
	}
	if got := offeredNames(offeredTools([]agentEntry{})); !reflect.DeepEqual(got, []string{"bash", "edit", "write"}) {
		t.Errorf("offeredTools(empty slice) = %v, want [bash edit write]", got)
	}
	roster := []agentEntry{{Name: "reviewer", Model: "m-r"}}
	if got := offeredNames(offeredTools(roster)); !reflect.DeepEqual(got, []string{"bash", "edit", "write", "subagent"}) {
		t.Errorf("offeredTools(roster) = %v, want [bash edit write subagent]", got)
	}
	// The entries are the package-level specs themselves, not copies that can
	// drift from the shape the Model is taught. toolSpec carries a map, so
	// DeepEqual is the only comparison available.
	if got := offeredTools(roster); !reflect.DeepEqual(got, []toolSpec{bashTool, editTool, writeTool, subagentTool}) {
		t.Errorf("offeredTools(roster) = %+v, want [bashTool editTool writeTool subagentTool]", got)
	}
}

// rosterNames reduces an entry list to the names a failure message can show.
func rosterNames(entries []agentEntry) []string {
	var names []string
	for _, entry := range entries {
		names = append(names, entry.Name)
	}
	return names
}

// TestSpawnable: spawns is a grant over the roster, so the roster answers twice
// — a name with no Agent File can never be honoured — and the answer keeps the
// roster's lexical order rather than the frontmatter's, which is what makes a
// spawns-derived offer byte-stable.
func TestSpawnable(t *testing.T) {
	roster := []agentEntry{
		{Name: "planner", Model: "m", Spawns: []string{"worker", "critic", "worker", "ghost"}},
		{Name: "critic", Model: "m"},
		{Name: "quiet", Model: "m"},
		{Name: "worker", Model: "m", Spawns: []string{"critic"}},
	}
	tests := []struct {
		name  string
		agent string
		want  []string
	}{
		{"top level is unrestricted", "", []string{"planner", "critic", "quiet", "worker"}},
		{"roster order wins over frontmatter order, unknown names drop, duplicates fold", "planner", []string{"critic", "worker"}},
		{"child of one", "worker", []string{"critic"}},
		{"naming nobody spawns nobody", "quiet", nil},
		{"caller off the roster spawns nobody", "stranger", nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := spawnable(tc.agent, roster)
			if !reflect.DeepEqual(rosterNames(got), tc.want) {
				t.Errorf("spawnable(%q) = %v, want %v", tc.agent, rosterNames(got), tc.want)
			}
		})
	}

	// The helper answers with the roster's own entries and never rewrites them:
	// a caller that appends to the result must not widen what the next run sees.
	base := []agentEntry{{Name: "a", Model: "m"}, {Name: "b", Model: "m"}}
	if got := spawnable("", base); !reflect.DeepEqual(got, base) || !reflect.DeepEqual(base, []agentEntry{{Name: "a", Model: "m"}, {Name: "b", Model: "m"}}) {
		t.Errorf("spawnable(\"\") = %v over roster %+v, want the roster's own entries and an unchanged roster", rosterNames(got), base)
	}
}

// TestToolsForEntry: a grant restricts the offer and never adds to it, the offer
// comes out in the canonical order whatever order the file typed the grants in,
// and an unknown grant name names no Tool at all.
func TestToolsForEntry(t *testing.T) {
	// The subagent leg asks the roster who the caller may name, so the fixture
	// varies that answer by name rather than by the entry handed in.
	roster := []agentEntry{
		{Name: "planner", Model: "m", Spawns: []string{"worker"}},
		{Name: "quiet", Model: "m"},
		{Name: "wide", Model: "m", Spawns: []string{"ghost"}},
		{Name: "worker", Model: "m"},
	}
	tests := []struct {
		name  string
		entry agentEntry
		want  []string
	}{
		{"no tools granted is the bash floor", agentEntry{Name: "quiet", Model: "m"}, []string{"bash"}},
		{"grants keep canonical order", agentEntry{Name: "editor", Model: "m", Tools: []string{"write", "bash", "edit"}}, []string{"bash", "edit", "write"}},
		{"bash alone stays bash", agentEntry{Name: "shell", Model: "m", Tools: []string{"bash"}}, []string{"bash"}},
		{"subagent with someone to name", agentEntry{Name: "planner", Model: "m", Tools: []string{"subagent", "bash"}}, []string{"bash", "subagent"}},
		{"subagent grant whose spawns name only unknown agents", agentEntry{Name: "wide", Model: "m", Tools: []string{"bash", "subagent"}}, []string{"bash"}},
		{"subagent grant with no spawns field names nobody", agentEntry{Name: "quiet", Model: "m", Tools: []string{"bash", "subagent"}}, []string{"bash"}},
		{"unknown grant names no Tool", agentEntry{Name: "quiet", Model: "m", Tools: []string{"bash", "read"}}, []string{"bash"}},
		{"unknown grant names cannot reach a fifth Tool", agentEntry{Name: "quiet", Model: "m", Tools: []string{"read", "web"}}, []string{"bash"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := offeredNames(toolsForEntry(tc.entry, roster))
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("toolsForEntry(%+v) = %v, want %v", tc.entry, got, tc.want)
			}
		})
	}

	// The offer is the package-level specs themselves, not copies that can drift
	// from the shape the Model is taught.
	if got := toolsForEntry(agentEntry{Name: "planner", Model: "m", Tools: []string{"subagent", "write", "edit", "bash"}}, roster); !reflect.DeepEqual(got, []toolSpec{bashTool, editTool, writeTool, subagentTool}) {
		t.Errorf("toolsForEntry(full grant) = %+v, want [bashTool editTool writeTool subagentTool]", got)
	}
}
