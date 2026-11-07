package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// writeHookManifest puts contents in a file of its own and hands back the path,
// which is the only thing loadHookManifest is given: it never resolves a
// default, so a test that wants a miss hands back a path it never wrote.
func writeHookManifest(t *testing.T, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "hooks.json")
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("write hook manifest fixture: %v", err)
	}
	return path
}

func TestLoadHookManifest(t *testing.T) {
	path := writeHookManifest(t, `{
	  "pre": [
	    {"path": "/lib/hooks/deny.sh", "tools": ["Bash", "Write"], "rewrites": false},
	    {"path": "/lib/hooks/sanitize.sh", "tools": ["Bash"], "rewrites": true}
	  ]
	}`)

	hooks, err := loadHookManifest(path)
	if err != nil {
		t.Fatalf("loadHookManifest(%s) = %v, want no error", path, err)
	}
	want := []guardHook{
		{path: "/lib/hooks/deny.sh", tools: []string{"Bash", "Write"}, rewrites: false},
		{path: "/lib/hooks/sanitize.sh", tools: []string{"Bash"}, rewrites: true},
	}
	if len(hooks) != len(want) {
		t.Fatalf("loadHookManifest returned %d hooks, want %d: %+v", len(hooks), len(want), hooks)
	}
	// Run order is file order, so the comparison is positional on purpose.
	for i, w := range want {
		got := hooks[i]
		if got.path != w.path || got.rewrites != w.rewrites || !slices.Equal(got.tools, w.tools) {
			t.Errorf("pre[%d] = %+v, want %+v", i, got, w)
		}
	}
}

func TestLoadHookManifestEmptyPre(t *testing.T) {
	path := writeHookManifest(t, `{"pre":[]}`)

	hooks, err := loadHookManifest(path)
	if err != nil {
		t.Fatalf("loadHookManifest(%s) = %v, want no error", path, err)
	}
	if len(hooks) != 0 {
		t.Errorf("loadHookManifest returned %d hooks, want none: %+v", len(hooks), hooks)
	}
}

func TestLoadHookManifestKeepsFileOrder(t *testing.T) {
	// Nothing sorts the entries: the manifest's order is the run order, so a
	// denial in slot 0 stays the thing that runs first.
	path := writeHookManifest(t, `{"pre":[{"path":"/lib/hooks/zulu.sh"},{"path":"/lib/hooks/alpha.sh"}]}`)

	hooks, err := loadHookManifest(path)
	if err != nil {
		t.Fatalf("loadHookManifest(%s) = %v, want no error", path, err)
	}
	want := []string{"/lib/hooks/zulu.sh", "/lib/hooks/alpha.sh"}
	if len(hooks) != len(want) {
		t.Fatalf("loadHookManifest returned %d hooks, want %d: %+v", len(hooks), len(want), hooks)
	}
	for i, w := range want {
		if hooks[i].path != w {
			t.Errorf("pre[%d].path = %q, want %q", i, hooks[i].path, w)
		}
	}
}

func TestLoadHookManifestMissingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.json")

	_, err := loadHookManifest(path)
	assertErrMessage(t, err, path, "cannot read")
}

func TestLoadHookManifestTopLevelNotAnObject(t *testing.T) {
	for _, tc := range []struct{ name, contents, want string }{
		{"array", `[{"path":"/lib/hooks/deny.sh"}]`, "array"},
		{"string", `"/lib/hooks/deny.sh"`, "string"},
		{"number", `42`, "number"},
		// A top-level null and an empty file are the two documents json would
		// decode into the same nil map as "no hooks", so they are named here.
		{"null", `null`, "null"},
		{"empty file", ``, "empty"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeHookManifest(t, tc.contents)

			_, err := loadHookManifest(path)
			assertErrMessage(t, err, path, "malformed JSON: top-level value is", tc.want, "want an object")
		})
	}
}

func TestLoadHookManifestMalformedJSON(t *testing.T) {
	for _, tc := range []struct{ name, contents string }{
		{"not JSON", `{`},
		{"trailing bytes", `{"pre":[]} {"pre":[]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeHookManifest(t, tc.contents)

			_, err := loadHookManifest(path)
			assertErrMessage(t, err, path, "malformed JSON")
		})
	}
}

func TestLoadHookManifestUnknownField(t *testing.T) {
	for _, tc := range []struct{ name, contents, want string }{
		{"post hooks are not in this slice", `{"pre":[],"post":[{"path":"/lib/hooks/summarize.sh"}]}`, "post"},
		{"near-miss typo", `{"Pre":[]}`, "Pre"},
		{"stray key", `{"pre":[],"version":2}`, "version"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeHookManifest(t, tc.contents)

			_, err := loadHookManifest(path)
			assertErrMessage(t, err, path, "unknown field", tc.want)
		})
	}
}

func TestLoadHookManifestUnknownFieldReportsLexicallyFirst(t *testing.T) {
	// The key order in the file must not decide which complaint is printed: the
	// same file always names the same key.
	path := writeHookManifest(t, `{"zulu":1,"alpha":2,"Mango":3}`)

	_, err := loadHookManifest(path)
	assertErrMessage(t, err, path, "unknown field", `"Mango"`)
	if strings.Contains(err.Error(), "alpha") || strings.Contains(err.Error(), "zulu") {
		t.Errorf("error %q names a key other than the lexically first", err.Error())
	}
}

func TestLoadHookManifestWrongType(t *testing.T) {
	for _, tc := range []struct{ name, contents, want string }{
		{"pre object", `{"pre":{}}`, `field "pre" must be an array, got object`},
		{"pre null", `{"pre":null}`, `field "pre" must be an array, got null`},
		{"pre string", `{"pre":"/lib/hooks/deny.sh"}`, `field "pre" must be an array, got string`},
		{"pre number", `{"pre":1}`, `field "pre" must be an array, got number`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeHookManifest(t, tc.contents)

			_, err := loadHookManifest(path)
			assertErrMessage(t, err, path, tc.want)
		})
	}
}

func TestLoadHookManifestBadEntry(t *testing.T) {
	for _, tc := range []struct{ name, contents, want string }{
		{"entry is a string", `{"pre":["/lib/hooks/deny.sh"]}`, `pre[0] must be an object, got string`},
		{"entry is null", `{"pre":[null]}`, `pre[0] must be an object, got null`},
		{"entry is an array", `{"pre":[["/lib/hooks/deny.sh"]]}`, `pre[0] must be an object, got array`},
		{"no path", `{"pre":[{}]}`, `pre[0]: missing field "path"`},
		{"empty path", `{"pre":[{"path":""}]}`, `pre[0]: field "path" must not be empty`},
		{"path number", `{"pre":[{"path":8080}]}`, `pre[0]: field "path" must be a string, got number`},
		{"path null", `{"pre":[{"path":null}]}`, `pre[0]: field "path" must be a string, got null`},
		{"path object", `{"pre":[{"path":{"p":"/x"}}]}`, `pre[0]: field "path" must be a string, got object`},
		{"entry unknown field", `{"pre":[{"path":"/x.sh","post":true}]}`, `pre[0]: unknown field "post"`},
		{"tools string", `{"pre":[{"path":"/x.sh","tools":"Bash"}]}`, `pre[0]: field "tools" must be an array, got string`},
		{"tools null", `{"pre":[{"path":"/x.sh","tools":null}]}`, `pre[0]: field "tools" must be an array, got null`},
		{"tools item number", `{"pre":[{"path":"/x.sh","tools":["Bash",8080]}]}`, `pre[0]: field "tools"[1] must be a string, got number`},
		{"tools item null", `{"pre":[{"path":"/x.sh","tools":[null]}]}`, `pre[0]: field "tools"[0] must be a string, got null`},
		{"rewrites string", `{"pre":[{"path":"/x.sh","rewrites":"yes"}]}`, `pre[0]: field "rewrites" must be a boolean, got string`},
		{"rewrites number", `{"pre":[{"path":"/x.sh","rewrites":1}]}`, `pre[0]: field "rewrites" must be a boolean, got number`},
		{"rewrites null", `{"pre":[{"path":"/x.sh","rewrites":null}]}`, `pre[0]: field "rewrites" must be a boolean, got null`},
		// The second entry is where the complaint lands when the first is fine,
		// which is what makes the position worth printing.
		{"later entry", `{"pre":[{"path":"/ok.sh"},{"path":"/x.sh","tools":["Bash",2]}]}`, `pre[1]: field "tools"[1] must be a string, got number`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeHookManifest(t, tc.contents)

			_, err := loadHookManifest(path)
			assertErrMessage(t, err, path, tc.want)
		})
	}
}

func TestGuardHookAppliesTo(t *testing.T) {
	for _, tc := range []struct {
		name string
		hook guardHook
		tool string
		want bool
	}{
		{"listed tool", guardHook{path: "/x.sh", tools: []string{"Bash", "Write"}}, "Bash", true},
		{"second listed tool", guardHook{path: "/x.sh", tools: []string{"Bash", "Write"}}, "Write", true},
		{"unlisted tool", guardHook{path: "/x.sh", tools: []string{"Bash"}}, "Read", false},
		// Exact match only: the manifest speaks the hook protocol's names, and
		// folding case here would silently widen a hook's reach.
		{"case differs", guardHook{path: "/x.sh", tools: []string{"Bash"}}, "bash", false},
		{"empty tools guards everything", guardHook{path: "/x.sh", tools: []string{}}, "Anything", true},
		{"absent tools guards everything", guardHook{path: "/x.sh"}, "Anything", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.hook.appliesTo(tc.tool); got != tc.want {
				t.Errorf("guardHook %+v .appliesTo(%q) = %v, want %v", tc.hook, tc.tool, got, tc.want)
			}
		})
	}
}

func TestLoadHookManifestNoPreField(t *testing.T) {
	path := writeHookManifest(t, `{}`)

	hooks, err := loadHookManifest(path)
	if err != nil {
		t.Fatalf("loadHookManifest(%s) = %v, want no error: a manifest that says nothing is a valid manifest", path, err)
	}
	if len(hooks) != 0 {
		t.Errorf("loadHookManifest returned %d hooks, want none: %+v", len(hooks), hooks)
	}
}

func TestHookToolName(t *testing.T) {
	// The manifest and the hooks themselves speak the hook protocol's tool
	// names; only bash needs translating today.
	for _, tc := range []struct{ driver, want string }{
		{"bash", "Bash"},
		{"Bash", "Bash"},
		{"read", "read"},
		{"", ""},
	} {
		if got := hookToolName(tc.driver); got != tc.want {
			t.Errorf("hookToolName(%q) = %q, want %q", tc.driver, got, tc.want)
		}
	}
}

// hookInterpreter resolves bash for the fixture scripts' shebang. The
// portable spelling is "#!/usr/bin/env bash", but a sandbox without /usr/bin
// — which is what this tree's CI runs in — makes every fixture unstartable and
// every assertion about a hook that did not run meaningless, so the interpreter
// is found on PATH the way main_test.go finds sleep.
func hookInterpreter(t *testing.T) string {
	t.Helper()
	bash, err := exec.LookPath("bash")
	if err != nil {
		t.Fatalf("locate bash for the hook fixtures: %v", err)
	}
	return bash
}

// writeGuardHookScript writes a real Guard Hook script — the parent spec's
// "real scripts, not doubles" — with its own shebang and mode bits, so
// guardCommand has to exec it the way the host bakes it rather than hand it to
// a shell.
func writeGuardHookScript(t *testing.T, name, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte("#!"+hookInterpreter(t)+"\n"+body+"\n"), 0o755); err != nil {
		t.Fatalf("write guard hook script %s: %v", name, err)
	}
	return path
}

// hookMarkerExists reports whether the script left its marker behind. A stat
// error other than "not there" is a broken fixture, not a verdict on the hook.
func hookMarkerExists(t *testing.T, path string) bool {
	t.Helper()
	_, err := os.Stat(path)
	if err == nil {
		return true
	}
	if os.IsNotExist(err) {
		return false
	}
	t.Fatalf("stat marker %s: %v", path, err)
	return false
}

func TestGuardCommandDenies(t *testing.T) {
	script := writeGuardHookScript(t, "deny.sh",
		`echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"credential path"}}'`)

	got := guardCommand("cat ~/.ssh/id_ed25519", "bash", []guardHook{{path: script}}, nil)

	// The "error: " prefix is what makes the Tool Result an error downstream,
	// and the script path is what the operator has to go read.
	want := "error: guard hook " + script + " denied this command: credential path"
	if got != want {
		t.Errorf("guardCommand = %q, want %q", got, want)
	}
}

func TestGuardCommandDenyWithoutReasonUsesFallback(t *testing.T) {
	for _, tc := range []struct{ name, body string }{
		{"missing reason", `echo '{"hookSpecificOutput":{"permissionDecision":"deny"}}'`},
		{"empty reason", `echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":""}}'`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			script := writeGuardHookScript(t, "deny.sh", tc.body)

			got := guardCommand("rm -rf /", "bash", []guardHook{{path: script}}, nil)

			want := "error: guard hook " + script + " denied this command: the hook gave no reason"
			if got != want {
				t.Errorf("guardCommand = %q, want %q", got, want)
			}
		})
	}
}

func TestGuardCommandProceedsUnlessTheHookDenies(t *testing.T) {
	// "null hookSpecificOutput" passes while a string or number there refuses:
	// null is JSON for "not there", so it is the absence of a verdict, whereas a
	// value of the wrong shape is a hook that answered something unreadable.
	for _, tc := range []struct{ name, body string }{
		{"allow", `echo '{"hookSpecificOutput":{"permissionDecision":"allow","permissionDecisionReason":"fine"}}'`},
		{"ask", `echo '{"hookSpecificOutput":{"permissionDecision":"ask"}}'`},
		{"near-miss decision", `echo '{"hookSpecificOutput":{"permissionDecision":"Deny"}}'`},
		{"no decision", `echo '{"hookSpecificOutput":{}}'`},
		{"no hookSpecificOutput", `echo '{"somethingElse":true}'`},
		{"null hookSpecificOutput", `echo '{"hookSpecificOutput":null}'`},
		{"empty stdout", `:`},
		{"whitespace stdout", `printf '  \n\t\n'`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			script := writeGuardHookScript(t, "hook.sh", tc.body)

			if got := guardCommand("ls", "bash", []guardHook{{path: script}}, nil); got != "" {
				t.Errorf("guardCommand = %q, want the empty string (proceed)", got)
			}
		})
	}
}

func TestGuardCommandFailsClosedOnAMisbehavingHook(t *testing.T) {
	// Silence is a verdict of "no opinion", but a hook that exits non-zero or
	// answers with unreadable stdout has said nothing usable, and the call is
	// refused rather than waved through.
	for _, tc := range []struct{ name, body, want string }{
		{"non-zero exit", `echo '{"hookSpecificOutput":{"permissionDecision":"allow"}}' >&2; exit 3`, "failed: exit status 3"},
		{"stdout is not JSON", `echo 'not json at all'`, "returned unparsable output"},
		{"JSON with trailing junk", `echo '{"hookSpecificOutput":{}} trailing'`, "returned unparsable output"},
		{"stdout is a bare string", `echo '"deny"'`, "returned unparsable output"},
		{"hookSpecificOutput is a string", `echo '{"hookSpecificOutput":"deny"}'`, "returned unparsable output"},
		{"hookSpecificOutput is a number", `echo '{"hookSpecificOutput":7}'`, "returned unparsable output"},
		{"hookSpecificOutput is an array", `echo '{"hookSpecificOutput":["deny"]}'`, "returned unparsable output"},
		{"hookSpecificOutput is a boolean", `echo '{"hookSpecificOutput":true}'`, "returned unparsable output"},
		{"decision is a number", `echo '{"hookSpecificOutput":{"permissionDecision":7}}'`, "returned unparsable output"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			script := writeGuardHookScript(t, "hook.sh", tc.body)

			got := guardCommand("ls", "bash", []guardHook{{path: script}}, nil)

			want := "error: guard hook " + script + " " + tc.want
			if got != want {
				t.Errorf("guardCommand = %q, want %q", got, want)
			}
		})
	}
}

func TestGuardCommandFailsClosedWhenTheHookCannotBeRun(t *testing.T) {
	// A manifest entry pointing at a script that is not on disk is the one
	// misbehaviour the fixture cannot script, so the path is simply never written.
	missing := filepath.Join(t.TempDir(), "never-installed.sh")

	got := guardCommand("ls", "bash", []guardHook{{path: missing}}, nil)

	if !strings.HasPrefix(got, "error: ") {
		t.Errorf("guardCommand = %q, want an error: prefixed refusal", got)
	}
	if !strings.Contains(got, missing) {
		t.Errorf("refusal %q does not name the hook %q", got, missing)
	}
	if !strings.Contains(got, "could not be run") {
		t.Errorf("refusal %q does not say the hook could not be run", got)
	}
	if strings.Contains(got, "denied") {
		t.Errorf("refusal %q blames the hook for a verdict it never made", got)
	}
}

func TestGuardCommandFailsClosedWhenTheHookHangs(t *testing.T) {
	old := guardHookTimeout
	guardHookTimeout = 100 * time.Millisecond
	defer func() { guardHookTimeout = old }()
	script := writeGuardHookScript(t, "hang.sh", "sleep 30")

	got := guardCommand("ls", "bash", []guardHook{{path: script}}, nil)

	want := "error: guard hook " + script + " timed out after 100ms, so the command is blocked"
	if got != want {
		t.Errorf("guardCommand = %q, want %q", got, want)
	}
}

func TestGuardCommandAFailingHookStopsTheChain(t *testing.T) {
	// The refusal a non-zero exit produces stops the chain like a denial does:
	// the second hook never gets to speak for a command already blocked.
	dir := t.TempDir()
	secondMarker := filepath.Join(dir, "second-ran")
	first := writeGuardHookScript(t, "first.sh", "exit 3")
	second := writeGuardHookScript(t, "second.sh",
		`touch `+secondMarker+`
echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"too late"}}'`)

	got := guardCommand("ls", "bash", []guardHook{{path: first}, {path: second}}, nil)

	if want := "error: guard hook " + first + " failed: exit status 3"; got != want {
		t.Errorf("guardCommand = %q, want %q", got, want)
	}
	if hookMarkerExists(t, secondMarker) {
		t.Errorf("hook %s ran although an earlier hook already refused", second)
	}
}

func TestGuardCommandNoHooksPaysNothing(t *testing.T) {
	for _, hooks := range [][]guardHook{nil, {}} {
		if got := guardCommand("ls", "bash", hooks, nil); got != "" {
			t.Errorf("guardCommand with no hooks = %q, want the empty string", got)
		}
	}
}

func TestGuardCommandSkipsHookThatDoesNotApply(t *testing.T) {
	dir := t.TempDir()
	marker := filepath.Join(dir, "ran")
	script := writeGuardHookScript(t, "write-only.sh",
		`touch `+marker+`
echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"no"}}'`)

	got := guardCommand("ls", "bash", []guardHook{{path: script, tools: []string{"Write"}}}, nil)

	if got != "" {
		t.Errorf("guardCommand = %q, want the empty string", got)
	}
	if hookMarkerExists(t, marker) {
		t.Errorf("hook %s ran for a tool its tools list does not name", script)
	}
}

func TestGuardCommandEmptyToolsApplies(t *testing.T) {
	dir := t.TempDir()
	marker := filepath.Join(dir, "ran")
	script := writeGuardHookScript(t, "any.sh",
		`touch `+marker+`
echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"because"}}'`)

	got := guardCommand("ls", "bash", []guardHook{{path: script, tools: []string{}}}, nil)

	if !hookMarkerExists(t, marker) {
		t.Errorf("hook %s did not run: an empty tools list guards every tool", script)
	}
	if want := "error: guard hook " + script + " denied this command: because"; got != want {
		t.Errorf("guardCommand = %q, want %q", got, want)
	}
}

func TestGuardCommandFirstRefusalStopsTheChain(t *testing.T) {
	dir := t.TempDir()
	secondMarker := filepath.Join(dir, "second-ran")
	first := writeGuardHookScript(t, "first.sh",
		`echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"first word"}}'`)
	second := writeGuardHookScript(t, "second.sh",
		`touch `+secondMarker+`
echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"too late"}}'`)

	got := guardCommand("ls", "bash", []guardHook{{path: first}, {path: second}}, nil)

	if want := "error: guard hook " + first + " denied this command: first word"; got != want {
		t.Errorf("guardCommand = %q, want %q", got, want)
	}
	if hookMarkerExists(t, secondMarker) {
		t.Errorf("hook %s ran although an earlier hook already refused", second)
	}
}

func TestGuardCommandSendsTheProtocolRequest(t *testing.T) {
	// The expected bytes are written out instead of generated: the payload is the
	// protocol's, so '<' and '&' travel literally and only '"' needs escaping. A
	// generated comparison would also accept \u003c-style HTML escapes, which
	// decode to the same value but are not the payload the protocol shows.
	for _, tc := range []struct{ name, command, want string }{
		{
			"plain command",
			"ls -la ~/.ssh",
			`{"tool_name":"Bash","tool_input":{"command":"ls -la ~/.ssh"}}`,
		},
		{
			"quotes and markup",
			`echo "<a> & 'b'"`,
			`{"tool_name":"Bash","tool_input":{"command":"echo \"<a> & 'b'\""}}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			received := filepath.Join(dir, "request.json")
			script := writeGuardHookScript(t, "record.sh", "cat > "+received)

			if got := guardCommand(tc.command, "bash", []guardHook{{path: script}}, nil); got != "" {
				t.Fatalf("guardCommand = %q, want the empty string", got)
			}
			raw, err := os.ReadFile(received)
			if err != nil {
				t.Fatalf("read what the hook received: %v", err)
			}
			if string(raw) != tc.want {
				t.Errorf("hook received %s, want exactly %s", raw, tc.want)
			}
		})
	}
}

func TestGuardCommandForwardsHookStderrToTheOperatorCopy(t *testing.T) {
	// A hook's own noise is the operator's progress copy, never the wire's, and
	// a nil writer means there is no copy to make.
	for _, tc := range []struct{ name, writer string }{
		{"stderr writer", "buffer"},
		{"nil stderr", "nil"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			script := writeGuardHookScript(t, "noisy.sh",
				`echo 'hook noise' >&2
echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"nope"}}'`)
			var out bytes.Buffer
			var stderr io.Writer
			if tc.writer == "buffer" {
				stderr = &out
			}

			got := guardCommand("ls", "bash", []guardHook{{path: script}}, stderr)

			if want := "error: guard hook " + script + " denied this command: nope"; got != want {
				t.Errorf("guardCommand = %q, want %q", got, want)
			}
			if tc.writer == "buffer" && !strings.Contains(out.String(), "hook noise") {
				t.Errorf("stderr copy = %q, want it to contain the hook's noise", out.String())
			}
			if strings.Contains(got, "hook noise") {
				t.Errorf("refusal %q carries the hook's stderr, which belongs to the operator's copy", got)
			}
		})
	}
}

func TestGuardCommandFailsClosedWhenTheHookOverrunsTheOutputCap(t *testing.T) {
	old := guardHookOutputLimit
	guardHookOutputLimit = 4096
	t.Cleanup(func() { guardHookOutputLimit = old })

	dir := t.TempDir()
	secondMarker := filepath.Join(dir, "second-ran")
	// The flood is whitespace and the verdict sits past the cap, so a cap that
	// quietly dropped the tail would hand back whitespace and read as silence.
	first := writeGuardHookScript(t, "flood.sh",
		`yes '' | head -c 8192
echo '{"hookSpecificOutput":{"permissionDecision":"allow","permissionDecisionReason":"fine"}}'`)
	second := writeGuardHookScript(t, "second.sh",
		`touch `+secondMarker+`
echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"too late"}}'`)

	got := guardCommand("ls", "bash", []guardHook{{path: first}, {path: second}}, nil)

	if !strings.HasPrefix(got, "error: ") {
		t.Errorf("guardCommand = %q, want an error: prefixed refusal", got)
	}
	if !strings.Contains(got, first) {
		t.Errorf("refusal %q does not name the hook %q", got, first)
	}
	if !strings.Contains(got, fmt.Sprintf("%d bytes", guardHookOutputLimit)) {
		t.Errorf("refusal %q does not name the output bound of %d bytes", got, guardHookOutputLimit)
	}
	if !strings.Contains(got, "blocked") {
		t.Errorf("refusal %q does not say the command is blocked", got)
	}
	if hookMarkerExists(t, secondMarker) {
		t.Errorf("hook %s ran although an earlier hook already refused", second)
	}
}

func TestGuardCommandOutputExactlyAtTheCapPasses(t *testing.T) {
	// The bound is inclusive: a hook that fills it is a hook that answered, so an
	// off-by-one that reads "reached" as "exceeded" refuses every long verdict.
	old := guardHookOutputLimit
	guardHookOutputLimit = 4096
	t.Cleanup(func() { guardHookOutputLimit = old })
	script := writeGuardHookScript(t, "exact.sh",
		fmt.Sprintf(`yes '' | head -c %d`, guardHookOutputLimit))

	if got := guardCommand("ls", "bash", []guardHook{{path: script}}, nil); got != "" {
		t.Errorf("guardCommand = %q, want the empty string (proceed) for output at the cap", got)
	}
}
