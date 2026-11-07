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

	manifest, err := loadHookManifest(path)
	if err != nil {
		t.Fatalf("loadHookManifest(%s) = %v, want no error", path, err)
	}
	hooks := manifest.pre
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

	manifest, err := loadHookManifest(path)
	if err != nil {
		t.Fatalf("loadHookManifest(%s) = %v, want no error", path, err)
	}
	if len(manifest.pre) != 0 {
		t.Errorf("loadHookManifest returned %d hooks, want none: %+v", len(manifest.pre), manifest.pre)
	}
}

func TestLoadHookManifestKeepsFileOrder(t *testing.T) {
	// Nothing sorts the entries: the manifest's order is the run order, so a
	// denial in slot 0 stays the thing that runs first.
	path := writeHookManifest(t, `{"pre":[{"path":"/lib/hooks/zulu.sh"},{"path":"/lib/hooks/alpha.sh"}]}`)

	manifest, err := loadHookManifest(path)
	if err != nil {
		t.Fatalf("loadHookManifest(%s) = %v, want no error", path, err)
	}
	want := []string{"/lib/hooks/zulu.sh", "/lib/hooks/alpha.sh"}
	if len(manifest.pre) != len(want) {
		t.Fatalf("loadHookManifest returned %d hooks, want %d: %+v", len(manifest.pre), len(want), manifest.pre)
	}
	for i, w := range want {
		if manifest.pre[i].path != w {
			t.Errorf("pre[%d].path = %q, want %q", i, manifest.pre[i].path, w)
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
		{"summarise is not the key", `{"pre":[],"summarize":{"path":"/lib/hooks/summarize.sh"}}`, "summarize"},
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

	manifest, err := loadHookManifest(path)
	if err != nil {
		t.Fatalf("loadHookManifest(%s) = %v, want no error: a manifest that says nothing is a valid manifest", path, err)
	}
	if len(manifest.pre) != 0 {
		t.Errorf("loadHookManifest returned %d hooks, want none: %+v", len(manifest.pre), manifest.pre)
	}
	if manifest.summary != nil {
		t.Errorf("summary = %+v, want nil", *manifest.summary)
	}
}

// The manifest lists the pre-call hooks in run order plus one post-call summary
// hook, so a post entry has to survive the load even though nothing applies it
// yet (the rewrite of a Tool Result is the next slice).
func TestLoadHookManifestSummaryHook(t *testing.T) {
	path := writeHookManifest(t, `{
	  "pre": [{"path": "/lib/hooks/deny.sh"}],
	  "post": {"path": "/lib/hooks/summarize.sh", "tools": ["Bash", "Write"]}
	}`)

	manifest, err := loadHookManifest(path)
	if err != nil {
		t.Fatalf("loadHookManifest(%s) = %v, want no error", path, err)
	}
	if len(manifest.pre) != 1 {
		t.Errorf("pre = %+v, want the one pre-call hook still parsed", manifest.pre)
	}
	if manifest.summary == nil {
		t.Fatalf("summary = nil, want the post-call entry parsed")
	}
	if manifest.summary.path != "/lib/hooks/summarize.sh" {
		t.Errorf("summary.path = %q, want %q", manifest.summary.path, "/lib/hooks/summarize.sh")
	}
	if want := []string{"Bash", "Write"}; !slices.Equal(manifest.summary.tools, want) {
		t.Errorf("summary.tools = %v, want %v", manifest.summary.tools, want)
	}
}

// A manifest that names only post-call work is still a manifest: the absent pre
// list must not stand in front of the summary hook.
func TestLoadHookManifestSummaryWithoutPre(t *testing.T) {
	path := writeHookManifest(t, `{"post":{"path":"/lib/hooks/summarize.sh"}}`)

	manifest, err := loadHookManifest(path)
	if err != nil {
		t.Fatalf("loadHookManifest(%s) = %v, want no error", path, err)
	}
	if len(manifest.pre) != 0 {
		t.Errorf("pre = %+v, want none", manifest.pre)
	}
	if manifest.summary == nil {
		t.Fatalf("summary = nil, want the post-call entry parsed even with no pre list")
	}
	if manifest.summary.path != "/lib/hooks/summarize.sh" {
		t.Errorf("summary.path = %q, want %q", manifest.summary.path, "/lib/hooks/summarize.sh")
	}
	if len(manifest.summary.tools) != 0 {
		t.Errorf("summary.tools = %v, want none: an absent tools list means every tool", manifest.summary.tools)
	}
}

func TestLoadHookManifestNoSummary(t *testing.T) {
	path := writeHookManifest(t, `{"pre":[{"path":"/lib/hooks/deny.sh"}]}`)

	manifest, err := loadHookManifest(path)
	if err != nil {
		t.Fatalf("loadHookManifest(%s) = %v, want no error: post is optional", path, err)
	}
	if manifest.summary != nil {
		t.Errorf("summary = %+v, want nil when the file names no post-call hook", *manifest.summary)
	}
	if len(manifest.pre) != 1 || manifest.pre[0].path != "/lib/hooks/deny.sh" {
		t.Errorf("pre = %+v, want the one pre-call hook", manifest.pre)
	}
}

// The one post-call entry is held to the same shape rules as a pre-call one, so
// every complaint names the manifest and says which key went wrong: "post" in
// the place a pre-call entry's position would occupy.
func TestLoadHookManifestBadSummary(t *testing.T) {
	for _, tc := range []struct{ name, contents, want string }{
		// One summary hook is the contract, so an array is refused as the wrong
		// shape rather than read as a list of them.
		{"array", `{"post":[{"path":"/lib/hooks/summarize.sh"}]}`, `field "post" must be an object, got array`},
		// null is a written key with nothing in it, not the silence of an absent
		// post, so it is refused the way "pre":null is.
		{"null", `{"post":null}`, `field "post" must be an object, got null`},
		{"string", `{"post":"/lib/hooks/summarize.sh"}`, `field "post" must be an object, got string`},
		{"number", `{"post":2}`, `field "post" must be an object, got number`},
		// rewrites belongs to the pre-call chain's promise about its input; a
		// post-call entry has no such flag to set.
		{"rewrites", `{"post":{"path":"/x.sh","rewrites":true}}`, `post: unknown field "rewrites"`},
		{"typo", `{"post":{"path":"/x.sh","tool":["Bash"]}}`, `post: unknown field "tool"`},
		{"no path", `{"post":{"tools":["Bash"]}}`, `post: missing field "path"`},
		{"empty path", `{"post":{"path":""}}`, `post: field "path" must not be empty`},
		{"path number", `{"post":{"path":8080}}`, `post: field "path" must be a string, got number`},
		{"path null", `{"post":{"path":null}}`, `post: field "path" must be a string, got null`},
		{"tools string", `{"post":{"path":"/x.sh","tools":"Bash"}}`, `post: field "tools" must be an array, got string`},
		{"tools null", `{"post":{"path":"/x.sh","tools":null}}`, `post: field "tools" must be an array, got null`},
		{"tools item number", `{"post":{"path":"/x.sh","tools":["Bash",8080]}}`, `post: field "tools"[1] must be a string, got number`},
		{"tools item null", `{"post":{"path":"/x.sh","tools":[null]}}`, `post: field "tools"[0] must be a string, got null`},
		// The pre-call list is still read, and still reported on, when the summary
		// entry is the broken half of the file.
		{"with a pre list", `{"pre":[{"path":"/ok.sh"}],"post":{"tools":["Bash"]}}`, `post: missing field "path"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeHookManifest(t, tc.contents)

			_, err := loadHookManifest(path)
			assertErrMessage(t, err, path, tc.want)
		})
	}
}

func TestSummaryHookAppliesTo(t *testing.T) {
	for _, tc := range []struct {
		name string
		hook summaryHook
		tool string
		want bool
	}{
		{"listed tool", summaryHook{path: "/x.sh", tools: []string{"Bash", "Write"}}, "Bash", true},
		{"unlisted tool", summaryHook{path: "/x.sh", tools: []string{"Bash"}}, "Read", false},
		{"case differs", summaryHook{path: "/x.sh", tools: []string{"Bash"}}, "bash", false},
		{"empty tools covers everything", summaryHook{path: "/x.sh", tools: []string{}}, "Anything", true},
		{"absent tools covers everything", summaryHook{path: "/x.sh"}, "Anything", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.hook.appliesTo(tc.tool); got != tc.want {
				t.Errorf("summaryHook %+v .appliesTo(%q) = %v, want %v", tc.hook, tc.tool, got, tc.want)
			}
		})
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

	_, got := guardCommand("cat ~/.ssh/id_ed25519", "bash", []guardHook{{path: script}}, nil)

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

			_, got := guardCommand("rm -rf /", "bash", []guardHook{{path: script}}, nil)

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

			if _, got := guardCommand("ls", "bash", []guardHook{{path: script}}, nil); got != "" {
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

			_, got := guardCommand("ls", "bash", []guardHook{{path: script}}, nil)

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

	_, got := guardCommand("ls", "bash", []guardHook{{path: missing}}, nil)

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

	_, got := guardCommand("ls", "bash", []guardHook{{path: script}}, nil)

	want := "error: guard hook " + script + " timed out after 100ms, so the command is blocked"
	if got != want {
		t.Errorf("guardCommand = %q, want %q", got, want)
	}
}

func TestGuardCommandFailsClosedKillsTheHooksProcessGroup(t *testing.T) {
	// A hook that detaches a grandchild must not leave it running past the bound:
	// killing only the hook would orphan the spinner, still ticking in the
	// background. The grandchild closes the inherited output fds so it cannot be
	// what WaitDelay is unblocking on — the test is about the kill, not the wait.
	old := guardHookTimeout
	guardHookTimeout = 300 * time.Millisecond
	t.Cleanup(func() { guardHookTimeout = old })
	marker := filepath.Join(t.TempDir(), "ticks")
	t.Setenv("SPINDRIFT_HOOK_TICK_MARKER", marker)
	script := writeGuardHookScript(t, "spin.sh", `( exec 1>&- 2>&-; while :; do echo tick >> "$SPINDRIFT_HOOK_TICK_MARKER"; sleep 0.05; done ) &
sleep 30`)

	_, got := guardCommand("ls", "bash", []guardHook{{path: script}}, nil)

	want := "error: guard hook " + script + " timed out after 300ms, so the command is blocked"
	if got != want {
		t.Fatalf("guardCommand = %q, want %q", got, want)
	}

	before, err := os.ReadFile(marker)
	if err != nil {
		t.Fatalf("read tick marker %s: %v", marker, err)
	}
	if len(before) == 0 {
		t.Fatalf("tick marker %s is empty: the detached grandchild never ran, so nothing was killed", marker)
	}
	time.Sleep(250 * time.Millisecond)
	after, err := os.ReadFile(marker)
	if err != nil {
		t.Fatalf("read tick marker %s: %v", marker, err)
	}
	if len(after) != len(before) {
		t.Errorf("detached grandchild survived the timeout: marker grew from %d to %d bytes", len(before), len(after))
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

	_, got := guardCommand("ls", "bash", []guardHook{{path: first}, {path: second}}, nil)

	if want := "error: guard hook " + first + " failed: exit status 3"; got != want {
		t.Errorf("guardCommand = %q, want %q", got, want)
	}
	if hookMarkerExists(t, secondMarker) {
		t.Errorf("hook %s ran although an earlier hook already refused", second)
	}
}

func TestGuardCommandNoHooksPaysNothing(t *testing.T) {
	for _, hooks := range [][]guardHook{nil, {}} {
		if _, got := guardCommand("ls", "bash", hooks, nil); got != "" {
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

	_, got := guardCommand("ls", "bash", []guardHook{{path: script, tools: []string{"Write"}}}, nil)

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

	_, got := guardCommand("ls", "bash", []guardHook{{path: script, tools: []string{}}}, nil)

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

	_, got := guardCommand("ls", "bash", []guardHook{{path: first}, {path: second}}, nil)

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

			if _, got := guardCommand(tc.command, "bash", []guardHook{{path: script}}, nil); got != "" {
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

			_, got := guardCommand("ls", "bash", []guardHook{{path: script}}, stderr)

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

	_, got := guardCommand("ls", "bash", []guardHook{{path: first}, {path: second}}, nil)

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

	if _, got := guardCommand("ls", "bash", []guardHook{{path: script}}, nil); got != "" {
		t.Errorf("guardCommand = %q, want the empty string (proceed) for output at the cap", got)
	}
}

func TestGuardCommandRefusesWhenJqIsUnavailable(t *testing.T) {
	// The marker is written with bash builtins (`: >`), because an emptied PATH
	// would make an external touch fail and a failed hook run would refuse for
	// the wrong reason — this has to prove the script was never started.
	dir := t.TempDir()
	marker := filepath.Join(dir, "ran")
	script := writeGuardHookScript(t, "approve.sh",
		`: > `+marker+`
echo '{"hookSpecificOutput":{"permissionDecision":"allow","permissionDecisionReason":"fine"}}'`)
	// Setenv before the call: LookPath reads the current env when guardCommand
	// asks, and the fixture's absolute shebang keeps it startable either way.
	t.Setenv("PATH", "")

	_, got := guardCommand("ls", "bash", []guardHook{{path: script}}, nil)

	if !strings.HasPrefix(got, "error: ") {
		t.Errorf("guardCommand = %q, want an error: prefixed refusal when jq is missing", got)
	}
	if !strings.Contains(got, "jq") {
		t.Errorf("refusal %q does not name jq", got)
	}
	if hookMarkerExists(t, marker) {
		t.Errorf("hook %s ran although jq was unavailable", script)
	}
}

func TestGuardCommandNoHooksDoesNotNeedJq(t *testing.T) {
	// The fast path stays first: a run with nothing to guard must not be refused
	// for a missing jq, or an unguarded deployment breaks over a tool it never
	// asked for.
	t.Setenv("PATH", "")

	if _, got := guardCommand("ls", "bash", nil, nil); got != "" {
		t.Errorf("guardCommand with no hooks = %q, want the empty string", got)
	}
}

func TestGuardCommandAppliesARewrite(t *testing.T) {
	// A rewriting hook's whole job is to hand back the command that runs, so the
	// chain's answer is that command and not the one the caller asked about.
	script := writeGuardHookScript(t, "sanitize.sh",
		`echo '{"hookSpecificOutput":{"permissionDecision":"allow","updatedInput":{"command":"echo scrubbed"}}}'`)

	ran, refusal := guardCommand("echo 'secret token abc'", "bash", []guardHook{{path: script, rewrites: true}}, nil)

	if refusal != "" {
		t.Fatalf("guardCommand refusal = %q, want the empty string", refusal)
	}
	if ran != "echo scrubbed" {
		t.Errorf("command to run = %q, want %q", ran, "echo scrubbed")
	}
}

func TestGuardCommandHandsTheRewrittenCommandToTheNextHook(t *testing.T) {
	// A rewrite is the command the guarded run is about, so the hooks after the
	// rewriter are asked about the rewritten command: a chain that re-checks the
	// original text lets a hook approve words nobody is going to execute.
	dir := t.TempDir()
	received := filepath.Join(dir, "second-request.json")
	first := writeGuardHookScript(t, "sanitize.sh",
		`echo '{"hookSpecificOutput":{"permissionDecision":"allow","updatedInput":{"command":"echo scrubbed"}}}'`)
	second := writeGuardHookScript(t, "record.sh", "cat > "+received)

	ran, refusal := guardCommand("echo 'secret token abc'", "bash", []guardHook{{path: first, rewrites: true}, {path: second}}, nil)

	if refusal != "" {
		t.Fatalf("guardCommand refusal = %q, want the empty string", refusal)
	}
	if ran != "echo scrubbed" {
		t.Errorf("command to run = %q, want %q", ran, "echo scrubbed")
	}
	raw, err := os.ReadFile(received)
	if err != nil {
		t.Fatalf("read what the second hook received: %v", err)
	}
	if want := `{"tool_name":"Bash","tool_input":{"command":"echo scrubbed"}}`; string(raw) != want {
		t.Errorf("second hook received %s, want exactly %s", raw, want)
	}
}

func TestGuardCommandRefusesWhenARewritingHookReturnsNoRewrite(t *testing.T) {
	// A hook the manifest declares a rewriter is the only thing standing between
	// the Model's text and the shell, so an answer that rewrites nothing is not a
	// pass: the Driver cannot tell "nothing to sanitize" from a hook that stopped
	// answering, and it refuses rather than run the original command unguarded.
	for _, tc := range []struct{ name, body string }{
		{"silent", `:`},
		{"allow without a rewrite", `echo '{"hookSpecificOutput":{"permissionDecision":"allow"}}'`},
		{"empty updatedInput command", `echo '{"hookSpecificOutput":{"permissionDecision":"allow","updatedInput":{"command":""}}}'`},
		{"updatedInput is null", `echo '{"hookSpecificOutput":{"permissionDecision":"allow","updatedInput":null}}'`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			script := writeGuardHookScript(t, "sanitize.sh", tc.body)

			ran, refusal := guardCommand("ls", "bash", []guardHook{{path: script, rewrites: true}}, nil)

			want := "error: guard hook " + script + " is a rewriting hook and returned no rewrite, so the command is blocked"
			if refusal != want {
				t.Errorf("guardCommand refusal = %q, want %q", refusal, want)
			}
			if ran != "" {
				t.Errorf("command to run = %q, want the empty string: a refused call runs nothing", ran)
			}
		})
	}
}

func TestGuardCommandARewritingHookOwesNothingForAnEmptyCommand(t *testing.T) {
	// The rewrite debt is owed against a command. There is nothing to sanitize in
	// the empty string, so a rewriting hook that answers nothing there passes —
	// refusing would block a call that cannot do anything.
	for _, body := range []string{`:`, `echo '{"hookSpecificOutput":{"permissionDecision":"allow"}}'`} {
		script := writeGuardHookScript(t, "sanitize.sh", body)

		ran, refusal := guardCommand("", "bash", []guardHook{{path: script, rewrites: true}}, nil)

		if refusal != "" {
			t.Errorf("guardCommand refusal = %q, want the empty string: an empty command owes no rewrite", refusal)
		}
		if ran != "" {
			t.Errorf("command to run = %q, want the empty string", ran)
		}
	}
}

func TestGuardCommandAppliesARewriteFromAHookThatDidNotDeclareIt(t *testing.T) {
	// rewrites is the promise that an answer arrives, not permission to rewrite:
	// honouring an updatedInput only from a declared rewriter would throw away a
	// sanitisation the Driver was actually handed.
	script := writeGuardHookScript(t, "quiet.sh",
		`echo '{"hookSpecificOutput":{"permissionDecision":"allow","updatedInput":{"command":"echo sanitised"}}}'`)

	ran, refusal := guardCommand("echo 'secret token abc'", "bash", []guardHook{{path: script}}, nil)

	if refusal != "" {
		t.Fatalf("guardCommand refusal = %q, want the empty string", refusal)
	}
	if ran != "echo sanitised" {
		t.Errorf("command to run = %q, want %q", ran, "echo sanitised")
	}
}

func TestGuardCommandDenialBeatsTheRewriteDebt(t *testing.T) {
	// A rewriting hook that denies has answered the question that matters, so the
	// refusal is its stated reason and not a complaint about the rewrite it was
	// never going to supply for a command it blocks.
	script := writeGuardHookScript(t, "sanitize.sh",
		`echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"leave it alone"}}'`)

	ran, refusal := guardCommand("ls", "bash", []guardHook{{path: script, rewrites: true}}, nil)

	if want := "error: guard hook " + script + " denied this command: leave it alone"; refusal != want {
		t.Errorf("guardCommand refusal = %q, want %q", refusal, want)
	}
	if ran != "" {
		t.Errorf("command to run = %q, want the empty string", ran)
	}
}

// summarizeToolOutput is the post-call half of the Hook Manifest. Its failure
// rule is the mirror of guardCommand's: everything the summary hook does wrong
// leaves the original Tool Result standing, because the Driver already has a
// truthful answer and a broken summariser is no reason to lose it.
func TestSummarizeToolOutputReplacesTheToolResult(t *testing.T) {
	script := writeGuardHookScript(t, "summarize.sh",
		`echo '{"hookSpecificOutput":{"updatedToolOutput":"SUMMARY"}}'`)

	got := summarizeToolOutput("bash", "ls -l", "file listing\nexit code: 0", &summaryHook{path: script}, nil)

	if got != "SUMMARY" {
		t.Errorf("summarizeToolOutput = %q, want the hook's replacement SUMMARY", got)
	}
}

func TestSummarizeToolOutputHandsTheHookTheCommandAndTheToolResult(t *testing.T) {
	stdin := filepath.Join(t.TempDir(), "stdin.json")
	script := writeGuardHookScript(t, "summarize.sh",
		`cat > `+stdin+`
echo '{"hookSpecificOutput":{"updatedToolOutput":"SUMMARY"}}'`)

	if got := summarizeToolOutput("bash", "ls -l", summarizeFixtureOutput, &summaryHook{path: script}, nil); got != "SUMMARY" {
		t.Fatalf("summarizeToolOutput = %q, want SUMMARY", got)
	}

	// The payload is the contract a hook author codes against: the protocol's tool
	// name, the command that actually ran, and the finished Tool Result — exit-code
	// line and all — as one JSON string.
	want := `{"tool_name":"Bash","tool_input":{"command":"ls -l"},"tool_response":"` + summarizeFixtureJSON + `"}`
	got, err := os.ReadFile(stdin)
	if err != nil {
		t.Fatalf("read the payload the hook was handed: %v", err)
	}
	if string(got) != want {
		t.Errorf("hook stdin = %s, want %s", got, want)
	}
}

// summarizeFixtureOutput is one finished Tool Result; summarizeFixtureJSON is
// the same text as it appears inside the payload's JSON string.
const (
	summarizeFixtureOutput = "total 4\ndrwxr-xr-x 2 root root 4096 Jan 1 00:00 dir\nexit code: 0"
	summarizeFixtureJSON   = `total 4\ndrwxr-xr-x 2 root root 4096 Jan 1 00:00 dir\nexit code: 0`
)

func TestSummarizeToolOutputKeepsTheOriginalWhenTheSummaryCannotBeHad(t *testing.T) {
	// The fail-open half: every one of these is a summary the Driver never got, and
	// a Tool Result it did get beats speculation about what the summary would have
	// said. guardCommand refuses the same list outright — the two callers disagree
	// because their defaults differ, not because the anomalies differ.
	for _, tc := range []struct {
		name string
		body string
	}{
		{"hook exits non-zero", "echo '{\"hookSpecificOutput\":{\"updatedToolOutput\":\"SUMMARY\"}}'\nexit 1"},
		{"hook writes unparsable output", "echo 'not json at all'"},
		{"hook answers JSON with no updatedToolOutput", `echo '{"hookSpecificOutput":{"permissionDecision":"allow"}}'`},
		{"hook answers an empty updatedToolOutput", `echo '{"hookSpecificOutput":{"updatedToolOutput":""}}'`},
		{"hook answers a null hookSpecificOutput", `echo '{"hookSpecificOutput":null}'`},
		{"hook writes nothing", "true"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			hook := &summaryHook{path: writeGuardHookScript(t, "summarize.sh", tc.body)}

			got := summarizeToolOutput("bash", "ls -l", "file listing\nexit code: 0", hook, nil)

			if got != "file listing\nexit code: 0" {
				t.Errorf("summarizeToolOutput = %q, want the original Tool Result unchanged", got)
			}
		})
	}
}

func TestSummarizeToolOutputKeepsTheOriginalWithoutASummaryHook(t *testing.T) {
	// The unconfigured case is the common one, and it is the same answer: the
	// command's own output, untouched.
	got := summarizeToolOutput("bash", "ls", "file listing\nexit code: 0", nil, nil)

	if got != "file listing\nexit code: 0" {
		t.Errorf("summarizeToolOutput = %q, want the original Tool Result", got)
	}
}

func TestSummarizeToolOutputKeepsTheOriginalWhenTheScriptCannotBeStarted(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "never-installed.sh")

	got := summarizeToolOutput("bash", "ls", "file listing\nexit code: 0", &summaryHook{path: missing}, nil)

	if got != "file listing\nexit code: 0" {
		t.Errorf("summarizeToolOutput = %q, want the original Tool Result: a summary hook that is not on disk loses the summary, not the answer", got)
	}
}

func TestSummarizeToolOutputKeepsTheOriginalWhenTheHookHangs(t *testing.T) {
	old := guardHookTimeout
	guardHookTimeout = 100 * time.Millisecond
	defer func() { guardHookTimeout = old }()
	script := writeGuardHookScript(t, "hang.sh", "sleep 30")

	got := summarizeToolOutput("bash", "ls", "file listing\nexit code: 0", &summaryHook{path: script}, nil)

	if got != "file listing\nexit code: 0" {
		t.Errorf("summarizeToolOutput = %q, want the original Tool Result: a fired bound loses the summary, not the answer", got)
	}
}

func TestSummarizeToolOutputSkipsAToolTheEntryDoesNotName(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "ran")
	script := writeGuardHookScript(t, "summarize.sh",
		`touch `+marker+`
echo '{"hookSpecificOutput":{"updatedToolOutput":"SUMMARY"}}'`)

	got := summarizeToolOutput("bash", "ls", "file listing\nexit code: 0", &summaryHook{path: script, tools: []string{"Write"}}, nil)

	if got != "file listing\nexit code: 0" {
		t.Errorf("summarizeToolOutput = %q, want the original for a tool the entry does not name", got)
	}
	if hookMarkerExists(t, marker) {
		t.Errorf("summary script %s ran for a tool its entry does not name", script)
	}
}

func TestSummarizeToolOutputCapsTheReplacementWithTheToolResultBound(t *testing.T) {
	// The same floor runBashTool applies, so a hook summarising upwards cannot hand
	// the Model a longer result than the command itself was allowed.
	old := bashToolMaxOutput
	bashToolMaxOutput = 64
	defer func() { bashToolMaxOutput = old }()
	long := strings.Repeat("s", 200)
	script := writeGuardHookScript(t, "summarize.sh",
		`printf '{"hookSpecificOutput":{"updatedToolOutput":"`+long+`"}}'`)

	got := summarizeToolOutput("bash", "ls", "file listing\nexit code: 0", &summaryHook{path: script}, nil)

	if !strings.HasPrefix(got, truncatedOutputMarker(bashToolMaxOutput)+"\n") {
		t.Errorf("summary replacement = %q, want it capped behind %q", got, truncatedOutputMarker(bashToolMaxOutput))
	}
	if tail := long[len(long)-bashToolMaxOutput:]; !strings.HasSuffix(got, tail) {
		t.Errorf("summary replacement = %q, want the last %d bytes of the hook's summary kept", got, bashToolMaxOutput)
	}
}
