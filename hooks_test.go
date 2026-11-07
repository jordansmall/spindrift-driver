package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
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
