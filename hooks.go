package main

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"sort"
)

// guardHook is one pre-call Guard Hook entry of a Hook Manifest: the script to
// run, the hook-protocol tool names it applies to, and whether it rewrites the
// tool input.
//
// rewrites is read and recorded but not enforced until issue #22 owns the
// rewrite path and the post-call summary hook — the same precedent this branch
// already sets for an Agent File's tools and spawns, which are parsed, recorded,
// and only later acted on. Nothing in this file executes a hook.
type guardHook struct {
	path     string
	tools    []string
	rewrites bool
}

// knownManifestFields and knownHookFields are the whole Hook Manifest schema,
// top level and per entry. Deliberately not more: the run order is the file
// order, so a "priority" or "enabled" key would be a second, contradicted
// answer to when a hook runs.
var (
	knownManifestFields = []string{"pre"}
	knownHookFields     = []string{"path", "tools", "rewrites"}
)

// loadHookManifest reads exactly one Hook Manifest and returns its pre-call
// Guard Hooks in manifest order — the run order is the file order and nothing
// here sorts them.
//
// Unlike loadDriverConfig this takes an explicit path and never resolves a
// default: which file the manifest is stays the caller's decision (issue #22's,
// once the Driver Config's hook_manifest field is wired up). It writes nothing
// anywhere, and an absent "pre" is the only silence it accepts — a file that
// cannot be read, a top-level value that is not an object (a top-level null or
// an empty file included, which json would otherwise hand back as a nil map and
// read as "no hooks"), and any unknown field are all errors naming the path.
func loadHookManifest(path string) ([]guardHook, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("hook manifest %s: cannot read: %v", path, err)
	}
	// Checked before decoding for the reason loadDriverConfig gives: the decode
	// error alone cannot tell "no file content" from "content of the wrong
	// shape", and the second reading must not quietly mean no hooks.
	if kind := jsonKind(raw); kind != "object" {
		return nil, fmt.Errorf("hook manifest %s: malformed JSON: top-level value is %s, want an object", path, kind)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, fmt.Errorf("hook manifest %s: malformed JSON: %v", path, err)
	}
	// Sorted before the unknown-key scan so a file with several complaints always
	// reports the same one: map iteration order would otherwise pick at random.
	for _, k := range sortedKeys(fields) {
		if !slices.Contains(knownManifestFields, k) {
			return nil, fmt.Errorf("hook manifest %s: unknown field %q", path, k)
		}
	}
	rawPre, ok := fields["pre"]
	if !ok {
		// A manifest that says nothing is a valid manifest: no pre-call hooks.
		return nil, nil
	}
	if kind := jsonKind(rawPre); kind != "array" {
		return nil, fmt.Errorf("hook manifest %s: field %q must be an array, got %s", path, "pre", kind)
	}
	var entries []json.RawMessage
	if err := json.Unmarshal(rawPre, &entries); err != nil {
		return nil, fmt.Errorf("hook manifest %s: field %q: %v", path, "pre", err)
	}
	hooks := make([]guardHook, 0, len(entries))
	for i, entry := range entries {
		hook, err := parseGuardHook(path, i, entry)
		if err != nil {
			return nil, err
		}
		hooks = append(hooks, hook)
	}
	return hooks, nil
}

// parseGuardHook decodes one element of "pre". Every complaint carries the
// manifest path and the element's position, because the file is the thing the
// operator has to go and fix and "entry 2" is how they find it.
func parseGuardHook(path string, idx int, raw json.RawMessage) (guardHook, error) {
	at := fmt.Sprintf("pre[%d]", idx)
	if kind := jsonKind(raw); kind != "object" {
		return guardHook{}, fmt.Errorf("hook manifest %s: %s must be an object, got %s", path, at, kind)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return guardHook{}, fmt.Errorf("hook manifest %s: %s: %v", path, at, err)
	}
	for _, k := range sortedKeys(fields) {
		if !slices.Contains(knownHookFields, k) {
			return guardHook{}, fmt.Errorf("hook manifest %s: %s: unknown field %q", path, at, k)
		}
	}
	var hook guardHook
	// The script is the whole entry — a hook with nothing to run is not a hook —
	// so path is required and non-empty, unlike the optional fields after it.
	rawPath, ok := fields["path"]
	if !ok {
		return guardHook{}, fmt.Errorf("hook manifest %s: %s: missing field %q", path, at, "path")
	}
	if kind := jsonKind(rawPath); kind != "string" {
		return guardHook{}, fmt.Errorf("hook manifest %s: %s: field %q must be a string, got %s", path, at, "path", kind)
	}
	if err := json.Unmarshal(rawPath, &hook.path); err != nil {
		return guardHook{}, fmt.Errorf("hook manifest %s: %s: field %q: %v", path, at, "path", err)
	}
	if hook.path == "" {
		return guardHook{}, fmt.Errorf("hook manifest %s: %s: field %q must not be empty", path, at, "path")
	}
	if rawTools, ok := fields["tools"]; ok {
		if kind := jsonKind(rawTools); kind != "array" {
			return guardHook{}, fmt.Errorf("hook manifest %s: %s: field %q must be an array, got %s", path, at, "tools", kind)
		}
		var items []json.RawMessage
		if err := json.Unmarshal(rawTools, &items); err != nil {
			return guardHook{}, fmt.Errorf("hook manifest %s: %s: field %q: %v", path, at, "tools", err)
		}
		for j, item := range items {
			if kind := jsonKind(item); kind != "string" {
				return guardHook{}, fmt.Errorf("hook manifest %s: %s: field %q[%d] must be a string, got %s", path, at, "tools", j, kind)
			}
			var name string
			if err := json.Unmarshal(item, &name); err != nil {
				return guardHook{}, fmt.Errorf("hook manifest %s: %s: field %q[%d]: %v", path, at, "tools", j, err)
			}
			hook.tools = append(hook.tools, name)
		}
	}
	if rawRewrites, ok := fields["rewrites"]; ok {
		if kind := jsonKind(rawRewrites); kind != "boolean" {
			return guardHook{}, fmt.Errorf("hook manifest %s: %s: field %q must be a boolean, got %s", path, at, "rewrites", kind)
		}
		if err := json.Unmarshal(rawRewrites, &hook.rewrites); err != nil {
			return guardHook{}, fmt.Errorf("hook manifest %s: %s: field %q: %v", path, at, "rewrites", err)
		}
	}
	return hook, nil
}

// appliesTo reports whether the hook guards the named tool. An empty tools list
// — absent or written as [] — matches everything: a hook author who listed no
// tools did not opt out of guarding, and a manifest that fails to name its
// tools must fail closed rather than let the tool run unguarded.
//
// The comparison is exact because the manifest speaks the hook protocol's own
// tool names ("Bash"), and translating between those and the Driver's tool
// names is issue #22's mapping concern, not this function's.
func (h guardHook) appliesTo(protocolName string) bool {
	if len(h.tools) == 0 {
		return true
	}
	return slices.Contains(h.tools, protocolName)
}

// sortedKeys gives the unknown-field scans a stable order, for the reason
// loadDriverConfig sorts before complaining.
func sortedKeys(fields map[string]json.RawMessage) []string {
	keys := make([]string, 0, len(fields))
	for k := range fields {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
