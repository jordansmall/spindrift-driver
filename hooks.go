package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
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
// and only later acted on. guardCommand below does execute the pre-call chain,
// but only its deny verdict acts.
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

// hookToolName is the Driver's tool name in the hook protocol's vocabulary, the
// one the manifest is written in; #22 extends this map when the Edit and Write
// Tools land.
func hookToolName(driverTool string) string {
	if driverTool == "bash" {
		return "Bash"
	}
	return driverTool
}

// hookRequest and hookToolInput are the pre-call payload a hook reads on stdin,
// in the protocol's tool vocabulary: {"tool_name":"Bash","tool_input":{"command":…}}.
type hookRequest struct {
	ToolName  string        `json:"tool_name"`
	ToolInput hookToolInput `json:"tool_input"`
}

type hookToolInput struct {
	Command string `json:"command"`
}

// hookResponse is the part of a hook's stdout this step reads. updatedInput —
// the rewrite path — and updatedToolOutput belong to #22, which owns rewriting
// and the post-call summary hook, so a hook that supplies them changes nothing
// here.
type hookResponse struct {
	HookSpecificOutput struct {
		PermissionDecision       string `json:"permissionDecision"`
		PermissionDecisionReason string `json:"permissionDecisionReason"`
	} `json:"hookSpecificOutput"`
}

// guardCommand runs command through the pre-call Guard Hooks in manifest order
// and returns the refusal to hand back as the Tool Result, or "" when every
// applicable hook let the call through. The first refusal stops the chain: a
// command already refused has no reason to be asked of a second hook.
//
// Only a literal "deny" decision refuses. Everything else — "allow", "ask", no
// decision, no hookSpecificOutput, and stdout that is empty or all whitespace,
// which is a hook's silence rather than a verdict — proceeds, because a hook
// that did not say no did not say no. The remaining silences are not verdicts
// either yet: non-zero exit, unparsable stdout, a script that cannot be started,
// and a hook that hangs are slice 4b's fail-closed cases, marked at each site
// below.
func guardCommand(command, driverTool string, hooks []guardHook, stderr io.Writer) string {
	if len(hooks) == 0 {
		// An unguarded run pays nothing: no marshalling, no fork, no wait.
		return ""
	}
	protocolName := hookToolName(driverTool)
	var request bytes.Buffer
	// Encoder rather than Marshal, for events.go's reason: markup in a command
	// must reach the hook literally instead of as \u003c escapes, and writing into
	// a bytes.Buffer cannot fail — Encode's trailing newline is the encoder's
	// doing, not part of the payload.
	enc := json.NewEncoder(&request)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(hookRequest{ToolName: protocolName, ToolInput: hookToolInput{Command: command}})
	payload := bytes.TrimSuffix(request.Bytes(), []byte("\n"))

	for _, hook := range hooks {
		if !hook.appliesTo(protocolName) {
			continue
		}
		cmd := exec.Command(hook.path)
		// The script is what the host baked — shebang, mode bits and all — so it is
		// exec'd directly. A shell in between would add a second quoting layer over
		// text the hook is handed as data, and a manifest path is not a command.
		cmd.Stdin = bytes.NewReader(payload)
		var stdout bytes.Buffer
		cmd.Stdout = &stdout
		// A hook's own noise goes to the operator's progress copy, never to the wire:
		// stdout carries only event lines, so a stray byte from a hook would be a
		// malformed event. Hence the buffer for stdout — the child is never handed
		// the parent's stdout fd — and nil here meaning there is no copy to write to,
		// so the noise is dropped rather than inherited.
		if stderr == nil {
			cmd.Stderr = io.Discard
		} else {
			cmd.Stderr = stderr
		}
		// No cmd.Dir and no cmd.Env, following runBashTool: the hook judges the
		// command in the same cwd and environment the command is about to run in.
		if err := cmd.Start(); err != nil {
			// 4b: a hook that never ran fails closed here. Until then it has no say.
			continue
		}
		// 4b reads the exit status and enforces a timeout here; until then the
		// status is not consulted, so it is deliberately not even kept.
		_ = cmd.Wait()
		if len(bytes.TrimSpace(stdout.Bytes())) == 0 {
			continue
		}
		var response hookResponse
		if err := json.Unmarshal(stdout.Bytes(), &response); err != nil {
			// 4b: unparsable stdout fails closed here.
			continue
		}
		if response.HookSpecificOutput.PermissionDecision != "deny" {
			continue
		}
		reason := response.HookSpecificOutput.PermissionDecisionReason
		if reason == "" {
			reason = "the hook gave no reason"
		}
		// The "error: " prefix is what turns this string into an error Tool Result
		// downstream, and the path is how the operator finds the script to ask why.
		return fmt.Sprintf("error: guard hook %s denied this command: %s", hook.path, reason)
	}
	return ""
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
