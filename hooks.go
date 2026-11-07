package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"slices"
	"sort"
	"time"
)

// guardHookTimeout bounds one Guard Hook run; tests inject a shorter value.
var guardHookTimeout = 5 * time.Second

// guardHookOutputLimit caps how much of one Guard Hook's stdout the Driver
// holds; tests inject a smaller value. The cap is inclusive: exactly this many
// bytes is an answer, one byte more is a hook that flooded instead.
var guardHookOutputLimit = 1 << 20

// guardHook is one pre-call Guard Hook entry of a Hook Manifest: the script to
// run, the hook-protocol tool names it applies to, and whether it rewrites the
// tool input.
//
// rewrites is a promise about the answer, never a gate on applying it: a hook
// carrying the flag that answers without a rewrite has broken that promise and
// the call is refused, while a hook without the flag that hands back updatedInput
// anyway has its rewrite applied just the same.
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

// hookResponse is the part of a hook's stdout this step reads. updatedInput is
// the rewrite path; updatedToolOutput belongs to #22's post-call summary hook,
// so a hook that supplies that changes nothing here.
//
// HookSpecificOutput is a pointer so that JSON null decodes as absent: no
// verdict is a pass, while a hookSpecificOutput of the wrong shape fails the
// decode and is a block. A plain struct field would read both as its zero
// value.
type hookResponse struct {
	HookSpecificOutput *hookDecision `json:"hookSpecificOutput"`
}

type hookDecision struct {
	PermissionDecision       string `json:"permissionDecision"`
	PermissionDecisionReason string `json:"permissionDecisionReason"`
	// UpdatedInput is a pointer for the same reason HookSpecificOutput is: JSON
	// null reads as "the hook offered no rewrite", not as a rewrite to the empty
	// command.
	UpdatedInput *hookToolInput `json:"updatedInput"`
}

// guardCommand runs command through the pre-call Guard Hooks in manifest order
// and returns the command to run together with the refusal to hand back as the
// Tool Result, "" for "no objection". The first refusal stops the chain: a
// command already refused has no reason to be asked of a second hook, and a
// refused call runs nothing, so the command returned with a refusal is the
// empty string. A hook that answers with updatedInput replaces the command the
// rest of the chain sees and the command the caller runs.
//
// A hook's silence is a pass and anything else is a block. Silence means an
// empty or whitespace-only stdout, an absent or null hookSpecificOutput, and
// any decision that is not the literal "deny" — those are hooks with no
// opinion. A script that cannot be started, a run that outlives
// guardHookTimeout, a non-zero exit, stdout past guardHookOutputLimit, and
// stdout that carries no readable verdict are the fail-closed half: the Driver
// refuses rather than accept an answer it never got, since it cannot tell a
// broken hook from an approving one. So is a rewriting hook's answer that rewrote
// nothing — the same doubt about a hook that may have stopped answering halfway.
//
// Every refusal carries the "error: " prefix, which is what turns this string
// into an error Tool Result downstream, and a refusal raised by a run names the
// hook's path so the operator can find the script to ask why — except the
// missing-jq precondition below, which belongs to the run, not to a hook.
func guardCommand(command, driverTool string, hooks []guardHook, stderr io.Writer) (string, string) {
	if len(hooks) == 0 {
		// An unguarded run pays nothing: no marshalling, no fork, no wait — and no
		// jq, since nothing here needs reading. The command runs as asked.
		return command, ""
	}
	if _, err := exec.LookPath("jq"); err != nil {
		// A guard that cannot read its own request must never be trusted as
		// approving: the host's baked hook scripts shell out to jq to parse the
		// stdin this function builds, so without jq a hook exits 0 having inspected
		// nothing and that silence reads as a pass. Refuse before any hook runs,
		// which is also before the command can run. The reason names jq rather than
		// a hook path — jq is a precondition of the guarded run, not of one hook.
		return "", "error: jq is not available, so the guarded command cannot be run"
	}
	protocolName := hookToolName(driverTool)

	for _, hook := range hooks {
		if !hook.appliesTo(protocolName) {
			continue
		}
		// Built per hook rather than once for the chain: a rewrite from an earlier
		// hook is the command the next hook is asked about.
		payload, err := encodeHookRequest(protocolName, command)
		if err != nil {
			// Fail closed like a hook that never answered: a request the Driver could
			// not build is a command no hook was ever asked about, so running it would
			// be running it unguarded.
			return "", fmt.Sprintf("error: guard hook request could not be built: %v", err)
		}
		// The run is judged before the verdict: a hook that never answered blocks
		// the call on its own, with nothing left to parse.
		stdout, refusal := runGuardHook(hook, payload, stderr)
		var answer hookResponse
		if refusal == "" {
			answer, refusal = hookVerdict(hook, stdout)
		}
		if refusal == "" {
			// Judged only after the verdicts, so a rewriting hook that denies is
			// refused for denying rather than for the rewrite it withheld.
			command, refusal = hookRewrite(hook, command, answer)
		}
		if refusal != "" {
			return "", refusal
		}
	}
	return command, ""
}

// hookRewrite decides what the chain runs next once a hook raised no refusal:
// the non-empty updatedInput.command the hook handed back, or command itself.
// A hook the manifest declares a rewriter owes that answer for any command worth
// rewriting, and withholding it refuses the call — the Driver cannot tell "there
// was nothing to sanitize" from a hook that stopped answering, and running the
// original text would be running it past the one hook standing in its way. The
// empty command is the exception: there is nothing in it to rewrite, so an
// unrewritten empty string is the answer the hook was owed.
//
// The rewrite is applied whether or not hook.rewrites is set; the flag only
// decides who is refused for not supplying one.
func hookRewrite(hook guardHook, command string, answer hookResponse) (string, string) {
	var updated *hookToolInput
	if answer.HookSpecificOutput != nil {
		updated = answer.HookSpecificOutput.UpdatedInput
	}
	if updated != nil && updated.Command != "" {
		return updated.Command, ""
	}
	if hook.rewrites && command != "" {
		return "", fmt.Sprintf("error: guard hook %s is a rewriting hook and returned no rewrite, so the command is blocked", hook.path)
	}
	return command, ""
}

// encodeHookRequest builds the stdin one hook is handed for command. Encoder
// rather than Marshal, for events.go's reason: markup in a command must reach
// the hook literally instead of as \u003c escapes. Encode's trailing newline is
// the encoder's doing, not part of the payload, so it comes off here.
func encodeHookRequest(protocolName, command string) ([]byte, error) {
	var request bytes.Buffer
	enc := json.NewEncoder(&request)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(hookRequest{ToolName: protocolName, ToolInput: hookToolInput{Command: command}}); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(request.Bytes(), []byte("\n")), nil
}

// runGuardHook executes one hook and returns its stdout, or, when the run
// itself failed, the refusal that blocks the call in place of an answer.
func runGuardHook(hook guardHook, payload []byte, stderr io.Writer) (stdout []byte, refusal string) {
	ctx, cancel := context.WithTimeout(context.Background(), guardHookTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, hook.path)
	// The script is what the host baked — shebang, mode bits and all — so it is
	// exec'd directly. A shell in between would add a second quoting layer over
	// text the hook is handed as data, and a manifest path is not a command.
	cmd.Stdin = bytes.NewReader(payload)
	out := &cappedWriter{limit: guardHookOutputLimit}
	cmd.Stdout = out
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
	// WaitDelay for the reason runBashTool gives: a grandchild the hook detached
	// can survive the kill holding the output pipes, and Wait would block on
	// their EOF forever — a fired bound that never produces a refusal.
	cmd.WaitDelay = time.Second
	if err := cmd.Start(); err != nil {
		return nil, fmt.Sprintf("error: guard hook %s could not be run: %v", hook.path, err)
	}
	err := cmd.Wait()
	if ctx.Err() != nil {
		// Judged before err, so the kill the bound fired reads as a timeout and not
		// as the signal death it was. The accepted race runBashTool documents
		// applies here too: a hook that finished between Wait and this check is
		// reported as timed out.
		return nil, fmt.Sprintf("error: guard hook %s timed out after %s, so the command is blocked", hook.path, guardHookTimeout)
	}
	if err != nil {
		// Includes a death on a signal, which exec words as "signal: …" — the
		// hook stopped without a verdict either way.
		return nil, fmt.Sprintf("error: guard hook %s failed: %v", hook.path, err)
	}
	if out.truncated {
		// Judged before any verdict is read out of the kept prefix: what was thrown
		// away may have been a deny, and an answer the Driver only half received is
		// not an answer.
		return nil, fmt.Sprintf("error: guard hook %s wrote more than %d bytes of output, so the command is blocked", hook.path, out.limit)
	}
	return out.buf.Bytes(), ""
}

// cappedWriter keeps at most limit bytes of what is written through it and
// records that the write went over. exec needs an io.Writer for a child's
// stdout and the stdlib has no bounded one.
//
// Write always reports the whole slice as consumed and never returns an error:
// an error here reaches exec.Cmd.Wait, and the run would be refused as a
// short write or a failed hook instead of by the message naming the output
// bound. The truncation flag is what the caller judges.
type cappedWriter struct {
	buf       bytes.Buffer
	limit     int
	truncated bool
}

func (w *cappedWriter) Write(p []byte) (int, error) {
	if room := w.limit - w.buf.Len(); len(p) > room {
		w.truncated = true
		if room > 0 {
			w.buf.Write(p[:room])
		}
		return len(p), nil
	}
	w.buf.Write(p)
	return len(p), nil
}

// hookVerdict turns one hook's stdout into the refusal it asks for, or "" when
// it raised no objection. The decoding happens here and the decoded answer comes
// back with the verdict, so hookRewrite reads the same parse rather than a
// second one: an answer that never got decoded — silence, or stdout this could
// not read — comes back as the zero hookResponse alongside its own verdict.
func hookVerdict(hook guardHook, stdout []byte) (hookResponse, string) {
	if len(bytes.TrimSpace(stdout)) == 0 {
		// Silence: a hook that said nothing, which is a pass and not a verdict the
		// Driver has to be able to read.
		return hookResponse{}, ""
	}
	var response hookResponse
	if err := json.Unmarshal(stdout, &response); err != nil {
		return hookResponse{}, fmt.Sprintf("error: guard hook %s returned unparsable output", hook.path)
	}
	if response.HookSpecificOutput == nil {
		return response, ""
	}
	if response.HookSpecificOutput.PermissionDecision != "deny" {
		return response, ""
	}
	reason := response.HookSpecificOutput.PermissionDecisionReason
	if reason == "" {
		reason = "the hook gave no reason"
	}
	return response, fmt.Sprintf("error: guard hook %s denied this command: %s", hook.path, reason)
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
