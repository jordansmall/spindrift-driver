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
	"syscall"
	"time"
)

// hookLimits are the two bounds one Guard chain holds every script it runs to:
// how long a run may take before its silence counts as no answer, and how much of
// its stdout the Driver keeps. A chain carries its own pair from the site that
// built it, so a test that needs a different bound builds a chain with it instead
// of editing state every other chain in the binary is reading.
//
// outputLimit is inclusive: exactly that many bytes is an answer, one byte more is
// a hook that flooded instead. The bound judges before any verdict is read out of
// the kept prefix, because a verdict the Driver only half received is not a
// verdict — the discarded tail may have held the denial.
type hookLimits struct {
	timeout     time.Duration
	outputLimit int
}

// newHookLimits states the production pair once, for the site that builds the
// chain a run guards with: 5s for one hook script and 1 MiB of its stdout.
func newHookLimits() hookLimits {
	return hookLimits{timeout: 5 * time.Second, outputLimit: 1 << 20}
}

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

// summaryHook is the manifest's one post-call entry: the script that summarises
// a Tool Result and the hook-protocol tool names its result is worth rewriting.
//
// Deliberately its own type rather than a guardHook without a rewrites flag: a
// summary entry is not a link in the pre-call chain, and keeping it separate
// means the fail-open semantics it will carry stay local to it instead of
// leaking into the chain's fail-closed rules.
type summaryHook struct {
	path  string
	tools []string
}

// guardChain is one Hook Manifest read into the value the Driver guards a call
// with: the pre-call Guard Hooks in run order and the one post-call summary
// hook, nil when the file names none.
//
// A struct rather than a second return value so a future field does not grow the
// return tuple again. It is held through a pointer the way runTally is, because a
// nil *guardChain is a value of its own: the run that configured no Hook Manifest
// carries no chain at all, and every guard method answers it without a jq lookup
// or a fork.
type guardChain struct {
	// limits bounds every script this chain runs; it arrives with the chain rather
	// than as state the whole binary shares, so no other run can re-bound it.
	limits  hookLimits
	pre     []guardHook
	summary *summaryHook
}

// knownManifestFields and knownHookFields are the whole Hook Manifest schema,
// top level and per pre-call entry; knownSummaryFields is the per-entry schema
// of the one post-call entry. Deliberately not more: the run order is the file
// order, so a "priority" or "enabled" key would be a second, contradicted
// answer to when a hook runs.
//
// A summary entry has no "rewrites" key: whether the post-call script rewrites
// the Tool Result is what it exists for, so the flag would only let a manifest
// switch the feature off in a way nothing reads.
var (
	knownManifestFields = []string{"pre", "post"}
	knownHookFields     = []string{"path", "tools", "rewrites"}
	knownSummaryFields  = []string{"path", "tools"}
)

// loadHookManifest reads exactly one Hook Manifest and returns the guard chain
// it describes: the pre-call Guard Hooks in manifest order — the run order is the
// file order and nothing here sorts them — together with the one post-call summary
// hook, nil when the file names none.
//
// Unlike loadDriverConfig this takes an explicit path and never resolves a
// default: which file the manifest is stays the caller's decision (issue #22's,
// once the Driver Config's hook_manifest field is wired up). It writes nothing
// anywhere, and an absent "pre" or "post" is the only silence it accepts — a
// file that cannot be read, a top-level value that is not an object (a top-level
// null or an empty file included, which json would otherwise hand back as a nil
// map and read as "no hooks"), and any unknown field are all errors naming the
// path.
//
// The limits pair is stamped onto the chain returned, so the bounds every script it
// runs are fixed where the chain is built.
func loadHookManifest(path string, limits hookLimits) (guardChain, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return guardChain{}, fmt.Errorf("hook manifest %s: cannot read: %v", path, err)
	}
	// Checked before decoding for the reason loadDriverConfig gives: the decode
	// error alone cannot tell "no file content" from "content of the wrong
	// shape", and the second reading must not quietly mean no hooks.
	if kind := jsonKind(raw); kind != "object" {
		return guardChain{}, fmt.Errorf("hook manifest %s: malformed JSON: top-level value is %s, want an object", path, kind)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return guardChain{}, fmt.Errorf("hook manifest %s: malformed JSON: %v", path, err)
	}
	// Sorted before the unknown-key scan so a file with several complaints always
	// reports the same one: map iteration order would otherwise pick at random.
	for _, k := range sortedKeys(fields) {
		if !slices.Contains(knownManifestFields, k) {
			return guardChain{}, fmt.Errorf("hook manifest %s: unknown field %q", path, k)
		}
	}
	// Both keys are optional, and each is read independently: a manifest that
	// names only post-call work still gets its summary hook parsed, so neither
	// key's absence may short-circuit the other.
	manifest := guardChain{limits: limits}
	if rawPre, ok := fields["pre"]; ok {
		if kind := jsonKind(rawPre); kind != "array" {
			return guardChain{}, fmt.Errorf("hook manifest %s: field %q must be an array, got %s", path, "pre", kind)
		}
		var entries []json.RawMessage
		if err := json.Unmarshal(rawPre, &entries); err != nil {
			return guardChain{}, fmt.Errorf("hook manifest %s: field %q: %v", path, "pre", err)
		}
		manifest.pre = make([]guardHook, 0, len(entries))
		for i, entry := range entries {
			hook, err := parseGuardHook(path, i, entry)
			if err != nil {
				return guardChain{}, err
			}
			manifest.pre = append(manifest.pre, hook)
		}
	}
	if rawPost, ok := fields["post"]; ok {
		summary, err := parseSummaryHook(path, rawPost)
		if err != nil {
			return guardChain{}, err
		}
		manifest.summary = &summary
	}
	return manifest, nil
}

// parseSummaryHook decodes the "post" object. Its complaints carry the manifest
// path and the field name "post" where parseGuardHook puts an entry's position:
// there is only ever one of them, so the name is how the operator finds it.
func parseSummaryHook(path string, raw json.RawMessage) (summaryHook, error) {
	at := "post"
	if kind := jsonKind(raw); kind != "object" {
		// field "post" rather than bare post for the shape complaint: post is a
		// top-level key, so its own wording follows the one "pre" uses.
		return summaryHook{}, fmt.Errorf("hook manifest %s: field %q must be an object, got %s", path, at, kind)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return summaryHook{}, fmt.Errorf("hook manifest %s: %s: %v", path, at, err)
	}
	for _, k := range sortedKeys(fields) {
		if !slices.Contains(knownSummaryFields, k) {
			return summaryHook{}, fmt.Errorf("hook manifest %s: %s: unknown field %q", path, at, k)
		}
	}
	var summary summaryHook
	// The script is the whole entry here too: a summary hook with nothing to run
	// is a manifest promising a summary no script will write.
	rawPath, ok := fields["path"]
	if !ok {
		return summaryHook{}, fmt.Errorf("hook manifest %s: %s: missing field %q", path, at, "path")
	}
	if kind := jsonKind(rawPath); kind != "string" {
		return summaryHook{}, fmt.Errorf("hook manifest %s: %s: field %q must be a string, got %s", path, at, "path", kind)
	}
	if err := json.Unmarshal(rawPath, &summary.path); err != nil {
		return summaryHook{}, fmt.Errorf("hook manifest %s: %s: field %q: %v", path, at, "path", err)
	}
	if summary.path == "" {
		return summaryHook{}, fmt.Errorf("hook manifest %s: %s: field %q must not be empty", path, at, "path")
	}
	if rawTools, ok := fields["tools"]; ok {
		tools, err := parseHookTools(path, at, rawTools)
		if err != nil {
			return summaryHook{}, err
		}
		summary.tools = tools
	}
	return summary, nil
}

// parseHookTools decodes an entry's "tools" value for either kind of manifest
// entry, naming the offending element by position. An absent key never reaches
// this: an empty list — written [] or left out — is the every-tool answer, not a
// decode of nothing.
func parseHookTools(path, at string, raw json.RawMessage) ([]string, error) {
	if kind := jsonKind(raw); kind != "array" {
		return nil, fmt.Errorf("hook manifest %s: %s: field %q must be an array, got %s", path, at, "tools", kind)
	}
	var items []json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("hook manifest %s: %s: field %q: %v", path, at, "tools", err)
	}
	var tools []string
	for j, item := range items {
		if kind := jsonKind(item); kind != "string" {
			return nil, fmt.Errorf("hook manifest %s: %s: field %q[%d] must be a string, got %s", path, at, "tools", j, kind)
		}
		var name string
		if err := json.Unmarshal(item, &name); err != nil {
			return nil, fmt.Errorf("hook manifest %s: %s: field %q[%d]: %v", path, at, "tools", j, err)
		}
		tools = append(tools, name)
	}
	return tools, nil
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
		tools, err := parseHookTools(path, at, rawTools)
		if err != nil {
			return guardHook{}, err
		}
		hook.tools = tools
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
	return toolMatches(h.tools, protocolName)
}

// appliesTo reports whether the summary entry has anything to say about the
// named tool.
func (h summaryHook) appliesTo(protocolName string) bool {
	return toolMatches(h.tools, protocolName)
}

// toolMatches is the tools rule both kinds of manifest entry share: an empty
// list means every tool.
func toolMatches(tools []string, protocolName string) bool {
	if len(tools) == 0 {
		return true
	}
	return slices.Contains(tools, protocolName)
}

// hookToolName is the Driver's tool name in the hook protocol's vocabulary, the
// one the manifest is written in. The file Tools never pass through it:
// readPresentation names "Read" for both, so one credential-path rule covers
// all three Tools.
func hookToolName(driverTool string) string {
	if driverTool == "bash" {
		return "Bash"
	}
	return driverTool
}

// hookRequest and hookToolInput are the pre-call payload a hook reads on stdin,
// in the protocol's tool vocabulary: {"tool_name":"Bash","tool_input":{"command":…}}
// for a command, {"tool_name":"Read","tool_input":{"file_path":…}} for a read.
// Both input fields are omitempty because only one is filled per request shape,
// so a Bash request carries no file_path key and a read no command key.
type hookRequest struct {
	ToolName  string        `json:"tool_name"`
	ToolInput hookToolInput `json:"tool_input"`
}

type hookToolInput struct {
	Command  string `json:"command,omitempty"`
	FilePath string `json:"file_path,omitempty"`
}

// hookPostRequest is what the one post-call summary hook reads on stdin: the
// pre-call payload plus tool_response, the finished Tool Result exactly as the
// Model would receive it (exit-code line included). tool_input carries the
// command that actually ran, which after a rewrite is not the command the Model
// asked for — the summary is a step on the way to the Model, not a witness to
// the host's policy, so it gets the truth about what executed.
type hookPostRequest struct {
	ToolName     string        `json:"tool_name"`
	ToolInput    hookToolInput `json:"tool_input"`
	ToolResponse string        `json:"tool_response"`
}

// hookResponse is the part of a hook's stdout this step reads. The pre-call
// chain reads permissionDecision and updatedInput out of it and ignores
// updatedToolOutput, which only the chain's summarize acts on; the summary side
// reads updatedToolOutput and ignores the rest. One struct for both because one
// script can legitimately answer both questions.
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
	// UpdatedToolOutput needs no pointer: an absent one, a JSON null and an empty
	// string all mean "no summary", and the summary path keeps the original output
	// for each of them. The pre-call chain never reads it.
	UpdatedToolOutput string `json:"updatedToolOutput"`
}

// guardPresentation is one call as the Guard chain is asked about it. Today
// there are two shapes: a command a hook may rewrite, and a file path shown to
// hooks as the Read shape.
type guardPresentation struct {
	protocolName string // the tool_name the hooks see
	subject      string // the command, or the file path
	encode       func(subject string) ([]byte, error)
	jqMissing    string // precondition refusal wording for this shape
	// feedsRewrite marks the shape whose subject a hook's answer can replace,
	// which is the only shape hookRewrite's policy applies to. It also decides how
	// often encode runs: a subject nothing can replace never changes, so the chain
	// is built one request instead of one per hook.
	feedsRewrite bool
	// summarizes marks the shape whose finished Tool Result the post-call summary
	// hook may replace. Only the command shape has one today: no file Tool's
	// result and no subagent's answer has ever been summarised.
	summarizes bool
}

// commandPresentation declares a bash-shaped call to the Guard chain: hooks see
// it under the protocol's name for bash, its subject is the command, and that
// command is the one a hook's updatedInput may replace — which is also why it is
// the shape whose result the summary hook is asked about.
func commandPresentation(driverTool, command string) guardPresentation {
	protocolName := hookToolName(driverTool)
	return guardPresentation{
		protocolName: protocolName,
		subject:      command,
		encode:       func(subject string) ([]byte, error) { return encodeHookRequest(protocolName, subject) },
		jqMissing:    "error: jq is not available, so the guarded command cannot be run",
		feedsRewrite: true,
		summarizes:   true,
	}
}

// readPresentation declares a file Tool call to the Guard chain as the Read
// shape — tool_name "Read" for every file Tool rather than two new protocol
// names — so one host rule that denies a credential path covers Edit and Write
// as well as bash. A path has nothing to rewrite, so every hook in the chain is
// asked about the same one request: the rule that refuses a silent hook for
// withholding a rewrite is command-only policy, and any updatedInput a hook
// answers a read with is ignored.
func readPresentation(path string) guardPresentation {
	return guardPresentation{
		protocolName: "Read",
		subject:      path,
		// Takes no subject: a path has nothing to rewrite.
		encode:    func(string) ([]byte, error) { return encodeHookReadRequest(path) },
		jqMissing: "error: jq is not available, so the guarded read cannot be run",
	}
}

// nonePresentation declares that a call has no shape the chain is asked about:
// a Subagent spawn, or a call this run may not make. Such a call has no subject
// a host rule rules on, and presenting it would still put it through
// guardThrough's jq precondition — refusing, on a host without jq, calls that
// run freely today.
func nonePresentation() guardPresentation { return guardPresentation{} }

// asksNothing reports the declaration guardThrough must never see: a presentation
// built by no constructor carries no encoder, and so nothing to ask.
func (p guardPresentation) asksNothing() bool { return p.encode == nil }

// guardThrough runs p through the pre-call Guard Hooks in manifest order and
// returns the subject to act on together with the first refusal to hand back as
// the Tool Result, "" for "no objection". The first refusal stops the chain: a
// call already refused has no reason to be asked of a second hook, and a refused
// call acts on nothing, so the subject returned with a refusal is the empty
// string.
//
// A hook's silence is a pass and anything else is a block. Silence means an
// empty or whitespace-only stdout, an absent or null hookSpecificOutput, and
// any decision that is not the literal "deny" — those are hooks with no
// opinion. A script that cannot be started, a run that outlives
// limits.timeout, a non-zero exit, stdout past limits.outputLimit, and
// stdout that carries no readable verdict are the fail-closed half: the Driver
// refuses rather than accept an answer it never got, since it cannot tell a
// broken hook from an approving one. So is a rewriting hook's answer that rewrote
// nothing — the same doubt about a hook that may have stopped answering halfway.
//
// Every refusal carries the "error: " prefix, which is how a refusal reads to the
// Model; the plan states the failure, not the prefix. A refusal raised by a run
// names the hook's path so the operator can find the script to ask why. The
// missing-jq precondition below is the exception: it belongs to the guarded run
// rather than to one hook, and each presentation words it for its own subject.
func guardThrough(hooks []guardHook, limits hookLimits, p guardPresentation, stderr io.Writer) (string, string) {
	if len(hooks) == 0 {
		// An unguarded run pays nothing: no marshalling, no fork, no wait — and no
		// jq, since nothing here needs reading. The call goes through as asked.
		return p.subject, ""
	}
	if _, err := exec.LookPath("jq"); err != nil {
		// A guard that cannot read its own request must never be trusted as
		// approving: the host's baked hook scripts shell out to jq to parse the
		// stdin the payload below carries, so without jq a hook exits 0 having
		// inspected nothing and that silence reads as a pass. Refuse before any hook
		// runs, which is also before the call can act.
		return "", p.jqMissing
	}
	subject := p.subject
	var payload []byte
	if !p.feedsRewrite {
		// Built once for the chain where nothing feeds forward: every hook is asked
		// about the same subject, so the request cannot differ between them. It is
		// built before the loop rather than inside it, so a build that fails refuses
		// the call even when no hook in the chain applies to this shape.
		var err error
		if payload, err = p.encode(subject); err != nil {
			return "", encodeRefusal(err)
		}
	}
	for _, hook := range hooks {
		if !hook.appliesTo(p.protocolName) {
			continue
		}
		if p.feedsRewrite {
			// Built per hook where a rewrite feeds forward: an earlier hook's answer is
			// the subject the next is asked about.
			var err error
			if payload, err = p.encode(subject); err != nil {
				return "", encodeRefusal(err)
			}
		}
		// The run is judged before the verdict: a hook that never answered blocks
		// the call on its own, with nothing left to parse.
		stdout, refusal := runHookScript(hook.path, payload, stderr, limits)
		var answer hookResponse
		if refusal == "" {
			answer, refusal = hookVerdict(hook, stdout)
		}
		if refusal == "" && p.feedsRewrite {
			// Judged only after the verdicts, so a rewriting hook that denies is
			// refused for denying rather than for the rewrite it withheld.
			subject, refusal = hookRewrite(hook, subject, answer)
		}
		if refusal != "" {
			return "", refusal
		}
	}
	return subject, ""
}

// encodeRefusal words the fail-closed judgement on a request the Driver could
// not build: a call no hook was ever asked about must not be acted on.
func encodeRefusal(err error) string {
	return fmt.Sprintf("error: guard hook request could not be built: %v", err)
}

// guard asks the chain about a call its plan already declared and returns the
// subject to act on with the first refusal, "" for "no objection".
//
// Both early answers are load-bearing. A nil chain is the run that configured no
// Hook Manifest, and it answers before the jq lookup guardThrough makes: there
// is nothing to guard. A presentation that declares none has to be left alone by
// a run that does have hooks, so it answers the same way rather than reaching the
// precondition of a shape it never declared.
func (c *guardChain) guard(p guardPresentation, stderr io.Writer) (string, string) {
	if c == nil || p.asksNothing() {
		return p.subject, ""
	}
	return guardThrough(c.pre, c.limits, p, stderr)
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
	return encodeHookPayload(hookRequest{ToolName: protocolName, ToolInput: hookToolInput{Command: command}})
}

// encodeHookReadRequest builds the stdin one hook is handed when the call under
// guard is a file Tool reading or writing path, in the shape readPresentation
// declares — tool_name "Read" rather than the Driver's own name for the call.
func encodeHookReadRequest(path string) ([]byte, error) {
	return encodeHookPayload(hookRequest{ToolName: "Read", ToolInput: hookToolInput{FilePath: path}})
}

// encodeHookPostRequest builds the stdin the summary hook is handed for a
// finished Tool Result, in the same encoder shape as encodeHookRequest.
func encodeHookPostRequest(protocolName, command, output string) ([]byte, error) {
	return encodeHookPayload(hookPostRequest{
		ToolName:     protocolName,
		ToolInput:    hookToolInput{Command: command},
		ToolResponse: output,
	})
}

func encodeHookPayload(payload any) ([]byte, error) {
	var request bytes.Buffer
	enc := json.NewEncoder(&request)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(payload); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(request.Bytes(), []byte("\n")), nil
}

// runHookScript executes one manifest script and returns its stdout, or, when
// the run itself failed, the refusal that blocks the call in place of an answer
// — which is how the pre-call chain reads it. The summary path ignores that
// string and keeps its original output instead: the same run mechanics, the
// opposite verdict on a failed run, because the two callers have different
// defaults to fall back on.
// The two bounds enforced here come from limits, the pair the calling chain was
// built with, and the timeout refusal quotes that value back, so a blocked call
// names the bound that actually stopped it.
func runHookScript(path string, payload []byte, stderr io.Writer, limits hookLimits) (stdout []byte, refusal string) {
	ctx, cancel := context.WithTimeout(context.Background(), limits.timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, path)
	// The script is what the host baked — shebang, mode bits and all — so it is
	// exec'd directly. A shell in between would add a second quoting layer over
	// text the hook is handed as data, and a manifest path is not a command.
	cmd.Stdin = bytes.NewReader(payload)
	out := &cappedWriter{limit: limits.outputLimit}
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
	// No cmd.Dir and no cmd.Env, following the Bash Tool's run: the hook judges the
	// command in the same cwd and environment the command is about to run in.
	// Kill the whole group, not just the hook: a grandchild the hook detached
	// would outlive a bare kill of the hook and keep the output pipes open, which
	// is exactly what the WaitDelay below only bounds.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	// WaitDelay for the reason the Bash Tool's run gives: a grandchild the hook detached
	// can survive the kill holding the output pipes, and Wait would block on
	// their EOF forever — a fired bound that never produces a refusal.
	cmd.WaitDelay = time.Second
	if err := cmd.Start(); err != nil {
		return nil, fmt.Sprintf("error: guard hook %s could not be run: %v", path, err)
	}
	err := cmd.Wait()
	if ctx.Err() != nil {
		// Judged before err, so the kill the bound fired reads as a timeout and not
		// as the signal death it was. The accepted race the Bash Tool documents
		// applies here too: a hook that finished between Wait and this check is
		// reported as timed out.
		return nil, fmt.Sprintf("error: guard hook %s timed out after %s, so the command is blocked", path, limits.timeout)
	}
	if err != nil {
		// Includes a death on a signal, which exec words as "signal: …" — the
		// hook stopped without a verdict either way.
		return nil, fmt.Sprintf("error: guard hook %s failed: %v", path, err)
	}
	if out.truncated {
		// Judged before any verdict is read out of the kept prefix: what was thrown
		// away may have been a deny, and an answer the Driver only half received is
		// not an answer.
		return nil, fmt.Sprintf("error: guard hook %s wrote more than %d bytes of output, so the command is blocked", path, out.limit)
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

// summarize hands a finished Tool Result to the chain's one post-call summary
// hook and returns its replacement, or output whenever the summary could not be
// had.
//
// Fail open, where the pre-call chain fails closed — and it is the same
// hook mechanics underneath, so the shared helper's refusal is dropped
// on purpose. guardThrough refuses because a hook it cannot hear from is
// indistinguishable from one that would have approved, and running an unguarded
// command is the harm. Here the Driver is holding a real Tool Result already: a
// summary hook that hangs, exits non-zero, floods, or answers with noise costs
// the run its summary and must not cost it the answer. The same doubt about a
// hook resolves the other way because the default action differs.
//
// So every one of these returns output untouched: no chain at all, no summary
// hook configured, or one whose tools list omits this tool (the script never
// runs), a script that cannot be started, a run past the chain's time bound, a
// non-zero exit, stdout past the chain's output cap, stdout that is not readable
// JSON, an absent hookSpecificOutput, and an absent or empty updatedToolOutput
// — a summary of nothing would replace a real result with the empty string,
// which is the loss this function exists to prevent.
//
// A replacement is capped by toolOutputCap, the output bound of the Tool whose
// result is being replaced — the cap that Bash Tool was built with — which the
// caller passes because that bound belongs to the Tool rather than here. A hook that summarises upwards
// cannot hand the Model a longer result than the command itself was allowed.
// Nothing re-appends the exit-code line: the replacement is the Tool Result
// verbatim, and planToolCall keeps the
// command's own verdict for is_error.
func (c *guardChain) summarize(driverTool, command, output string, stderr io.Writer, toolOutputCap int) string {
	if c == nil || c.summary == nil {
		return output
	}
	hook := c.summary
	protocolName := hookToolName(driverTool)
	if !hook.appliesTo(protocolName) {
		return output
	}
	payload, err := encodeHookPostRequest(protocolName, command, output)
	if err != nil {
		return output
	}
	stdout, _ := runHookScript(hook.path, payload, stderr, c.limits)
	var response hookResponse
	if err := json.Unmarshal(stdout, &response); err != nil || response.HookSpecificOutput == nil {
		return output
	}
	summary := response.HookSpecificOutput.UpdatedToolOutput
	if summary == "" {
		return output
	}
	return truncateToolOutput(summary, toolOutputCap)
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
