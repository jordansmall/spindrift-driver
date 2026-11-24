package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// runResult is what one helper-driven run reports back: everything a run can be
// asked about, in the shape a test asserts on.
type runResult struct {
	code int
	// stdout is the raw event stream, UNnormalised: a run writes its t.TempDir()
	// run directory into tool targets and outputs, and the caller comparing a
	// byte-stable artifact does its own substitution before comparing. Normalising
	// here would hide the path the caller has to normalise.
	stdout []byte
	// events is stdout read back through stdoutEvents.
	events []map[string]any
	stderr string
	// requests holds every request body the scripted server took, in arrival order.
	requests [][]byte
}

// runOption configures one aspect of a helper-driven run.
type runOption func(*runSettings)

// envAssignment is one environment variable a caller wants set.
type envAssignment struct {
	key   string
	value string
}

// runSettings is the accumulated option list a runDriver call runs under.
type runSettings struct {
	flags  []string
	env    []envAssignment
	roster []agentSpec
	// manifest holds the Hook Manifest contents; manifestSet tells "the caller
	// asked for one" apart from the empty default.
	manifest    string
	manifestSet bool
}

// withFlags appends extra Driver flags after the prompt in argv — --model,
// --agent, --max-turns — the way an operator types them.
func withFlags(args ...string) runOption {
	return func(s *runSettings) { s.flags = append(s.flags, args...) }
}

// withEnv sets one environment variable for the run. It is repeatable, and the
// pairs land after the API-key default and before the helper's own pins, so a
// caller can steer $HOME but never the endpoint or the Driver Config.
func withEnv(key, value string) runOption {
	return func(s *runSettings) { s.env = append(s.env, envAssignment{key: key, value: value}) }
}

// withRoster writes one Agent File per spec into the directory the run's Driver
// Config names as agents_dir. File names carry the spec index, so the roster's
// lexical order is the order the caller wrote the specs in.
func withRoster(specs ...agentSpec) runOption {
	return func(s *runSettings) { s.roster = append(s.roster, specs...) }
}

// withHookManifest points that same Driver Config at a Hook Manifest holding
// contents — hook_manifest is a config key with no flag and no environment
// variable of its own.
func withHookManifest(contents string) runOption {
	return func(s *runSettings) {
		s.manifest = contents
		s.manifestSet = true
	}
}

// runDriver runs the Driver in-process against a scripted Model and reports what
// came of it.
//
// The base URL and the Driver Config path are the helper's to own — pinned last,
// over anything inherited — so an environment left over from a developer's shell
// can never point a test's run at a real endpoint or a real roster. Everything
// else, including the API key the caller overrides, stays the caller's.
func runDriver(t *testing.T, script map[string][]modelEntry, prompt string, opts ...runOption) runResult {
	t.Helper()
	st := runSettings{}
	for _, opt := range opts {
		opt(&st)
	}

	srv := newScriptedModelServer(t, script)
	cfgPath := runDriverConfig(t, st)

	t.Setenv(envAPIKey, runDriverFakeKey)
	for _, kv := range st.env {
		t.Setenv(kv.key, kv.value)
	}
	subagentEnv(t, cfgPath, srv.url())

	var stdout, stderr bytes.Buffer
	code := run(append([]string{prompt}, st.flags...), strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	srv.checkNoUnscriptedRequests(t)

	raw := stdout.String()
	return runResult{
		code:     code,
		stdout:   []byte(raw),
		events:   stdoutEvents(t, raw),
		stderr:   stderr.String(),
		requests: srv.requestBodies(t),
	}
}

// runDriverFakeKey is the bearer token a helper-driven run authenticates with: a
// placeholder that names no provider and matches none of the credential shapes
// golden_contract_test.go scans for, so a test log that ever carried it would not
// read as a leak of something real.
const runDriverFakeKey = "runhelper-placeholder"

// A helper-driven run reports the four things a run can be asked about: the exit
// code, the event stream read back, whatever the Driver wrote to stderr, and the
// requests the scripted Model actually took.
func TestRunDriverReturnsCodeEventsStderrAndRequests(t *testing.T) {
	got := runDriver(t, map[string][]modelEntry{
		"m-a": {replyCompletion(`{"model":"m-a","choices":[{"index":0,"message":{"role":"assistant","content":"all done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":3}}`)},
	}, "p", withFlags("--model", "m-a"))

	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q stdout = %q", got.code, got.stderr, got.stdout)
	}
	if len(got.events) == 0 {
		t.Fatalf("no events parsed; stdout = %q", got.stdout)
	}
	if first := got.events[0]["kind"]; first != "start" {
		t.Errorf("first event kind = %v, want %q", first, "start")
	}
	if last := got.events[len(got.events)-1]["kind"]; last != "result" {
		t.Errorf("last event kind = %v, want %q", last, "result")
	}
	// stderr is the Driver's progress channel rather than an event channel: even a
	// clean run logs one POST line per Model call, so "clean" here is exactly that
	// line and nothing else — no retry note, no diagnostic.
	if lines := strings.Split(strings.TrimSuffix(got.stderr, "\n"), "\n"); len(lines) != 1 || !strings.HasPrefix(lines[0], "spindrift-driver: POST ") {
		t.Errorf("stderr = %q, want only the one POST progress line a clean run logs", got.stderr)
	}

	// stdout is handed back raw and UNnormalised so a caller can do its own run-dir
	// substitution: the events are that same buffer read back, so the raw bytes have
	// to still carry what the parse found rather than a cleaned-up copy.
	if !strings.Contains(string(got.stdout), "all done") {
		t.Errorf("raw stdout = %q, want the answer the parsed events carry", got.stdout)
	}

	if len(got.requests) != 1 {
		t.Fatalf("server recorded %d request(s), want 1: %s", len(got.requests), got.requests)
	}
	if model := requestModel(t, got.requests[0]); model != "m-a" {
		t.Errorf("recorded request model = %q, want %q", model, "m-a")
	}
}

// runDriverConfig writes the ONE Driver Config a helper-driven run reads and
// hands back its path, so a roster and a Hook Manifest are configured the way an
// operator configures them — two keys of one file — rather than through any
// channel a test invents. With neither option set the file is `{}`: the run still
// reads one config, and it is the config the helper pinned.
func runDriverConfig(t *testing.T, st runSettings) string {
	t.Helper()
	fields := map[string]string{}
	if len(st.roster) > 0 {
		fields["agents_dir"] = runRosterDir(t, st.roster)
	}
	if st.manifestSet {
		fields["hook_manifest"] = writeHookManifest(t, st.manifest)
	}
	raw, err := json.Marshal(fields)
	if err != nil {
		t.Fatalf("marshal the run's Driver Config: %v", err)
	}
	return writeDriverConfig(t, string(raw))
}

// runRosterDir writes one Agent File per spec into a fresh directory through the
// one writer every roster fixture uses, so a helper roster and a fixture roster
// load identically.
func runRosterDir(t *testing.T, specs []agentSpec) string {
	t.Helper()
	dir := t.TempDir()
	writeAgentFixtureFiles(t, dir, specs)
	return dir
}

// Extra flags go into argv after the prompt, so a run can be steered to another
// model: with both queues scripted, the one the flag names answers and the other
// is never POSTed to.
func TestRunDriverExtraFlagsReachTheDriver(t *testing.T) {
	got := runDriver(t, map[string][]modelEntry{
		"m-a": {replyCompletion(`{"choices":[{"index":0,"message":{"role":"assistant","content":"the a answer"}}]}`)},
		"m-b": {replyCompletion(`{"choices":[{"index":0,"message":{"role":"assistant","content":"the b answer"}}]}`)},
	}, "p", withFlags("--model", "m-b"))

	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q stdout = %q", got.code, got.stderr, got.stdout)
	}
	if len(got.requests) != 1 {
		t.Fatalf("server recorded %d request(s), want 1", len(got.requests))
	}
	if model := requestModel(t, got.requests[0]); model != "m-b" {
		t.Errorf("recorded request model = %q, want %q: the flag never reached the Driver", model, "m-b")
	}
	if !strings.Contains(string(got.stdout), "the b answer") {
		t.Errorf("stdout = %q, want m-b's answer", got.stdout)
	}
}

// requestModel reads the model id out of one recorded request body, which is the
// only field a run-driven test routes on (see scriptedModelServer).
func requestModel(t *testing.T, raw []byte) string {
	t.Helper()
	var body struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("recorded request body %q is not JSON: %v", raw, err)
	}
	return body.Model
}

// A caller's environment reaches the run through the same injected lookupEnv the
// Driver reads, including $HOME: the config the helper pins names no agents_dir,
// so the run falls back to the home-relative roster and --agent resolves there.
// Without the HOME assignment the same run has no roster at all and fails on the
// unknown agent, which is what makes this more than a run that ignored its env.
func TestRunDriverCallerEnvReachesTheRun(t *testing.T) {
	script := func() map[string][]modelEntry {
		return map[string][]modelEntry{
			"m-reviewer": {replyCompletion(`{"choices":[{"index":0,"message":{"role":"assistant","content":"reviewed"}}]}`)},
		}
	}

	home := t.TempDir()
	writeAgentFile(t, homeRosterDir(t, home), "reviewer.md", "---\nname: reviewer\nmodel: m-reviewer\n---\nreview things\n")

	got := runDriver(t, script(), "p", withEnv("HOME", home), withFlags("--agent", "reviewer"))
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q stdout = %q", got.code, got.stderr, got.stdout)
	}
	if len(got.requests) != 1 {
		t.Fatalf("server recorded %d request(s), want 1", len(got.requests))
	}
	if model := requestModel(t, got.requests[0]); model != "m-reviewer" {
		t.Errorf("recorded request model = %q, want the roster entry's model %q", model, "m-reviewer")
	}

	// The non-vacuity half: an empty home carries no roster, so the same argv is
	// an unknown agent and a failed run.
	empty := runDriver(t, script(), "p", withEnv("HOME", t.TempDir()), withFlags("--agent", "reviewer"))
	if empty.code == 0 {
		t.Fatalf("exit code = 0 with an empty HOME, want the run to fail: the roster it answered from was not the caller's home\nstdout = %q", empty.stdout)
	}
	if !strings.Contains(empty.stderr, "unknown agent") {
		t.Errorf("stderr = %q, want it to name the unknown agent", empty.stderr)
	}
}

// homeRosterDir is the roster directory a run falls back to under a given HOME
// (defaultAgentsDirPath joined to it), created so an Agent File can be written
// straight into it.
func homeRosterDir(t *testing.T, home string) string {
	t.Helper()
	dir := filepath.Join(home, defaultAgentsDirPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("create home roster dir %s: %v", dir, err)
	}
	return dir
}

// The roster option is the same roster an operator points agents_dir at: one
// directory of Agent Files named after their spec index, so --agent resolves the
// entry and its model answers.
func TestRunDriverRosterOptionSelectsTheEntry(t *testing.T) {
	got := runDriver(t, map[string][]modelEntry{
		"m-a":        {replyCompletion(`{"choices":[{"index":0,"message":{"role":"assistant","content":"the parent answer"}}]}`)},
		"m-reviewer": {replyCompletion(`{"choices":[{"index":0,"message":{"role":"assistant","content":"reviewed, no blockers"}}]}`)},
	}, "p",
		withRoster(agentSpec{name: "reviewer", model: "m-reviewer", description: "reviews a diff", body: "review the change and report blockers"}),
		withFlags("--agent", "reviewer"),
	)

	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q stdout = %q", got.code, got.stderr, got.stdout)
	}
	if len(got.requests) != 1 {
		t.Fatalf("server recorded %d request(s), want 1", len(got.requests))
	}
	if model := requestModel(t, got.requests[0]); model != "m-reviewer" {
		t.Errorf("recorded request model = %q, want the roster entry's model %q", model, "m-reviewer")
	}
	// The Agent File's body is the run's system prompt, so the prompt the endpoint
	// saw carries it — the entry was read, not just looked up by name.
	if !strings.Contains(string(got.requests[0]), "review the change and report blockers") {
		t.Errorf("recorded request = %q, want the entry's body as the system prompt", got.requests[0])
	}
}

// The Hook Manifest option guards the run the way a configured manifest guards an
// operator's: the deny hook's refusal comes back as the Tool Result, the command
// never runs, and the run itself still succeeds.
func TestRunDriverHookManifestOptionGuardsTheRun(t *testing.T) {
	dir := t.TempDir()
	marker := filepath.Join(dir, "command-ran.marker")
	bashTurn := replyCompletion(fmt.Sprintf(`{"choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":%q}}]}}]}`,
		fmt.Sprintf(`{"command":%q}`, "printf ran > "+marker)))
	// Every turn that calls a Tool asks again afterwards, so each run needs the
	// answer that closes it.
	script := func() map[string][]modelEntry {
		return map[string][]modelEntry{
			"m-a": {bashTurn, replyCompletion(`{"choices":[{"index":0,"message":{"role":"assistant","content":"reported the refusal"}}]}`)},
		}
	}
	hookScript := writeGuardHookScript(t, "deny.sh",
		`echo '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"no shell from a scripted run"}}'`)

	got := runDriver(t, script(), "p",
		withHookManifest(`{"pre":[{"path":"`+hookScript+`","tools":["Bash"]}]}`),
		withFlags("--model", "m-a"),
	)
	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0 — a refusal is a Tool Result, not a failed run; stderr = %q stdout = %q", got.code, got.stderr, got.stdout)
	}
	result, ok := toolResultFor(t, got.events, "call_1")
	if !ok {
		t.Fatalf("no tool_result event for call_1: %q", got.stdout)
	}
	output, _ := result["tool_output"].(string)
	for _, want := range []string{"error: guard hook " + hookScript, "no shell from a scripted run"} {
		if !strings.Contains(output, want) {
			t.Errorf("tool_output = %q, want it to carry %q", output, want)
		}
	}
	if result["is_error"] != true {
		t.Errorf("is_error = %v, want true for a refused call", result["is_error"])
	}
	if hookMarkerExists(t, marker) {
		t.Errorf("marker %s exists: the configured deny hook did not stop the command", marker)
	}

	// The non-vacuity half: the same script with no manifest runs the command.
	unguarded := runDriver(t, script(), "p", withFlags("--model", "m-a"))
	if unguarded.code != 0 {
		t.Fatalf("unguarded exit code = %d, want 0; stderr = %q stdout = %q", unguarded.code, unguarded.stderr, unguarded.stdout)
	}
	if !hookMarkerExists(t, marker) {
		t.Errorf("marker %s does not exist: the command did not run with no hook_manifest configured", marker)
	}
}

// toolResultFor finds the Tool Result that answers one tool_call id.
func toolResultFor(t *testing.T, events []map[string]any, toolID string) (map[string]any, bool) {
	t.Helper()
	for _, ev := range events {
		if ev["kind"] == "tool_result" && ev["tool_id"] == toolID {
			return ev, true
		}
	}
	return nil, false
}
