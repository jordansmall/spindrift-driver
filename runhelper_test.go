package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
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
	// requests holds every request the scripted server took, in arrival order, wire
	// facts and body together: the body and the headers a test asserts on belong to
	// the same POST, so they are read off one entry.
	requests []recordedRequest
	// gaps holds the wait between consecutive request arrivals, so a backoff test
	// reads the spacing the run actually left rather than reconstructing it from the
	// arrival times.
	gaps []time.Duration
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
	// stdoutFile, when set, is where the run's event stream goes instead of the
	// helper's buffer, because something running during the run reads it there.
	stdoutFile string
	// now is the clock the run stamps and measures with; nil means the wall clock.
	now func() int64
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

// withStdoutFile sends the run's event stream to path rather than capturing it in
// memory, for a run whose own mid-run witness — a Hook grepping the stream while
// the guarded call executes — has to open it. The run's reported stdout and events
// are that file read back once the run returns.
func withStdoutFile(path string) runOption {
	return func(s *runSettings) { s.stdoutFile = path }
}

// withClock runs the Driver on a clock the caller built rather than the wall
// clock, so the stamps and the durations one run reports land on a grid the test
// names — the in-process counterpart of the $SPINDRIFT_TEST_NOW_MS pin a spawned
// child Driver gets.
func withClock(now func() int64) runOption {
	return func(s *runSettings) { s.now = now }
}

// completeConfigTimeout is the helper's own idle bound, deliberately not the
// chatAttemptTimeout package var: tests inject that var to drive timeout
// scenarios, and a config built while an injection is in effect would otherwise
// inherit a bound no caller of this helper asked for.
const completeConfigTimeout = 30 * time.Second

// completeConfig returns Settings that are complete as returned — model,
// endpoint, turn cap, idle bound, Tool set and system prompt (#66) — so a caller
// overwrites only the field its own assertion is about.
func completeConfig(baseURL string) config {
	return config{
		model:        "complete-config-model",
		baseURL:      baseURL,
		maxTurns:     defaultMaxTurns,
		timeout:      completeConfigTimeout,
		tools:        resolveTools(agentEntry{}, nil),
		systemPrompt: systemPrompt,
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

	// A caller whose run has to show something INSIDE the run its own event stream —
	// a Guard Hook that greps the wire while the call is executing — names a file to
	// write it to instead of the buffer below, and the result is read back off disk.
	var stdout, stderr bytes.Buffer
	out := io.Writer(&stdout)
	var outFile *os.File
	if st.stdoutFile != "" {
		f, err := os.Create(st.stdoutFile)
		if err != nil {
			t.Fatalf("create the run's stdout file %s: %v", st.stdoutFile, err)
		}
		defer f.Close()
		outFile, out = f, f
	}
	now := st.now
	if now == nil {
		now = realNowMS
	}
	code := runWithClock(now, append([]string{prompt}, st.flags...), strings.NewReader(""), out, &stderr, os.LookupEnv)
	srv.checkNoUnscriptedRequests(t)

	raw := stdout.String()
	if outFile != nil {
		// Closed here rather than left to the deferred Close, so the read-back never
		// depends on the run's writer staying unbuffered. The mid-run Hook grep is what
		// witnesses that property, and fails on its own if it ever stops holding.
		if err := outFile.Close(); err != nil {
			t.Fatalf("close the run's stdout file %s: %v", st.stdoutFile, err)
		}
		b, err := os.ReadFile(st.stdoutFile)
		if err != nil {
			t.Fatalf("read back the run's stdout file %s: %v", st.stdoutFile, err)
		}
		raw = string(b)
	}
	return runResult{
		code:     code,
		stdout:   []byte(raw),
		events:   stdoutEvents(t, raw),
		stderr:   stderr.String(),
		requests: srv.recordedRequests(t),
		gaps:     srv.gaps(),
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
		t.Fatalf("server recorded %d request(s), want 1: %v", len(got.requests), got.requests)
	}
	if model := requestModel(t, got.requests[0].body); model != "m-a" {
		t.Errorf("recorded request model = %q, want %q", model, "m-a")
	}
}

// The wire facts a run test asserts about how the Model was asked for come back on
// the run's own result: one record per POST holding method, path and request
// headers next to the body, so the body and the headers never come from two
// different entries. A run that POSTed once asked for nothing a gap could measure,
// so gaps stays empty even though the record is full.
func TestRunDriverReportsWireFactsAndArrivalGaps(t *testing.T) {
	got := runDriver(t, map[string][]modelEntry{
		"m-a": {replyCompletion(`{"choices":[{"index":0,"message":{"role":"assistant","content":"wire facts"}}]}`)},
	}, "p", withFlags("--model", "m-a"))

	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q stdout = %q", got.code, got.stderr, got.stdout)
	}
	if len(got.requests) != 1 {
		t.Fatalf("server recorded %d request(s), want 1", len(got.requests))
	}
	rec := got.requests[0]
	if rec.method != "POST" {
		t.Errorf("recorded method = %q, want POST", rec.method)
	}
	if rec.path != "/chat/completions" {
		t.Errorf("recorded path = %q, want /chat/completions", rec.path)
	}
	if ct := rec.header.Get("Content-Type"); ct == "" {
		t.Errorf("recorded Content-Type is empty, want the JSON type the run sent: %v", rec.header)
	}
	if model := requestModel(t, rec.body); model != "m-a" {
		t.Errorf("recorded request model = %q, want %q", model, "m-a")
	}
	if len(got.gaps) != 0 {
		t.Errorf("gaps = %v, want none for a run that POSTed once", got.gaps)
	}
}

// A migrated run test asserts on the conversation each POST carried rather than on
// raw JSON: messageBodies returns one []recordedMessage per POST, in arrival order,
// read off the same recorded requests the wire facts come from, and
// requestMessages reads one of those bodies on its own.
func TestRunDriverResultExposesPostedMessages(t *testing.T) {
	bashTurn := replyCompletion(fmt.Sprintf(`{"choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":%q}}]}}]}`,
		`{"command":"echo one"}`))
	got := runDriver(t, map[string][]modelEntry{
		"m": {bashTurn, replyCompletion(`{"choices":[{"index":0,"message":{"role":"assistant","content":"all done"},"finish_reason":"stop"}]}`)},
	}, "summarise the diff", withFlags("--model", "m"))

	if got.code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr = %q stdout = %q", got.code, got.stderr, got.stdout)
	}
	if len(got.requests) != 2 {
		t.Fatalf("server recorded %d request(s), want the 2 the Tool loop made: %v", len(got.requests), got.requests)
	}

	bodies := got.messageBodies(t)

	// The first POST carries only the opening turn the run started from.
	first := bodies[0]
	if len(first) != 2 {
		t.Fatalf("first POST carries %d message(s), want the opening system and user turns: %+v", len(first), first)
	}
	if first[0].Role != "system" {
		t.Errorf("first POST opens with role %q, want %q", first[0].Role, "system")
	}
	if first[1].Role != "user" || first[1].Content != "summarise the diff" {
		t.Errorf("first POST user turn = %+v, want role %q with the prompt %q as content", first[1], "user", "summarise the diff")
	}

	// Arrival order, not a reroll: the second POST is that same conversation with
	// the first turn's Tool Call and its result appended.
	second := bodies[1]
	if len(second) <= len(first) {
		t.Fatalf("second POST carries %d message(s), want the first POST's %d plus the Tool turn: %+v", len(second), len(first), second)
	}
	if !reflect.DeepEqual(second[:len(first)], first) {
		t.Errorf("second POST does not start with the first POST's conversation:\nfirst  = %+v\nsecond = %+v", first, second)
	}
	if sent := second[len(first)]; sent.Role != "assistant" || len(sent.ToolCalls) != 1 || sent.ToolCalls[0].ID != "call_1" {
		t.Errorf("second POST continues with %+v, want the assistant turn calling call_1", sent)
	}
	if last := second[len(second)-1]; last.Role != "tool" || last.ToolCallID != "call_1" {
		t.Errorf("second POST ends with %+v, want the tool result answering call_1", last)
	}

	// The single-body counterpart reads the same conversation out of one recorded
	// body, for a test that only inspects one POST.
	if one := requestMessages(t, got.requests[1].body); !reflect.DeepEqual(one, second) {
		t.Errorf("requestMessages of the second POST = %+v, want the same conversation messageBodies reported: %+v", one, second)
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
	if model := requestModel(t, got.requests[0].body); model != "m-b" {
		t.Errorf("recorded request model = %q, want %q: the flag never reached the Driver", model, "m-b")
	}
	if !strings.Contains(string(got.stdout), "the b answer") {
		t.Errorf("stdout = %q, want m-b's answer", got.stdout)
	}
}

// messageBodies returns the conversation each POST sent, one []recordedMessage per
// entry of r.requests and in the same arrival order — the message-level view a
// migrated run test asserts on where it used to read the old fake's rawBodies.
func (r runResult) messageBodies(t *testing.T) [][]recordedMessage {
	t.Helper()
	out := make([][]recordedMessage, 0, len(r.requests))
	for _, req := range r.requests {
		out = append(out, requestMessages(t, req.body))
	}
	return out
}

// requestMessages reads the messages out of one recorded request body, failing the
// test on a body that is not valid JSON the way the old fake's accessor did.
func requestMessages(t *testing.T, raw []byte) []recordedMessage {
	t.Helper()
	var body struct {
		Messages []recordedMessage `json:"messages"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("recorded request body %q is not JSON: %v", raw, err)
	}
	return body.Messages
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
	if model := requestModel(t, got.requests[0].body); model != "m-reviewer" {
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
	if model := requestModel(t, got.requests[0].body); model != "m-reviewer" {
		t.Errorf("recorded request model = %q, want the roster entry's model %q", model, "m-reviewer")
	}
	// The Agent File's body is the run's system prompt, so the prompt the endpoint
	// saw carries it — the entry was read, not just looked up by name.
	if !strings.Contains(string(got.requests[0].body), "review the change and report blockers") {
		t.Errorf("recorded request = %q, want the entry's body as the system prompt", got.requests[0].body)
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

// TestCompleteConfigIsCompleteForARun pins the contract of completeConfig: the
// settings it returns are the settings a run starts from, so no caller fills a
// field in later (#66).
func TestCompleteConfigIsCompleteForARun(t *testing.T) {
	cfg := completeConfig("http://127.0.0.1:9/v1")
	if cfg.model == "" {
		t.Error("model is empty, want a model a run can post with")
	}
	if cfg.baseURL != "http://127.0.0.1:9/v1" {
		t.Errorf("baseURL = %q, want the endpoint passed in", cfg.baseURL)
	}
	if cfg.timeout <= 0 {
		t.Errorf("timeout = %v, want a positive idle bound", cfg.timeout)
	}
	if cfg.maxTurns <= 0 {
		t.Errorf("maxTurns = %d, want a positive turn cap", cfg.maxTurns)
	}
	if len(cfg.tools.specs()) == 0 {
		t.Error("offered Tool set is empty, want the set a run posts with")
	}
	if cfg.systemPrompt == "" {
		t.Error("systemPrompt is empty, want the prompt a run starts from")
	}
}
