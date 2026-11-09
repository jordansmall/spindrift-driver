package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// goldenRateLimitedModel is the model id of the roster entry whose spawn must
// fail, and the one the scripted server answers with 429. Routing by the model in
// the request body is the only way to fail exactly one child: a spawn inherits its
// parent's endpoint through the environment (see subagentEnv) and cannot be
// pointed anywhere else.
const goldenRateLimitedModel = "m-scout"

// goldenRetryAfter is the Retry-After header that scripted 429 answers with. It is
// an absolute HTTP-date rather than delta-seconds: retryAfterMS adds a delta to
// whatever reading of its own pinned clock the child takes at that moment, so a
// delta would tie reset_at to how far that child had counted, while a date names
// one instant in every process — 1787274123000, the epoch milliseconds reset_at
// carries.
const goldenRetryAfter = "Fri, 21 Aug 2026 01:02:03 GMT"

// goldenRateLimitBody is the 429's body, which the error event carries verbatim.
// It carries no trailing newline, which is why the handler writes it by hand
// rather than through http.Error: the newline that helper appends would then be
// part of the rejection the artifact records.
const goldenRateLimitBody = `{"error":{"type":"rate_limit_error","message":"m-scout is over its quota"}}`

// The two child Drivers stamp on their own pinned grid (envTestNowMS is the only
// channel to a process this test starts), and both spawns read the same spec, so no
// single value can line a child's ticks up with the point in the parent's stream
// where that child was spawned. That is what makes timestamp monotonicity a
// per-stream property of this artifact rather than a global one.
const (
	goldenChildClockStart = 1700000000000
	goldenChildClockStep  = 100
)

// goldenUpdate is the regeneration switch for the committed golden log, bound to
// the -update flag TestMain registers. It is a *bool rather than a bool because
// the value only exists once m.Run() has parsed the command line.
var goldenUpdate *bool

// goldenPassPath is the committed contract artifact: the stdout of one scripted
// Pass, byte for byte. It is a file rather than a Go literal so a downstream
// decoder author can point a parser at it, and it lives in testdata so tooling
// skips it as source.
const goldenPassPath = "testdata/pass.jsonl"

// goldenRegenerateCmd is named in every failure this test can produce: an
// artifact that drifted is only actionable if the reader knows how to rewrite it.
const goldenRegenerateCmd = "go test -run TestGoldenPassLog -update"

// goldenRunDir is the single spelling the artifact gives the run's working
// directory. The file Tools resolve their path argument through filepath.Abs, so
// every write/edit tool_target and tool_output names an absolute path — and the
// Pass writes into a t.TempDir(), which differs per run and per machine. Without
// this substitution the committed file would record whoever generated it.
const goldenRunDir = "/golden-run"

// TestGoldenPassLog compares one scripted Pass through the whole Driver against
// the committed golden byte for byte.
//
// The stream is produced in-process — run() into a bytes.Buffer with
// pinSteppingClock fixing the clock and newScriptedSSERawServer fixing every
// response — because the nix check gives the suite bash and coreutils only (no
// jq, no git, no network), so regeneration cannot shell out.
//
// The run is shaped to put every interesting event form in one Pass: a Model
// call that streams thinking ahead of its prose and then a succeeding bash tool
// call; a call whose answer arrives only as chunks, two content deltas and a
// write call's arguments split across two frames, so the log pins reassembly of
// both channels; a whole-body call that edits the file that write created; two
// Subagent calls, one whose child answers and one whose child is rate-limited, so a
// relayed child block sits inside each pair it was spawned for; three calls with no
// prose at all (which is why they carry no text event) that send one failing bash
// command and then the same command verbatim twice more until the repeated-call
// breaker refuses to run it; then a final answer. That yields every kind the Driver
// writes: start, reasoning, text, usage, tool_call, tool_result in its success,
// is_error, and refused shapes — the refused one an is_error result with no
// exit-code line, because no command ran — error, and result.
//
// The two file Tool pairs report an absolute tool_target, so their path is
// goldenPassStream's temporary run directory with goldenRunDir substituted for it
// — the artifact states one path, not the machine that produced it.
//
// The error kind reaches this artifact only through the relay. A Pass that fails
// ends on its own error event and never writes a result, so the only way to hold
// both kinds in one file is a child whose Model call is rejected: its error line is
// forwarded onto the parent's stdout, and the parent goes on to close with its own
// result.
func TestGoldenPassLog(t *testing.T) {
	got := goldenPassStream(t)

	// Shape is asserted on the produced stream either way, so -update cannot
	// quietly commit a degenerate artifact (a run that emitted nothing, or lost a
	// tool_result pair) and then have the byte compare bless it.
	assertStreamContract(t, got)

	if *goldenUpdate {
		if err := os.MkdirAll(filepath.Dir(goldenPassPath), 0o755); err != nil {
			t.Fatalf("create %s: %v", filepath.Dir(goldenPassPath), err)
		}
		if err := os.WriteFile(goldenPassPath, got, 0o644); err != nil {
			t.Fatalf("write %s: %v", goldenPassPath, err)
		}
		t.Logf("wrote %s: %d bytes, %d event(s)", goldenPassPath, len(got), len(strings.Split(strings.TrimSuffix(string(got), "\n"), "\n")))
		return
	}

	want, err := os.ReadFile(goldenPassPath)
	if err != nil {
		t.Fatalf("read golden %s: %v\nregenerate it with: %s", goldenPassPath, err, goldenRegenerateCmd)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("%s drifted from what the Driver emits:\n%s\nregenerate with: %s", goldenPassPath, describeByteDrift(got, want), goldenRegenerateCmd)
	}
}

// goldenPassStream drives one Pass and returns its raw stdout. The scripted server
// supplies nine responses for the parent and one for the reviewer child; the bash
// commands are printf-only, one succeeds, one fails, and the same failing command
// runs a second time before the breaker refuses it on its third consecutive send,
// so nothing here reads the repo, the clock, or the environment beyond the run
// directory it writes into.
//
// The three failing calls carry distinct tool_calls ids over byte-identical
// arguments: the repeated-call breaker keys on the tool name and the raw
// arguments string rather than the id, and the log has to show that instead of
// having the test assert it.
//
// The script is keyed by model id rather than by POST index (goldenModelServer)
// because the two spawns share this one endpoint with their parent: the model in the
// request body is the only thing that tells a child call from the parent's.
//
// Only two responses are framed by hand. The first has to: its thinking arrives as
// the delta.reasoning_content chunks a real reasoning endpoint sends, which a
// whole completion body cannot express. The second has to as well: it is the
// chunked replay, prose split into delta.content fragments and one write call's
// arguments split across two frames, closed by the usage chunk a streaming
// endpoint sends last — none of which sseChatCompletion can produce from a whole
// body. The rest stay whole bodies through that helper, so the frames behind every
// other event are the ones this artifact has always pinned.
func goldenPassStream(t *testing.T) []byte {
	t.Helper()
	// A stepping clock, not a frozen one: the result event's durations have to be
	// non-zero to be worth pinning. The step is read-per-event, not wall clock, so
	// 250ms keeps the pinned numbers plausible rather than degenerate: a quarter
	// second between events, a quarter second per round trip, a several-second run.
	pinSteppingClock(t, 1700000000000, 250)
	// The children this run spawns are separate processes that pin their own clock
	// from this spec on the way in (pinChildClockFromEnv), which is the only thing
	// that makes a relayed child line byte-stable: the relay forwards the child's
	// stamp rather than restamping it.
	t.Setenv(envTestNowMS, fmt.Sprintf("%d:%d", goldenChildClockStart, goldenChildClockStep))
	// The roster the parent's Subagent calls resolve against, and the one endpoint the
	// parent and both children post to. The API key is a placeholder that names no
	// provider and is shaped like no credential: this artifact is the file a host
	// reads, and a run that records one is a leak.
	cfgPath := agentFixtureEntries(t,
		agentSpec{name: "reviewer", model: "m-reviewer", description: "reviews a diff", body: "review the change and report blockers"},
		agentSpec{name: "scout", model: "m-scout", description: "surveys a tree", body: "survey the tree and report what is in it"},
	)
	t.Setenv(envAPIKey, "golden-test-placeholder")

	// The file Tools really write and really read back, so they get a directory of
	// their own rather than the repo this test runs in. Their absolute tool_target is
	// why the stream is normalised through goldenRunDir before returning.
	runDir := t.TempDir()
	notesPath := filepath.Join(runDir, "notes.txt")
	// %q rather than hand-written escapes: Go's quoting of this body is legal JSON
	// quoting, so the arguments string needs no second pass to be parseable.
	writeArgs := fmt.Sprintf(`{"path":%q,"content":%q}`, notesPath, "# spindrift notes\nalpha line\nbeta line\n")
	editArgs := fmt.Sprintf(`{"path":%q,"old_string":"alpha line","new_string":"ALPHA line, replaced by the edit"}`, notesPath)
	// A Subagent call's arguments nest a JSON object inside a JSON string inside the
	// completion body, so they travel through %q too rather than a second escaping.
	reviewerArgs := `{"agent":"reviewer","task":"review the notes file for regressions"}`
	scoutArgs := `{"agent":"scout","task":"survey the repository tree"}`
	// The write call's arguments cut at its midpoint: merge() only concatenates
	// fragments, so the cut can fall anywhere — including inside a \n escape — and
	// the reassembled arguments still parse to the same call.
	writeArgsCut := len(writeArgs) / 2
	const failingArgs = `{"command":"printf 'this one fails'; exit 3"}`

	parentScript := []string{
		// Call 1: thinking in two fragments (so the event proves accumulation, not
		// one frame relabelled) ahead of prose with real newlines, which the emitter
		// must escape rather than trim, plus a bash call that succeeds.
		sseReasoningFrames("llama-3.1-8b",
			"reading the tree before saying anything about it; ",
			"one safe command first, then the file work") +
			sseChatCompletion(`{"model":"llama-3.1-8b","usage":{"prompt_tokens":104,"completion_tokens":21,"prompt_cache_hit_tokens":64},"choices":[{"message":{"role":"assistant","content":"I will look at the repo first,\nthen report.\n","tool_calls":[{"id":"call_alpha","type":"function","function":{"name":"bash","arguments":"{\"command\":\"printf 'hello world'\"}"}}]}}]}`),
		// Call 2: the chunked replay. Two delta.content fragments make one sentence,
		// which the text event has to carry joined rather than as two lines; then the
		// write call arrives as two tool_calls fragments, the id and name only in the
		// first; then usage on a choices-less chunk of its own.
		sseStream(
			`{"model":"llama-3.1-8b","choices":[{"index":0,"delta":{"content":"I will write the notes file now, "}}]}`,
			`{"choices":[{"index":0,"delta":{"content":"then patch one line of it."}}]}`,
			fmt.Sprintf(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_write","type":"function","function":{"name":"write","arguments":%q}}]}}]}`, writeArgs[:writeArgsCut]),
			fmt.Sprintf(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":%q}}]},"finish_reason":"tool_calls"}]}`, writeArgs[writeArgsCut:]),
			`{"model":"llama-3.1-8b","choices":[],"usage":{"prompt_tokens":152,"completion_tokens":34}}`,
		),
		// Call 3: no content at all (so no text event) and the edit of the file call 2
		// wrote — one old_string match, so replace_all stays off the arguments.
		sseChatCompletion(fmt.Sprintf(`{"model":"llama-3.1-8b","usage":{"prompt_tokens":196,"completion_tokens":24},"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_edit","type":"function","function":{"name":"edit","arguments":%q}}]}}]}`, editArgs)),
		// Call 4: the Subagent call that succeeds, with no prose of its own — what the
		// delegation produced arrives as relayed child lines inside the pair, and the
		// Tool Result carries the child's answer.
		sseChatCompletion(fmt.Sprintf(`{"model":"llama-3.1-8b","usage":{"prompt_tokens":240,"completion_tokens":16},"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_spawn_reviewer","type":"function","function":{"name":"subagent","arguments":%q}}]}}]}`, reviewerArgs)),
		// Call 5: the spawn whose child never gets an answer. Its model is the one the
		// server rate-limits, so its relayed block is a start and an error.
		sseChatCompletion(fmt.Sprintf(`{"model":"llama-3.1-8b","usage":{"prompt_tokens":284,"completion_tokens":16},"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_spawn_scout","type":"function","function":{"name":"subagent","arguments":%q}}]}}]}`, scoutArgs)),
		// Call 6: a command that exits 3.
		sseChatCompletion(fmt.Sprintf(`{"model":"llama-3.1-8b","usage":{"prompt_tokens":328,"completion_tokens":18},"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_beta","type":"function","function":{"name":"bash","arguments":%q}}]}}]}`, failingArgs)),
		// Call 7: the same tool name and the same arguments string under a fresh call
		// id, so the breaker is only at two in a row and the command really runs again.
		sseChatCompletion(fmt.Sprintf(`{"model":"llama-3.1-8b","usage":{"prompt_tokens":372,"completion_tokens":18},"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_beta_2","type":"function","function":{"name":"bash","arguments":%q}}]}}]}`, failingArgs)),
		// Call 8: identical a third time, so the breaker refuses it — the synthetic
		// Tool Result this log pins, handed out without bash ever being reached. One
		// refusal only: a third in a row would end the run on an error event.
		sseChatCompletion(fmt.Sprintf(`{"model":"llama-3.1-8b","usage":{"prompt_tokens":416,"completion_tokens":18},"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_beta_3","type":"function","function":{"name":"bash","arguments":%q}}]}}]}`, failingArgs)),
		// Call 9: the final answer, from a different model id, reporting cache read in
		// the OpenAI spelling.
		sseChatCompletion(`{"model":"qwen2.5-coder","usage":{"prompt_tokens":460,"completion_tokens":12,"prompt_tokens_details":{"cached_tokens":25}},"choices":[{"message":{"role":"assistant","content":"Wrote and edited notes.txt and delegated a review; the repeated command failed with exit code 3 and its third send was refused."}}]}`),
	}

	srv := newGoldenModelServer(t, map[string][]string{
		// The parent's own nine calls, in the order it makes them.
		goldenParentModel: parentScript,
		// The reviewer child's one call: prose plus usage and no tool call, so its whole
		// run is the start, text, usage and result block the relay forwards. Its prose
		// is its own words, nowhere near the parent's thinking fragments.
		"m-reviewer": {
			sseChatCompletion(`{"model":"m-reviewer","usage":{"prompt_tokens":512,"completion_tokens":32,"prompt_cache_hit_tokens":64},"choices":[{"message":{"role":"assistant","content":"Reviewed the notes file: the heading is intact and the replaced line reads cleanly; no blockers."}}]}`),
		},
		// goldenRateLimitedModel gets no queue at all: every POST it makes is a 429.
	})
	subagentEnv(t, cfgPath, srv.srv.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"inspect the repo and report", "--model", goldenParentModel, "--base-url", srv.srv.URL}, strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != 0 {
		t.Fatalf("golden Pass exited %d, want 0; stderr = %q", code, stderr.String())
	}
	// Nine parent calls and one per child — and the rate-limited child costs exactly
	// one POST, because a 429 returns without a retry rather than burning attempts.
	if got := srv.requestBodies(t); len(got) != 11 {
		t.Fatalf("golden Pass made %d Model call(s), want 11: nine parent plus one each for the reviewer and the scout", len(got))
	}
	// The file Tools resolve their path argument to an absolute path, so runDir is
	// written into every write and edit tool_target and tool_output. It is a
	// t.TempDir(), which differs between runs and between machines: swapping it for
	// goldenRunDir here — before both the -update write below and the byte compare —
	// is what lets one committed artifact hold on any host.
	return bytes.ReplaceAll(stdout.Bytes(), []byte(runDir), []byte(goldenRunDir))
}

// goldenParentModel is the model id the parent sends: the argv --model value, which
// is only ever a queue key and never the model field of a scripted response, so the
// ids the result event reports come from what answered rather than from argv.
const goldenParentModel = "spindrift-router"

// goldenModelServer scripts a relaying Pass. Parent and children share the one
// endpoint a spawn cannot redirect, so responses route on the model id in the
// request body: every model has its own FIFO of raw SSE bodies, and one model is
// answered with 429 on every POST it makes.
//
// An exhausted queue answers 500 naming the model rather than looping: a run that
// asked a question nobody scripted has to fail loudly on the turn that asked it.
type goldenModelServer struct {
	mu      sync.Mutex
	queues  map[string][]string
	limited string
	bodies  [][]byte
	srv     *httptest.Server
}

func newGoldenModelServer(t *testing.T, queues map[string][]string) *goldenModelServer {
	t.Helper()
	s := &goldenModelServer{queues: queues, limited: goldenRateLimitedModel}
	s.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		s.bodies = append(s.bodies, body)
		var req struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(body, &req)
		rateLimited := req.Model == s.limited
		frames, exhausted := "", false
		if !rateLimited {
			queue := s.queues[req.Model]
			if len(queue) == 0 {
				exhausted = true
			} else {
				s.queues[req.Model] = queue[1:]
				frames = queue[0]
			}
		}
		s.mu.Unlock()

		switch {
		case rateLimited:
			w.Header().Set("Retry-After", goldenRetryAfter)
			w.WriteHeader(http.StatusTooManyRequests)
			// Written out by hand rather than through http.Error: that helper appends a
			// newline, and the rejection body reaches the error event verbatim.
			io.WriteString(w, goldenRateLimitBody)
		case exhausted:
			http.Error(w, "no scripted response left for model "+req.Model, http.StatusInternalServerError)
		default:
			writeSSE(w, frames)
		}
	}))
	t.Cleanup(s.srv.Close)
	return s
}

func (s *goldenModelServer) requestBodies(t *testing.T) [][]byte {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([][]byte, len(s.bodies))
	copy(out, s.bodies)
	return out
}

// sseReasoningFrames scripts thinking as the delta.reasoning_content chunks a
// reasoning endpoint streams before any content. Fragments are prose written
// through %q, so a fragment containing a quote or a backslash would need
// escaping this helper does not attempt.
func sseReasoningFrames(model string, fragments ...string) string {
	var b strings.Builder
	for _, f := range fragments {
		b.WriteString(sseFrame(fmt.Sprintf(`{"model":%q,"choices":[{"index":0,"delta":{"reasoning_content":%q}}]}`, model, f)))
	}
	return b.String()
}

// eventStream names the clock one line's timestamp belongs to: the parent
// process's own, or the one the relayed child at that role is running on.
func eventStream(ev map[string]any) string {
	if ev["in_subagent"] == true {
		role, _ := ev["role"].(string)
		return "subagent " + role
	}
	return "parent"
}

// assertStreamContract checks what a byte compare cannot: that the file is a
// legal JSONL event stream and still the run this golden is meant to describe.
// Reading it as plain JSON is what issue #19 demands of a decoder — no driverkit
// import, no shared type.
func assertStreamContract(t *testing.T, raw []byte) {
	t.Helper()
	events := stdoutEvents(t, string(raw)) // every line one JSON object, no blanks, one trailing newline each
	// Monotonicity is per stream rather than global: the relay keeps the child's own
	// timestamp (events.go:25-26), so the parent and every spawned child stamp on the
	// grid pinned for that process alone, and cross-process ordering is not a
	// property this artifact pins.
	lastTS := map[string]int64{}
	kinds := make([]string, 0, len(events))
	for i, ev := range events {
		kind, ok := ev["kind"].(string)
		if !ok || kind == "" {
			t.Errorf("event %d has no non-empty kind: %v", i+1, ev)
			continue
		}
		kinds = append(kinds, kind)
		if _, ok := ev["timestamp"]; !ok {
			t.Errorf("%s event %d carries no timestamp field", kind, i+1)
		}
		ts, ok := ev["timestamp"].(float64)
		if !ok || ts <= 0 {
			t.Errorf("%s event %d timestamp = %v, want epoch milliseconds", kind, i+1, ev["timestamp"])
			continue
		}
		stream := eventStream(ev)
		if prev, seen := lastTS[stream]; seen && int64(ts) < prev {
			t.Errorf("event %d timestamp %d goes backwards from %d within the %s stream", i+1, int64(ts), prev, stream)
		}
		lastTS[stream] = int64(ts)
	}

	// The two spawn turns are the only lines that are not the parent's own: a relayed
	// child block sits inside the tool_call/tool_result pair it was spawned for, and
	// the scout's block ends on an error rather than a result.
	wantKinds := []string{
		"start", "reasoning", "text", "usage", "tool_call", "tool_result",
		"start", "text", "usage", "tool_call", "tool_result",
		"start", "usage", "tool_call", "tool_result",
		"start", "usage", "tool_call", "start", "text", "usage", "result", "tool_result",
		"start", "usage", "tool_call", "start", "error", "tool_result",
		"start", "usage", "tool_call", "tool_result",
		"start", "usage", "tool_call", "tool_result",
		"start", "usage", "tool_call", "tool_result",
		"start", "text", "usage", "result",
	}
	if strings.Join(kinds, ",") != strings.Join(wantKinds, ",") {
		t.Fatalf("kind sequence = %s, want %s", strings.Join(kinds, ","), strings.Join(wantKinds, ","))
	}

	// The artifact is machine-bound the moment the run's own directory survives into
	// it, and the only paths a Pass can report come from the file Tools — so a line
	// naming the test that made the temp dir means normalisation did not happen.
	if strings.Contains(string(raw), t.Name()) {
		t.Errorf("the stream names %q, so the run directory was not normalised to %s", t.Name(), goldenRunDir)
	}

	// The tool_result shapes the golden exists to pin: clean ones — a command that
	// exited 0, the two file Tools reporting a sentence rather than command output,
	// and a spawn whose child answered — two is_error reporting a real exit code, one
	// is_error for a child that was rate-limited, and one is_error refusal that never
	// ran and so carries no exit-code line at all.
	var failed, clean, refused int
	for _, ev := range events {
		if ev["kind"] != "tool_result" {
			continue
		}
		isErr, present := ev["is_error"]
		if !present {
			clean++
			continue
		}
		if isErr != true {
			t.Errorf("is_error present but %v: omitempty must keep a clean result off the wire entirely", isErr)
		}
		failed++
		// The repeated-call refusal is a third shape, not just another failure: the
		// command never reached bash, so the only thing a decoder can read is the
		// "error: " prefix and the demand for a different action.
		output, _ := ev["tool_output"].(string)
		if !strings.Contains(output, "exit code:") && strings.Contains(output, "take a different action") {
			refused++
		}
	}
	if clean != 4 || failed != 4 {
		t.Errorf("tool_result shapes = %d clean and %d is_error, want 4 clean and 4 is_error", clean, failed)
	}
	if refused != 1 {
		t.Errorf("%d tool_result(s) read as the synthetic refusal, want exactly one: is_error true, no exit code: line, and text asking for a different action", refused)
	}

	// The file Tool pairs the widened scenario exists to pin: one write and one edit
	// of the same file, each as a tool_call/tool_result pair whose tool_target is the
	// normalised absolute path — the proof that the substitution ran, since a real
	// temp path here would differ on the next machine to regenerate.
	var fileToolLines int
	for _, ev := range events {
		name, _ := ev["tool_name"].(string)
		if (ev["kind"] != "tool_call" && ev["kind"] != "tool_result") || (name != "write" && name != "edit") {
			continue
		}
		fileToolLines++
		if wantTarget := goldenRunDir + "/notes.txt"; ev["tool_target"] != wantTarget {
			t.Errorf("%s %s tool_target = %v, want %s", ev["kind"], name, ev["tool_target"], wantTarget)
		}
	}
	if fileToolLines != 4 {
		t.Errorf("%d file Tool lines, want 4: the write and the edit, each as a tool_call/tool_result pair", fileToolLines)
	}

	// What the two spawns owe the artifact: relayed lines are the only ones carrying
	// role and in_subagent, the Subagent Tool Results are the only ones carrying
	// subagent, and exactly one line carries reset_at.
	relayed := relayedEvents(t, string(raw))
	roles := map[string]int{}
	for _, ev := range relayed {
		role, ok := ev["role"].(string)
		if !ok || role == "" {
			t.Errorf("relayed %v line carries no role: role and in_subagent travel together", ev["kind"])
			continue
		}
		roles[role]++
	}
	for _, name := range []string{"reviewer", "scout"} {
		if roles[name] == 0 {
			t.Errorf("no relayed line is attributed to %q: %v", name, roles)
		}
	}
	if len(roles) != 2 {
		t.Errorf("%d relayed roles %v, want exactly reviewer and scout", len(roles), roles)
	}

	var spawned []string
	var spawnClean, spawnFailed, resetLines, parentErrors int
	for _, ev := range events {
		if ev["in_subagent"] != true {
			if _, ok := ev["role"]; ok {
				t.Errorf("parent %v line carries role=%v: attribution belongs to relayed lines only", ev["kind"], ev["role"])
			}
			if _, ok := ev["in_subagent"]; ok {
				t.Errorf("parent %v line carries in_subagent: it is set on the relay path and nowhere else", ev["kind"])
			}
			if ev["kind"] == "error" {
				parentErrors++
			}
		}
		if ev["kind"] == "tool_result" && ev["tool_name"] == "subagent" {
			name, _ := ev["subagent"].(string)
			if name == "" {
				t.Errorf("Subagent Tool Result carries no subagent field: %v", ev)
				continue
			}
			spawned = append(spawned, name)
			// planSubagentCall names both tool_target and subagent after the entry.
			if ev["tool_target"] != name {
				t.Errorf("subagent %q Tool Result tool_target = %v, want the roster entry name as well", name, ev["tool_target"])
			}
			if ev["is_error"] == true {
				spawnFailed++
			} else {
				spawnClean++
			}
		}
		if _, present := ev["reset_at"]; present {
			resetLines++
			if ev["kind"] != "error" {
				t.Errorf("%v line carries reset_at: only the error kind reports a reset", ev["kind"])
			}
		}
	}
	if got := strings.Join(spawned, ","); got != "reviewer,scout" {
		t.Errorf("Subagent Tool Results report subagent %q, want reviewer then scout", got)
	}
	if spawnClean != 1 || spawnFailed != 1 {
		t.Errorf("%d clean and %d is_error Subagent Tool Results, want the reviewer's answer and the scout's failure", spawnClean, spawnFailed)
	}
	if resetLines != 1 {
		t.Errorf("%d lines carry reset_at, want exactly one: the relayed 429 the scout child hit", resetLines)
	}
	if parentErrors != 0 {
		t.Errorf("%d parent lines carry the error kind, want none: a Pass that errors never writes its result", parentErrors)
	}

	// The scout's rejection is the artifact's only rate limit, and its reset instant
	// is the absolute HTTP-date the scripted 429 named — absolute so that it does not
	// move with whichever tick of its own grid the child had reached.
	resetAt, err := http.ParseTime(goldenRetryAfter)
	if err != nil {
		t.Fatalf("goldenRetryAfter %q is not an HTTP-date: %v", goldenRetryAfter, err)
	}
	var rateLimitedText, reviewerAnswer, delegatedOutput string
	for _, ev := range relayed {
		switch {
		case ev["kind"] == "error":
			rateLimitedText, _ = ev["text"].(string)
			if ms, _ := ev["reset_at"].(float64); int64(ms) != resetAt.UnixMilli() {
				t.Errorf("relayed error reset_at = %v, want %d, the instant %q names", ev["reset_at"], resetAt.UnixMilli(), goldenRetryAfter)
			}
		case ev["kind"] == "result" && ev["role"] == "reviewer":
			reviewerAnswer, _ = ev["text"].(string)
		}
	}
	if !strings.Contains(rateLimitedText, "429") || !strings.Contains(rateLimitedText, goldenRateLimitedModel) {
		t.Errorf("relayed error text = %q, want the scout's 429 with the rate-limited model named in the body", rateLimitedText)
	}
	// A successful delegation hands the child's answer up: the parent's Tool Result is
	// the child's own result text, verbatim.
	for _, ev := range events {
		if ev["kind"] == "tool_result" && ev["tool_name"] == "subagent" && ev["subagent"] == "reviewer" {
			delegatedOutput, _ = ev["tool_output"].(string)
		}
	}
	if reviewerAnswer == "" || delegatedOutput != reviewerAnswer {
		t.Errorf("reviewer child result text %q is not the parent's Subagent Tool Result %q", reviewerAnswer, delegatedOutput)
	}

	// parentResult rather than the last line: a relaying run carries the child's result
	// too and in_subagent is the only thing that separates them — and it pins that the
	// last line is the parent's own, which is another way of saying this Pass finished.
	result := parentResult(t, string(raw))
	if turns, _ := result["num_turns"].(float64); int(turns) != 9 {
		t.Errorf("result num_turns = %v, want 9 parent Model calls", result["num_turns"])
	}
	for _, name := range []string{"duration_ms", "duration_api_ms"} {
		if v, _ := result[name].(float64); v <= 0 {
			t.Errorf("result %s = %v, want a non-zero pinned duration", name, result[name])
		}
	}
	tokens, _ := result["tokens"].(map[string]any)
	// The parent's own nine calls — 104+152+196+240+284+328+372+416+460 input,
	// 21+34+24+16+16+18+18+18+12 output, and 64 (llama.cpp spelling) + 25 (OpenAI
	// details spelling) cache read — plus the reviewer child's 512/32/64, which the
	// relay folds in per usage event. The refused call counts because the Model was
	// charged for it; the scout contributes nothing, because a 429 leaves no usage to
	// fold.
	for name, want := range map[string]float64{"input": 3064, "output": 209, "cache_read": 153} {
		if got, _ := tokens[name].(float64); got != want {
			t.Errorf("result tokens[%s] = %v, want the nine parent calls and the delegated reviewer summed to %v", name, tokens[name], want)
		}
	}
	models, _ := result["models"].([]any)
	if len(models) != 3 || models[0] != "llama-3.1-8b" || models[1] != "m-reviewer" || models[2] != "qwen2.5-coder" {
		t.Errorf("result models = %v, want the three ids that answered, sorted: m-scout never got past its 429, so it is absent", result["models"])
	}

	// The thinking the golden exists to pin: exactly one event, on the call that
	// streamed it, riding ahead of the text event it produced.
	thinkingAt := -1
	var thinking string
	for i, ev := range events {
		if ev["kind"] != "reasoning" {
			continue
		}
		if thinkingAt >= 0 {
			t.Errorf("second reasoning event %d: one message's fragments belong in one event", i+1)
			continue
		}
		thinkingAt = i
		thinking, _ = ev["text"].(string)
		if ev["message_id"] != "msg-1" {
			t.Errorf("reasoning event message_id = %v, want msg-1, the one call that thinks", ev["message_id"])
		}
	}
	if thinking == "" {
		t.Fatalf("no reasoning event carrying thinking: the first scripted call must stream it")
	}
	if next := thinkingAt + 1; next >= len(events) || events[next]["kind"] != "text" {
		t.Errorf("reasoning event %d does not ride ahead of a text event", thinkingAt+1)
	}
	// Thinking is its own channel: neither the prose event nor the result event may
	// carry any of it, or a decoder that reads only the answer gets it twice. This
	// walk now covers the relayed child prose and the child's result as well, which is
	// why the reviewer's answer shares no wording with the fragments above.
	for _, ev := range events {
		if ev["kind"] != "text" && ev["kind"] != "result" {
			continue
		}
		if prose, _ := ev["text"].(string); prose != "" && strings.Contains(prose, thinking) {
			t.Errorf("%s text %q carries the thinking, which stays out of the answer", ev["kind"], prose)
		}
	}
}

// describeByteDrift points at the first differing line rather than dumping two
// whole streams: the reader needs to know which event moved, not re-diff a dozen
// lines by eye.
func describeByteDrift(got, want []byte) string {
	g := strings.Split(string(got), "\n")
	w := strings.Split(string(want), "\n")
	for i := 0; i < len(g) || i < len(w); i++ {
		var a, b string
		if i < len(g) {
			a = g[i]
		}
		if i < len(w) {
			b = w[i]
		}
		if a != b {
			return fmt.Sprintf("line %d:\n  emitted: %s\n  golden:  %s\n(emitted %d bytes, golden %d bytes)", i+1, a, b, len(got), len(want))
		}
	}
	return fmt.Sprintf("bytes equal but framing differs (emitted %d bytes, golden %d bytes)", len(got), len(want))
}
