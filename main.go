package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
)

const defaultBaseURL = "http://localhost:8080/v1"

// defaultMaxTurns is the built-in cap on the agent loop, named so the flag
// default and the usage text read one value rather than two that can drift.
const defaultMaxTurns = 300
const envBaseURL = "SPINDRIFT_BASE_URL"
const envAPIKey = "SPINDRIFT_API_KEY"

// systemPrompt is deliberately minimal and hardcoded: the Model gets repo context
// via the user prompt.
const systemPrompt = "You are a coding agent working in this repository. Answer with the final response only; it is the Driver's final result event."

type config struct {
	model    string
	baseURL  string
	prompt   string
	maxTurns int
	timeout  time.Duration
	// apiKey comes from the environment or the Driver Config by design: a
	// --api-key flag would put the secret in argv, where process listings and
	// shell history expose it.
	apiKey string
	// agentsDir is the Agent Files directory: the Driver Config's agents_dir, or
	// the home-relative default, or "" when neither resolves — which is "no
	// Agent Files", never an error.
	agentsDir string
	// agents is the roster loaded from agentsDir at startup, empty when there is
	// none. tools is what this run offers the Model, derived from that roster by
	// offeredTools so the offer and the roster cannot disagree later.
	agents []agentEntry
	tools  []toolSpec
	// hooks is the Guard Hook list read from the Driver Config's hook_manifest
	// once at startup. An empty list means no Hook Manifest is configured, so
	// nothing is guarded: unlike agentsDir the manifest has no home-relative
	// default, so an absent key has nothing to fall back to.
	hooks []guardHook
	// summary is the Hook Manifest's one post-call entry, nil when no Hook
	// Manifest is configured or the configured one names no post call. Nothing
	// reads it yet: the slice that rewrites a Tool Result with it is the next one.
	summary *summaryHook
	// agent is the roster entry this run was started as by --agent, "" for a
	// top-level run. It is the fact that makes a run a child, and the tool offer
	// is read with it: a child never gets the Subagent Tool.
	agent string
	// systemPrompt is the selected Agent File's body, the system prompt that
	// entry's run starts from. "" means no entry, so the package default applies
	// — which is every hand-built config's case as much as a top-level run's.
	systemPrompt string
}

// usage is a var, not a const: the --timeout and --max-turns defaults are
// interpolated from chatAttemptTimeout and defaultMaxTurns so the text cannot
// drift from the values in effect.
var usage = `usage: spindrift-driver "prompt" --model <model> [--base-url <url>] [--max-turns <n>] [--timeout <dur>] [--agent <name>]

The prompt may be given positionally or on stdin. --model is required.
Precedence for every setting: flag, then environment, then the Driver Config, then the built-in default.
The Driver Config is JSON at the path $` + envConfigPath + ` names, else $HOME/` + defaultDriverConfigPath + `; it carries base_url, api_key, timeout, agents_dir and hook_manifest.
An absent file at the default path is not an error; a missing file that $` + envConfigPath + ` names, or any file that cannot be
parsed, is a startup error, never a fallback.
A configured hook_manifest lists the Guard Hooks every Bash Tool Call runs through; one that cannot be read is a startup error, never an unguarded run.
The API key comes from $` + envAPIKey + ` or that file, never from argv.
--base-url defaults to $SPINDRIFT_BASE_URL or ` + defaultBaseURL + `.
--max-turns caps the number of agent turns (default ` + fmt.Sprint(defaultMaxTurns) + `).
--agent <name> runs as that Agent File entry: its model id names the model unless --model
gives one, and the file body replaces the system prompt. A run selected this way is never offered
the Subagent Tool, so a subagent cannot spawn another.
--timeout is an idle bound on one chat attempt: the dial, the response headers and the
gap between received bytes, default ` + chatAttemptTimeout.String() + `. A stream that keeps
arriving may run far past it; raise it (--timeout 15m) only when the server thinks for a
long time before its first token, or when a proxy buffers the response.
`

// run is the process-boundary seam: tests drive it in-process, main wraps it.
func run(args []string, stdin io.Reader, stdout, stderr io.Writer, lookupEnv func(string) (string, bool)) int {
	events := eventWriter{w: stdout}
	cfg, err := parseArgs(args, stdin, lookupEnv)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) { // -h/--help: usage is the output, not an error
			// It goes to stderr because stdout carries only JSONL events.
			fmt.Fprint(stderr, usage)
			return 0
		}
		fmt.Fprintf(stderr, "spindrift-driver: %v\n%s", err, usage)
		return errorExit(events, 2, err)
	}
	// Autonomous tool loop: each turn posts the whole conversation with the
	// tools parameter; the server decides tool calls (never parsed from text).
	// The empty-field fallback is what keeps a run that came from anywhere other
	// than parseArgs — and therefore has no systemPrompt of its own — on the
	// package prompt it has always used.
	sys := cfg.systemPrompt
	if sys == "" {
		sys = systemPrompt
	}
	messages := []chatMessage{
		{Role: "system", Content: sys},
		{Role: "user", Content: cfg.prompt},
	}
	// Run bookkeeping for the result event, all of it local: the Model-call
	// index (which is the msg-<n> id every event of that call shares), the
	// instant the run began, the accumulated round-trip milliseconds, the summed
	// token tally and the set of model ids that answered.
	call := 0
	// toolSeq counts Tool Calls across the whole run, so a server that sends no
	// call id still gets pair keys that stay unique turn after turn.
	toolSeq := 0
	runStart := nowMS()
	var apiMS int64
	var totals tokenTotals
	// delegated sums what the child Drivers this run spawns spend, kept apart from
	// the parent's own tally so its usage events stay one per own Model call.
	var delegated tokenTotals
	seenModels := map[string]struct{}{}
	sess := session{cfg: cfg, events: events, stderr: stderr, delegated: &delegated, models: seenModels}
	// Repeated-call breaker state: the last call seen and how many times in a row.
	// These are run() locals rather than package counters because a package counter
	// would carry across every run() in one process, and the state is per run.
	var lastKey string
	repeat := 0
	// refusals counts how many calls in a row the breaker refused: a call that
	// actually ran means the Model changed course, so the count starts over.
	refusals := 0
	for turn := 0; turn < cfg.maxTurns; turn++ {
		call++
		messageID := fmt.Sprintf("msg-%d", call)
		// start goes out before the request exists, so its timestamp precedes the
		// first byte and its message id can only be Driver-assigned (#20).
		_ = events.emit(event{Kind: "start", MessageID: messageID})
		apiStart := nowMS()
		reply, err := sendChat(cfg, messages, stderr)
		apiMS += nowMS() - apiStart
		if err != nil {
			// A call that failed has no usage to report, so the stream stops at
			// start here; errorExit adds the error event.
			fmt.Fprintf(stderr, "spindrift-driver: %v\n", err)
			return errorExit(events, 1, err)
		}
		assistant := reply.Message
		// Reasoning goes out ahead of the prose it produced, as one event per
		// completed message like text is, and rides Text because the contract has no
		// second prose field — a reader tells the two apart by kind alone.
		if reply.Reasoning != "" {
			_ = events.emit(event{Kind: "reasoning", MessageID: messageID, Text: reply.Reasoning})
		}
		if assistant.Content != "" {
			// Verbatim: no trimming and no newline normalisation — the trailing
			// newline JSONL needs is the emitter's, never part of the text.
			_ = events.emit(event{Kind: "text", MessageID: messageID, Text: assistant.Content})
		}
		callTokens := reply.Tokens
		totals = totals.add(callTokens)
		seenModels[reply.Model] = struct{}{}
		// Exactly one usage per Model call and never a running total: the reader
		// sums these, and a message id is what lets it dedup.
		_ = events.emit(event{Kind: "usage", MessageID: messageID, Model: reply.Model, Tokens: &callTokens})
		if len(assistant.ToolCalls) == 0 {
			// No tool calls: the content is the final answer, reported both as the
			// prose event above and as the authoritative result event.
			_ = events.emitResult(resultPayload{
				Kind:          "result",
				Text:          assistant.Content,
				NumTurns:      call,
				DurationMs:    nowMS() - runStart,
				DurationAPIMs: apiMS,
				Tokens:        totals.add(delegated),
				Models:        sortedModelIDs(seenModels),
			})
			return 0
		}
		// OpenAI convention: replay the assistant message verbatim (with its
		// tool_calls), then one tool message per call.
		messages = append(messages, assistant)
		for _, tc := range assistant.ToolCalls {
			toolSeq++
			toolID := tc.ID
			if toolID == "" {
				toolID = fmt.Sprintf("call-%d", toolSeq)
			}
			plan := planToolCall(tc, sess)
			// tool_call goes out before the command starts: a command can run for
			// minutes, and a reader has to see the call at the moment the Model made
			// it. The pair stays adjacent, and both lines carry the message id of the
			// Model call that asked for them.
			_ = events.emit(event{
				Kind:       "tool_call",
				MessageID:  messageID,
				ToolID:     toolID,
				ToolName:   tc.Function.Name,
				ToolInput:  plan.input,
				ToolTarget: plan.target,
			})
			// The key is the call exactly as the Model sent it: the tool name and the
			// raw arguments wire string, joined by a NUL neither can contain. Keying on
			// the decoded arguments would follow Go map order instead of what was sent.
			key := tc.Function.Name + "\x00" + tc.Function.Arguments
			if key == lastKey {
				repeat++
			} else {
				lastKey, repeat = key, 1
			}
			var output string
			refused := repeat >= repeatedCallRefuseLimit
			if refused {
				refusals++
				output = repeatedCallRefusal(tc.Function.Name, repeat)
			} else {
				// Only the breaker's own refusals chain. A call a Guard Hook blocked comes
				// back through plan.run() like any other result, and the Model does get to
				// react to it, so it breaks the streak rather than feeding the counter.
				refusals = 0
				output = plan.run()
			}
			result := toolResultEvent(messageID, toolID, tc.Function.Name, plan, output)
			if refused {
				// The plan's own rule cannot judge a refusal: the bash rule closes over a
				// verdict only plan.run() writes, and a refused call never got there.
				result.IsError = true
			}
			_ = events.emit(result)
			if refusals >= repeatedCallEndLimit {
				// Returning before the replay leaves the refusal out of the conversation
				// the run is ending over, and any later call of this batch never got a
				// tool_call, so no pair is left open by the early exit.
				breaker := fmt.Errorf("repeated the same %s call so often it was refused %d times in a row, so the run ended", tc.Function.Name, refusals)
				fmt.Fprintf(stderr, "spindrift-driver: %v\n", breaker)
				return errorExit(events, 1, breaker)
			}
			// The replayed Tool Result message is unchanged, idless calls included:
			// the synthesised tool id belongs to the event stream, not to the Model.
			messages = append(messages, chatMessage{Role: "tool", Content: output, ToolCallID: tc.ID})
		}
	}
	turnLimit := fmt.Errorf("reached the turn limit of %d without a final answer", cfg.maxTurns)
	fmt.Fprintf(stderr, "spindrift-driver: %v\n", turnLimit)
	return errorExit(events, 1, turnLimit)
}

// sortedModelIDs returns the run's distinct model ids in sorted order: the
// result event names them as a set, and a sorted slice keeps the stream
// byte-stable for a decoder (and for a golden log) instead of following
// map iteration order.
func sortedModelIDs(seen map[string]struct{}) []string {
	out := make([]string, 0, len(seen))
	for id := range seen {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// errorExit is the one funnel every non-zero exit passes through, so no failure
// path can return without the error event stdout owes. The event text is the
// raw error verbatim, never flattened through boundBody: that cap exists so the
// stderr progress line stays one line, and JSON escaping already keeps this
// line whole however many newlines the error carries. A failed stdout write has
// nowhere to be reported and changes nothing — stderr already carries the
// operator's copy, and the exit code stays the authoritative signal.
func errorExit(events eventWriter, code int, err error) int {
	e := event{Kind: "error", Text: err.Error()}
	// The reset instant rides the error value rather than a parameter so the call
	// sites that have no reset keep their exact call shape.
	var rej *rejection
	if errors.As(err, &rej) {
		e.ResetAt = rej.resetAt
	}
	_ = events.emit(e)
	return code
}

// rejection is a non-retryable HTTP rejection captured whole: the status line,
// the body exactly as it arrived, and the reset instant when the server named
// one. A plain fmt.Errorf cannot carry the last of those to errorExit, which is
// the only place that knows how to stamp an event field. Error reproduces the
// long-standing wording so stderr and the error event read as they always did.
type rejection struct {
	status string
	body   string
	// resetAt is absolute epoch milliseconds, 0 when the server named no reset.
	resetAt int64
}

func (r *rejection) Error() string {
	return "server returned " + r.status + ": " + r.body
}

// retryAfterMS turns a Retry-After header into an absolute epoch millisecond:
// delta-seconds first (the form a rate limiter answers with), then an HTTP-date,
// which is already absolute and so ignores now. Deltas go through the caller's
// nowMS reading, never time.Now, or a golden log could never compare byte for
// byte. A header that is absent, unparsable or negative names no reset, and so
// does a delta too large to add to now without wrapping: a wrapped instant is
// indistinguishable from a real one to whatever reads the event.
func retryAfterMS(h http.Header, now int64) (int64, bool) {
	v := strings.TrimSpace(h.Get("Retry-After"))
	if v == "" {
		return 0, false
	}
	if secs, err := strconv.ParseInt(v, 10, 64); err == nil {
		if secs < 0 || secs > (math.MaxInt64-now)/1000 {
			return 0, false
		}
		return now + secs*1000, true
	}
	if t, err := http.ParseTime(v); err == nil {
		return t.UnixMilli(), true
	}
	return 0, false
}

// toolPlan is one assistant tool call read before it runs. The tool_call event
// has to carry the decoded arguments and the target while the command has yet to
// start, so the read cannot live on the far side of execution.
type toolPlan struct {
	// input is the arguments JSON decoded to a value, or the raw arguments
	// string when they are not valid JSON at all.
	input any
	// target is the input field that best identifies what the call acted on —
	// the command for bash — and "" when the tool is unknown or the arguments
	// cannot be parsed, which is the contract's way of saying "unknown".
	target string
	// subagent is the agent a Subagent Tool call acted on, and "" for every other
	// tool, so the subagent key stays off every ordinary tool_result line.
	subagent string
	// isError is the rule that decides is_error for this plan's output; nil means
	// toolResultIsError, so a tool whose verdict is not that suffix test states its
	// own: a Subagent Tool Result is the child's answer, and a Bash Tool Result is
	// judged on what the command exited with, before any summary hook replaces the
	// text.
	isError func(output string) bool
	// run produces the exact Tool Result string that goes back into the
	// conversation.
	run func() string
}

// session is what a tool call needs from the run that made it. A Subagent Tool
// call reaches further than the config — it relays onto the run's own event
// stream and charges the spend it delegates back to the run — so the run hands
// over one of these instead of another parameter each time.
type session struct {
	cfg    config
	events eventWriter
	stderr io.Writer
	// delegated sums child spend behind a pointer because the session travels into
	// a plan closure by value: a field of its own would add to a copy.
	delegated *tokenTotals
	// models is the run's model-id set, which a child's model id joins here.
	models map[string]struct{}
}

// toolResultEvent is the tool_result half of the pair. The plan brings its own
// is_error rule because the exit-code suffix rule describes a command and
// nothing else: a result with no exit-code line — a subagent's answer — would
// otherwise be reported as a failure on every success.
func toolResultEvent(messageID, toolID, toolName string, plan toolPlan, output string) event {
	isError := plan.isError
	if isError == nil {
		isError = toolResultIsError
	}
	return event{
		Kind:       "tool_result",
		MessageID:  messageID,
		ToolID:     toolID,
		ToolName:   toolName,
		ToolTarget: plan.target,
		ToolOutput: output,
		IsError:    isError(output),
		Subagent:   plan.subagent,
	}
}

// repeatedCallRefuseLimit is how many identical calls in a row the breaker lets
// run before it starts refusing: the third gets the synthetic Tool Result in
// place of running. The two 3s of the breaker count different things — calls and
// refusals — so each gets its own name rather than one shared literal that would
// tie two independent limits together.
const repeatedCallRefuseLimit = 3

// repeatedCallEndLimit is how many refusals in a row the breaker hands out
// before it ends the run: past that many identical calls the refusal is clearly
// not being heeded, and spending turns on it only repeats itself.
const repeatedCallEndLimit = 3

// repeatedCallRefusal is the synthetic Tool Result a refused call gets in place
// of running. It carries the "error: " prefix the Guard Hook refusal established
// as how a refusal reads, and names the call so the Model can tell which action
// to change.
func repeatedCallRefusal(toolName string, repeat int) string {
	return fmt.Sprintf("error: %s was already called %d times in a row with these exact arguments, so it was not run again; take a different action", toolName, repeat-1)
}

// planToolCall parses one tool call without executing it. An unknown tool and
// unparseable arguments are ordinary results rather than crashes, so the plan
// answers with the very error text the Model has always received.
//
// A Subagent Tool call needs the run itself — the roster to resolve the name
// against, the event writer to relay the child's lines onto, and the tallies to
// charge its spend to — so the run passes a session rather than a global.
func planToolCall(call toolCall, s session) toolPlan {
	input := decodeToolArguments(call.Function.Arguments)
	if call.Function.Name == "subagent" {
		return planSubagentCall(call, input, s)
	}
	if call.Function.Name != "bash" {
		name := call.Function.Name
		return toolPlan{input: input, run: func() string { return fmt.Sprintf("error: unknown tool %q", name) }}
	}
	var args struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return toolPlan{input: input, run: func() string { return fmt.Sprintf("error: cannot parse tool arguments: %v", err) }}
	}
	// The Guard chain runs when the call runs, not when it is planned: a hook is a
	// script bounded by guardHookTimeout, and planning must stay parsing so the
	// tool_call event is already on the wire before any hook fires. The refusal is
	// what the Model is told, and its "error: " prefix is what toolResultIsError
	// already reads as a failed Tool Result — so the refusal needs no new event
	// grammar, only that runBashTool is never reached. A rewrite is host policy on
	// what executes, so it replaces the command runBashTool is handed while
	// tool_target keeps reporting the command the Model asked for.
	// is_error stays the command's own verdict: a summary replaces the Tool Result
	// text and has no exit-code line of its own, so letting toolResultIsError read
	// the summary would flip every summarised success to an error. The rule closes
	// over failed, which run writes from the pre-summary result — an ordering that
	// is load bearing, because the run loop calls plan.run() before it hands the
	// output to toolResultEvent.
	var failed bool
	return toolPlan{
		input:   input,
		target:  args.Command,
		isError: func(string) bool { return failed },
		run: func() string {
			command, refusal := guardCommand(args.Command, call.Function.Name, s.cfg.hooks, s.stderr)
			if refusal != "" {
				// Judged by the same rule so a blocked call keeps reporting is_error true
				// off its "error: " prefix. A refused call never ran, so there is no Tool
				// Result to summarise and the summary hook is not asked.
				failed = toolResultIsError(refusal)
				return refusal
			}
			result := runBashTool(command)
			failed = toolResultIsError(result)
			return summarizeToolOutput(call.Function.Name, command, result, s.cfg.summary, s.stderr)
		},
	}
}

// decodeToolArguments yields the value the tool_call event carries. Arguments
// that are not valid JSON come back as the raw string instead of nothing: a
// malformed call is the fact worth reporting, and its text is what the Model is
// told failed.
func decodeToolArguments(raw string) any {
	var v any
	if err := json.Unmarshal([]byte(raw), &v); err != nil {
		return raw
	}
	return v
}

// toolResultIsError is the is_error test: a Tool Result always closes with an
// exit-code line, so anything that is not a clean 0 — a non-zero exit, a signal,
// a timeout, or an error prefix that never ran a command — reads as the tool
// having failed. A succeeding command leaves is_error false, which omitempty
// keeps off the wire entirely.
//
// Gotcha: the clean-0 literal lives in runBashTool's success return, not here,
// and this match is a suffix test on the last line it appends. Reword that
// suffix there and every tool still planned by this rule flips to is_error true
// silently — the two sites have to change together. toolResultEvent is the
// third: it is where a plan opts out of this rule, so a new tool that needs a
// different verdict sets toolPlan.isError rather than bending the suffix here.
func toolResultIsError(output string) bool {
	return !strings.HasSuffix(output, "\nexit code: 0")
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr, os.LookupEnv))
}

func parseArgs(args []string, stdin io.Reader, lookupEnv func(string) (string, bool)) (config, error) {
	fs := flag.NewFlagSet("spindrift-driver", flag.ContinueOnError)
	fs.SetOutput(io.Discard) // usage errors are reported by run, keep parse silent
	baseURL := fs.String("base-url", "", "OpenAI-compatible API base URL (default: $SPINDRIFT_BASE_URL or "+defaultBaseURL+")")
	model := fs.String("model", "", "model name (required)")
	maxTurns := fs.Int("max-turns", defaultMaxTurns, "maximum agent turns")
	agentName := fs.String("agent", "", "run as this Agent File entry: its model id is used unless --model names one, and its body is the system prompt")
	timeout := fs.Duration("timeout", chatAttemptTimeout, "idle timeout for one chat request: caps the dial, the response headers and the gap between received bytes, not the whole generation; accepts Go durations such as 90s or 2m")
	pre, post := splitAtEndOfFlags(args)
	if err := fs.Parse(reorderFlagsFirst(pre)); err != nil {
		return config{}, err
	}
	// fs.Visit sees only the flags actually given: --timeout defaults to
	// chatAttemptTimeout rather than the zero value, so *timeout alone cannot
	// tell an unset flag from one the operator typed -- and the Driver Config
	// must lose to a value that was typed.
	timeoutSet := false
	fs.Visit(func(f *flag.Flag) { timeoutSet = timeoutSet || f.Name == "timeout" })
	positional := append(fs.Args(), post...)
	if len(positional) > 1 {
		return config{}, fmt.Errorf("unexpected argument %q: only one prompt is accepted", positional[1])
	}

	// read before any validation: a Driver Config the operator cannot fix makes
	// the rest of the arguments moot
	file, err := readDriverConfig(lookupEnv)
	if err != nil {
		return config{}, err
	}

	if *maxTurns < 1 {
		return config{}, fmt.Errorf("--max-turns must be at least 1, got %d", *maxTurns)
	}
	if timeoutSet && *timeout <= 0 {
		return config{}, fmt.Errorf("--timeout must be positive, got %v", *timeout)
	}

	cfg := config{
		model:    *model,
		baseURL:  *baseURL,
		maxTurns: *maxTurns,
		timeout:  *timeout,
	}
	// the file never carries a non-positive duration (readDriverConfig rejects
	// those), so the unset-flag case still lands on chatAttemptTimeout
	if !timeoutSet && file.timeout != 0 {
		cfg.timeout = file.timeout
	}
	// flag > env > Driver Config > built-in default; a set-but-empty env value
	// reads as unset, so it never blanks out a value the file supplied
	if cfg.baseURL == "" {
		if v, ok := lookupEnv(envBaseURL); ok && v != "" {
			cfg.baseURL = v
		} else if file.baseURL != "" {
			cfg.baseURL = file.baseURL
		} else {
			cfg.baseURL = defaultBaseURL
		}
	}

	// a set-but-empty env key reads as unset, so it falls through to the file
	// instead of blanking it; with neither set, no Authorization header is sent
	if v, ok := lookupEnv(envAPIKey); ok && v != "" {
		cfg.apiKey = v
	} else {
		cfg.apiKey = file.apiKey
	}

	// No flag and no environment variable here, so no precedence ladder: the
	// file's value is taken verbatim — no Clean, no tilde expansion — so a later
	// error names the directory the operator wrote.
	cfg.agentsDir = agentsDirPath(lookupEnv)
	if file.agentsDir != "" {
		cfg.agentsDir = file.agentsDir
	}
	// An agents_dir the Driver Config names is a directory the operator expects
	// to be in use, so its absence is fatal exactly as a $SPINDRIFT_CONFIG-named
	// file's is; the home-relative default is only where Agent Files would go, so
	// not finding it reads as "no Agent Files". A file that exists and does not
	// parse is fatal either way. loadAgentDir already names the offending
	// directory or file in every message it returns, so a wrap here would add
	// nothing but a repeated path.
	if cfg.agentsDir != "" {
		entries, err := loadAgentDir(cfg.agentsDir)
		if err != nil && (file.agentsDir != "" || !errors.Is(err, errNoAgentDir)) {
			return config{}, err
		}
		cfg.agents = entries
	}
	// The manifest path comes only from the Driver Config — no flag and no
	// environment variable — so the precedence ladder does not apply. A configured
	// manifest that cannot be read or does not parse is fatal: a Box must never
	// run unguarded because a file went missing. loadHookManifest already names
	// the manifest path in every message it returns, so nothing is wrapped here.
	if file.hookManifest != "" {
		manifest, err := loadHookManifest(file.hookManifest)
		if err != nil {
			return config{}, err
		}
		cfg.hooks = manifest.pre
		cfg.summary = manifest.summary
	}
	// Resolved against the loaded roster, never opened from a path in argv: the
	// configured directory is the only place an Agent File comes from.
	cfg.agent = *agentName
	if cfg.agent != "" {
		entry, ok := findAgent(cfg.agents, cfg.agent)
		if !ok {
			return config{}, fmt.Errorf("unknown agent %q", cfg.agent)
		}
		// The entry's model id is used verbatim — no effort suffix, no splitting on
		// "/" — because it is a router's key, not something the Driver composes.
		if cfg.model == "" {
			cfg.model = entry.Model
		}
		cfg.systemPrompt = entry.SystemPrompt
	}
	// Below the roster, because an entry can supply the model: a run named by
	// --agent has one whether or not --model was typed.
	if cfg.model == "" {
		return config{}, errors.New("--model is required")
	}
	cfg.tools = offeredTools(cfg.agents)
	if cfg.agent != "" {
		// A child is never offered the Subagent Tool: a subagent able to spawn is
		// unbounded recursion, and the roster's own spawns field is not enforcement
		// until #26 lands. bash is the whole offer because this tree has no file
		// Tools to add yet (#24).
		cfg.tools = []toolSpec{bashTool}
	}

	prompt, err := promptFrom(positional, stdin)
	if err != nil {
		return config{}, fmt.Errorf("read stdin: %w", err)
	}
	cfg.prompt = prompt
	if cfg.prompt == "" {
		return config{}, errors.New("prompt is required: pass it positionally or via stdin")
	}
	return cfg, nil
}

// splitAtEndOfFlags splits argv at the first standalone "--" element:
// tokens before it go through flag parsing, tokens after it are positional
// no matter how they look — this is the only way to pass a dash-leading
// prompt (e.g. "-weird" as the question itself).
func splitAtEndOfFlags(args []string) (pre, post []string) {
	for i, a := range args {
		if a == "--" {
			return args[:i], args[i+1:]
		}
	}
	return args, nil
}

// reorderFlagsFirst moves positional args to the end of argv, preserving
// relative order within each group. flag.Parse stops at the first positional,
// but the issue's argv shape puts the prompt before the flags. A flag of the
// form -name=value carries its own value; otherwise the next arg is assumed
// to be the flag's value (both of our flags take values).
// args never contains a standalone "--": splitAtEndOfFlags already split it
// off, so no end-of-flags handling is needed here.
func reorderFlagsFirst(args []string) []string {
	flags := make([]string, 0, len(args))
	positional := make([]string, 0, len(args))
	expectValue := false
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case expectValue:
			flags = append(flags, a)
			expectValue = false
		case len(a) > 1 && a[0] == '-':
			flags = append(flags, a)
			expectValue = !strings.Contains(a, "=")
		default:
			positional = append(positional, a)
		}
	}
	return append(flags, positional...)
}

// promptFrom returns the positional prompt verbatim; stdin is trimmed
// (pipes commonly carry trailing whitespace, argv does not).
func promptFrom(args []string, stdin io.Reader) (string, error) {
	if len(args) > 0 {
		return args[0], nil
	}
	if stdin == nil {
		return "", nil
	}
	b, err := io.ReadAll(stdin)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(b)), nil
}

// bashToolTimeout bounds one bash tool call; tests inject a short value.
var bashToolTimeout = 120 * time.Second

// bashToolMaxOutput caps how many bytes of command output a Tool Result
// keeps; tests inject a small value.
var bashToolMaxOutput = 10 * 1024

// chatRetryAttempts is the total number of attempts made for one chat request
// (so 3 means two retries); tests inject a smaller value.
var chatRetryAttempts = 3

// chatRetryBackoff is the wait before the second attempt, doubled on each
// further attempt; tests inject a shorter value.
var chatRetryBackoff = 100 * time.Millisecond

// chatAttemptTimeout is an idle bound on one attempt: the dial, the response
// headers, and the gap between received bytes of the streamed body — a stream
// that keeps arriving may run far past it. It is the fallback default
// for hand-built configs and the --timeout flag default; tests inject a
// shorter value.
var chatAttemptTimeout = 5 * time.Minute

// truncateToolOutput is the one output bound: it keeps the last
// bashToolMaxOutput bytes of a Tool Result, under which it is the identity.
// Both a command's own output and a summary hook's replacement come through
// here, so a summariser cannot hand the Model a longer result than the command
// was allowed — the cap is the floor the Driver holds even when a hook fails.
func truncateToolOutput(result string) string {
	if len(result) <= bashToolMaxOutput {
		return result
	}
	tail := result[len(result)-bashToolMaxOutput:]
	// never split a multi-byte UTF-8 rune: drop continuation bytes at the cut
	for len(tail) > 0 && tail[0]&0xC0 == 0x80 {
		tail = tail[1:]
	}
	return truncatedOutputMarker(len(tail)) + "\n" + string(tail)
}

// truncatedOutputMarker precedes the kept tail when output exceeded the cap,
// naming the real kept byte count (may fall a few bytes under the cap after
// the UTF-8 guard drops a partial rune).
func truncatedOutputMarker(kept int) string {
	return fmt.Sprintf("... output truncated, showing last %d bytes ...", kept)
}

// runBashTool executes one tool call's shell command and returns the Tool
// Result: combined stdout+stderr plus a trailing exit-code line. It runs
// under bash -c in the Driver's own cwd (no persistence between calls). A
// failing command — non-zero exit, or one that cannot even start — is an
// ordinary Tool Result for the model, never a Go error.
func runBashTool(command string) string {
	ctx, cancel := context.WithTimeout(context.Background(), bashToolTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "bash", "-c", command)
	// kill the whole process group on timeout: children of bash would
	// otherwise outlive the kill as orphans still holding the output pipes.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	// a child that escaped the group (e.g. setsid) survives the SIGKILL
	// still holding the output pipes; without a bound, Wait blocks forever
	// on their EOF and the fired timeout never produces a Tool Result.
	cmd.WaitDelay = time.Second
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	err := cmd.Start()
	if err != nil {
		return fmt.Sprintf("error: cannot start command: %v", err)
	}
	err = cmd.Wait()
	// trim exactly one trailing newline so the exit-code line sits directly
	// after the last output line without collapsing intentional blank lines.
	result := strings.TrimSuffix(output.String(), "\n")
	// keep only the tail: truncating here — before the exit-code line is appended
	// — guarantees that line always survives.
	result = truncateToolOutput(result)
	if ctx.Err() != nil {
		// a command finishing in the window between Wait returning and this
		// check can be misreported as a timeout; accepted race, since avoiding
		// it would need a second Wait.
		// must be distinct from a plain signal kill so the model can tell a
		// timeout apart from a command that died on a signal
		return result + fmt.Sprintf("\nexit code: timed out after %s", bashToolTimeout)
	}
	if err == nil {
		return result + "\nexit code: 0"
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		if exitErr.ExitCode() < 0 {
			// no exit code: the process was killed by a signal
			return result + fmt.Sprintf("\nexit code: signal %s", exitErr.Sys().(syscall.WaitStatus).Signal())
		}
		return result + fmt.Sprintf("\nexit code: %d", exitErr.ExitCode())
	}
	// command could not be started at all (e.g. bash missing)
	return fmt.Sprintf("error: cannot start command: %v", err)
}

type chatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// toolCall mirrors the OpenAI assistant tool_calls entry; Arguments stays a
// JSON-encoded string on the wire, not a decoded object.
type toolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function toolCallFunction `json:"function"`
}

type toolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type toolSpec struct {
	Type     string       `json:"type"`
	Function functionSpec `json:"function"`
}

type functionSpec struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// bashTool is the single tool advertised to the model: tool calling happens
// only via the tools parameter, never by parsing model text.
var bashTool = toolSpec{
	Type: "function",
	Function: functionSpec{
		Name:        "bash",
		Description: "Run a shell command in the repository and return its output.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"command": map[string]any{"type": "string", "description": "The shell command to run."},
			},
			"required": []string{"command"},
		},
	},
}

// subagentTool delegates one task to a subagent chosen by name; offeredTools
// puts it in the offer only when a roster exists. The description names no
// agents on purpose — issue #26 owns the agent-listing wording.
var subagentTool = toolSpec{
	Type: "function",
	Function: functionSpec{
		Name:        "subagent",
		Description: "Delegate one task to a named subagent and return that subagent's final answer.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"agent": map[string]any{"type": "string", "description": "The name of the subagent to delegate to."},
				"task":  map[string]any{"type": "string", "description": "The task for the subagent to carry out."},
			},
			"required": []string{"agent", "task"},
		},
	},
}

// streamOptions is the OpenAI knob that asks a streaming completion for a
// final usage-bearing chunk; without it a streaming reply reports no tokens.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Tools    []toolSpec    `json:"tools,omitempty"`
	// Unconditional: #20 rules out a non-streaming fallback, so no flag gates these.
	Stream        bool          `json:"stream"`
	StreamOptions streamOptions `json:"stream_options"`
}

// chatChunk is one SSE data payload of a streamed chat completion. The answer
// rides under delta rather than message, and usage may arrive on a final chunk
// that carries no choices at all.
type chatChunk struct {
	Model   string `json:"model"`
	Choices []struct {
		// Absent from a single-choice stream, where JSON leaves it 0 — the value
		// that readChatStream folds in.
		Index int `json:"index"`
		Delta struct {
			Content string `json:"content"`
			// ReasoningContent is the llama.cpp/vLLM-family spelling of a thinking
			// delta: reasoning servers send the answer in Content and the thought
			// that produced it here.
			ReasoningContent string           `json:"reasoning_content"`
			ToolCalls        []streamToolCall `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *chatUsage `json:"usage"`
}

// streamToolCall is one fragment of a streamed tool call. Unlike the whole
// toolCall it reassembles into, every field may be absent — only the first
// fragment of a call names it, the rest append arguments — and Index is a
// pointer because "no index field" is not the same statement as "index 0".
type streamToolCall struct {
	Index    *int                   `json:"index"`
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function streamToolCallFunction `json:"function"`
}

type streamToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// toolCallStream reassembles tool-call fragments into whole calls, held in the
// order each call first appeared so the loop runs them in the order the Model
// asked.
type toolCallStream struct {
	keys  []int
	calls []toolCall
}

// merge folds one delta's fragments into the calls gathered so far, keyed by the
// fragment's index. A fragment with no index is keyed by its position within
// that delta instead: servers that deliver a whole call in a single chunk often
// omit it, and position is the only slot available — keying those on 0 would
// merge two such calls into one. An index-less fragment naming an id the slot
// does not already hold is a second such call, not a continuation: merging them
// would splice two JSON argument objects into one unparseable call.
func (s *toolCallStream) merge(fragments []streamToolCall) {
	for pos, frag := range fragments {
		key := pos
		if frag.Index != nil {
			key = *frag.Index
		}
		at := -1
		for i, k := range s.keys {
			if k == key {
				at = i
				break
			}
		}
		// The replacement key is negative because a real index never is.
		if at >= 0 && frag.Index == nil && frag.ID != "" && s.calls[at].ID != "" && s.calls[at].ID != frag.ID {
			key, at = -(len(s.keys) + 1), -1
		}
		if at < 0 {
			s.keys = append(s.keys, key)
			s.calls = append(s.calls, toolCall{})
			at = len(s.calls) - 1
		}
		call := &s.calls[at]
		// A later fragment repeats a name or id without changing it; only an
		// argument fragment adds to the call.
		if frag.ID != "" {
			call.ID = frag.ID
		}
		if frag.Type != "" {
			call.Type = frag.Type
		}
		if frag.Function.Name != "" {
			call.Function.Name = frag.Function.Name
		}
		call.Function.Arguments += frag.Function.Arguments
	}
}

// chatUsage mirrors the OpenAI-shaped usage object. Fields are float64 so a
// server reporting a fractional count cannot make the whole response
// unparseable, and they match tokenTotals without conversion.
type chatUsage struct {
	PromptTokens     float64 `json:"prompt_tokens"`
	CompletionTokens float64 `json:"completion_tokens"`
	// PromptCacheHitTokens is the llama.cpp/vLLM spelling of cache read;
	// OpenAI spells the same number prompt_tokens_details.cached_tokens.
	PromptCacheHitTokens float64                 `json:"prompt_cache_hit_tokens"`
	PromptTokensDetails  *chatPromptTokenDetails `json:"prompt_tokens_details"`
}

type chatPromptTokenDetails struct {
	CachedTokens float64 `json:"cached_tokens"`
}

// chatReply is one Model call's answer: the assistant message to replay into
// the conversation, plus the things a usage event needs that the message
// itself does not carry — the model that answered and the token tally.
// CacheWrite5m, CacheWrite1h and CostUSD stay zero: this server family
// reports none of them.
//
// Reasoning is the thinking the server streamed alongside the answer. It lives
// off Message so it can never be replayed to the server as part of the
// assistant turn or surface in a text event.
type chatReply struct {
	Message   chatMessage
	Reasoning string
	Model     string
	Tokens    tokenTotals
}

// totals flattens the usage object to a call's tally, taking cache read from
// whichever spelling the server used (the non-zero one). A nil usage
// object — the common case for a plain llama.cpp endpoint — answers zeros with
// no error, because the call still owes exactly one usage event.
func (u *chatUsage) totals() tokenTotals {
	if u == nil {
		return tokenTotals{}
	}
	cacheRead := u.PromptCacheHitTokens
	if cacheRead == 0 && u.PromptTokensDetails != nil {
		cacheRead = u.PromptTokensDetails.CachedTokens
	}
	return tokenTotals{Input: u.PromptTokens, Output: u.CompletionTokens, CacheRead: cacheRead}
}

// wrapTimeout rewords a deadline failure for the operator: the runtime says
// only "context deadline exceeded (...)", which names neither the endpoint nor
// the bound that fired; stage names the leg that expired ("" for the round
// trip). Any other transport error (connection-refused most of all) already
// identifies itself, so it is returned untouched. %w preserves errors.Is on
// the original cause.
func wrapTimeout(target string, timeout time.Duration, stage string, err error) error {
	if isDeadline(err) {
		return fmt.Errorf("POST %s timed out after %s%s: %w", target, timeout, stage, err)
	}
	return err
}

// isDeadline recognises every flavour of "the clock ran out", because the bound
// is enforced in three places and no two report alike: a context deadline, the
// transport's own response-header timeout (a net.Error), and a read deadline the
// net poller fired on the body, which arrives as os.ErrDeadlineExceeded with no
// context involved. Anything else is a broken connection rather than an expiry,
// and keeps its own wording.
func isDeadline(err error) bool {
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, os.ErrDeadlineExceeded) {
		return true
	}
	var netErr net.Error
	return errors.As(err, &netErr) && netErr.Timeout()
}

// readFailure rewords a body-read failure the way the round trip's own deadline is
// reworded. Only the non-deadline cause gets the read-response prefix:
// wrapTimeout already names that leg, so wrapping first would say it twice.
func readFailure(target string, timeout time.Duration, err error) error {
	if !isDeadline(err) {
		err = fmt.Errorf("read response: %w", err)
	}
	return wrapTimeout(target, timeout, " while reading the response", err)
}

// errIdleExpiry stands in for whatever the transport reports once idleBody cuts
// a stalled stream off: it claims a net-poller deadline, which is exactly the
// kind of expiry a gap between bytes is.
var errIdleExpiry = fmt.Errorf("no bytes received: %w", os.ErrDeadlineExceeded)

// idleBody bounds the silence between bytes on a response body instead of its
// total length: Read re-arms the timer, so a stream that keeps talking may run
// far longer than timeout while one that falls silent is cut off after it.
// Expiry closes the source — the only way to interrupt the blocked read without
// owning the connection — and the error that read then returns is reported as
// errIdleExpiry so the caller can tell a stall from a torn connection.
type idleBody struct {
	src     io.ReadCloser
	timeout time.Duration

	timer   *time.Timer
	expired atomic.Bool
}

func newIdleBody(src io.ReadCloser, timeout time.Duration) *idleBody {
	return &idleBody{src: src, timeout: timeout}
}

func (b *idleBody) Read(p []byte) (int, error) {
	if b.expired.Load() {
		return 0, errIdleExpiry
	}
	if b.timer == nil {
		b.timer = time.AfterFunc(b.timeout, b.expire)
	} else {
		b.timer.Reset(b.timeout)
	}
	n, err := b.src.Read(p)
	// Bytes that arrived before the timer fired, and an EOF that beat it, are
	// part of the answer; only a failed read can be the stall.
	if err != nil && !errors.Is(err, io.EOF) && b.expired.Load() {
		return n, errIdleExpiry
	}
	return n, err
}

func (b *idleBody) expire() {
	b.expired.Store(true)
	b.src.Close()
}

func (b *idleBody) Close() error {
	if b.timer != nil {
		b.timer.Stop()
	}
	return b.src.Close()
}

// sseDoneMarker is the payload that closes a stream: it names no chunk, so
// parsing it as one would be the same mistake as parsing a heartbeat.
const sseDoneMarker = "[DONE]"

// sseMaxLineBytes caps one scanned line. bufio's 64KB default is shorter than
// the frame a server sends when it answers in a single chunk — a long tool
// call's arguments included — so a healthy stream would fail as too long.
const sseMaxLineBytes = 16 << 20

// streamReadError marks a connection failure while the response body was in
// flight, which is flakiness the retry loop owns. A stream that read cleanly but
// carried nothing parseable is an answer, so it is deliberately not one.
type streamReadError struct{ err error }

func (e streamReadError) Error() string { return e.err.Error() }
func (e streamReadError) Unwrap() error { return e.err }

// readChatStream folds a text/event-stream chat completion into the single
// assistant message the caller replays, using the standard library alone
// (ADR-0001). In SSE a blank line separates events, a leading ':' is a keep-alive
// comment, and every other line is a "field: value" pair — so only data: carries
// a chunk and event:/id:/retry: lines are skipped rather than parsed. status is
// the response's own status line, which the unparseable-payload wording names.
// delta.reasoning_content accumulates on its own into chatReply.Reasoning, so a
// reasoning model's thinking never enters the message the loop replays.
func readChatStream(body io.Reader, status string) (chatReply, error) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), sseMaxLineBytes)
	var (
		reply     chatReply
		content   strings.Builder
		reasoning strings.Builder
		chunks    int
		calls     toolCallStream
	)
	var parseErr error
	for scanner.Scan() {
		field, payload, ok := strings.Cut(scanner.Text(), ":")
		if !ok || field != "data" {
			continue
		}
		// the spec allows exactly one optional space after the colon
		payload = strings.TrimPrefix(payload, " ")
		if payload == sseDoneMarker {
			break
		}
		if payload == "" {
			// an empty data field is an event with nothing in it, not a chunk
			continue
		}
		var chunk chatChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			if parseErr == nil {
				parseErr = fmt.Errorf("unparseable response (status %s): %w", status, err)
			}
			continue
		}
		chunks++
		if chunk.Model != "" && reply.Model == "" {
			reply.Model = chunk.Model
		}
		if chunk.Usage != nil {
			reply.Tokens = chunk.Usage.totals()
		}
		for _, choice := range chunk.Choices {
			// Only choice 0 is an answer this Driver could have asked for: it never
			// sends n>1 and the loop replays exactly one assistant message, so a
			// second choice is an alternative answer, not a continuation to append.
			if choice.Index != 0 {
				continue
			}
			content.WriteString(choice.Delta.Content)
			reasoning.WriteString(choice.Delta.ReasoningContent)
			// a tool call delivered whole in one delta is one fragment of itself:
			// merging it needs no special case
			calls.merge(choice.Delta.ToolCalls)
		}
	}
	// The read leg outranks everything else: a stream that broke mid-frame leaves
	// a partial line the scanner hands over as a final token, and reporting that
	// as unparseable would bury the transport failure the caller has to retry.
	if err := scanner.Err(); err != nil {
		return chatReply{}, streamReadError{err: err}
	}
	if parseErr != nil {
		return chatReply{}, parseErr
	}
	if chunks == 0 {
		// the streaming analogue of the old empty-choices 200: a permanent answer,
		// not flakiness, so it must not be retried into silence
		return chatReply{}, errors.New("server returned a stream with no chunks")
	}
	// deltas carry no role, but the replayed message has to read as an assistant
	// turn
	reply.Message.Role = "assistant"
	reply.Message.Content = content.String()
	reply.Reasoning = reasoning.String()
	reply.Message.ToolCalls = calls.calls
	return reply, nil
}

// sendChat performs one streaming chat completion round trip against an
// OpenAI-compatible server and returns the assistant message the chunks assemble
// into, alongside the response model and token tally. The response's own model
// wins; a server that omits it falls back to the configured --model, because a
// usage event with no model id would drop the call from the per-model
// breakdown.
func sendChat(cfg config, messages []chatMessage, stderr io.Writer) (chatReply, error) {
	// base URL may or may not end in /; join safely
	target := strings.TrimSuffix(cfg.baseURL, "/") + "/chat/completions"
	// A config assembled anywhere but parseArgs has no offer of its own, and the
	// bash-only offer is what predates the tool list, so it stays the default.
	tools := cfg.tools
	if len(tools) == 0 {
		tools = []toolSpec{bashTool}
	}
	reqBody, err := json.Marshal(chatRequest{
		Model:         cfg.model,
		Messages:      messages,
		Tools:         tools,
		Stream:        true,
		StreamOptions: streamOptions{IncludeUsage: true},
	})
	if err != nil {
		return chatReply{}, err
	}
	// clamped so an injected knob of 0 (or less) still makes one attempt
	// instead of reporting an empty answer with a nil error.
	attempts := chatRetryAttempts
	if attempts < 1 {
		attempts = 1
	}
	// A hand-built config carries no --timeout, and a zero timeout means no bound
	// at all rather than no wait, so fall back to the package default.
	timeout := cfg.timeout
	if timeout <= 0 {
		timeout = chatAttemptTimeout
	}
	// The bound is an idle one, so it cannot be http.Client.Timeout: that expires
	// the whole round trip and would cut off a stream still delivering bytes. Each
	// leg gets its own instead — the transport bounds the dial and the wait for
	// response headers, idleBody bounds each gap between received bytes. The two
	// legs run in sequence, so a first byte can land up to two bounds after the
	// request: the bound is per leg, not one deadline over the round trip plus
	// the first read. Cloned so only these two settings differ from the defaults,
	// since mutating DefaultTransport would change every other client in the
	// process.
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.DialContext = (&net.Dialer{Timeout: timeout}).DialContext
	transport.ResponseHeaderTimeout = timeout
	// nothing reuses this per-call transport once sendChat returns, and its idle
	// connections park a read loop each until IdleConnTimeout
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport}
	var lastErr error
	delay := chatRetryBackoff
	// Only a 5xx or a transport failure is retried, and only inside this one
	// request: a 4xx, an unparseable chunk and an empty stream are answers, not
	// flakiness, so they return immediately.
	for attempt := 1; attempt <= attempts; attempt++ {
		if attempt > 1 {
			fmt.Fprintf(stderr, "spindrift-driver: retry %d/%d in %s (%s)\n", attempt, attempts, delay, lastErr)
			time.Sleep(delay)
			delay *= 2
		}
		// reqBody is marshalled once above, but the bytes.Reader over it is
		// one-shot, so every attempt needs a freshly built request.
		req, err := http.NewRequest(http.MethodPost, target, bytes.NewReader(reqBody))
		if err != nil {
			return chatReply{}, err
		}
		req.Header.Set("Content-Type", "application/json")
		if cfg.apiKey != "" {
			// absent key means no header, not a bare (invalid) "Bearer "
			req.Header.Set("Authorization", "Bearer "+cfg.apiKey)
		}
		fmt.Fprintf(stderr, "spindrift-driver: POST %s\n", target)

		resp, err := client.Do(req)
		if err != nil {
			lastErr = wrapTimeout(target, timeout, "", err)
			continue
		}
		// Only a 200 is the event stream. An error status answers with a plain
		// body, which the retry note and the 4xx return still need whole.
		if resp.StatusCode != http.StatusOK {
			// an error answer is not an event stream, but it is still a body the
			// server may stop writing halfway through, so the same idle bound applies
			src := newIdleBody(resp.Body, timeout)
			body, err := io.ReadAll(src)
			src.Close()
			if err != nil {
				lastErr = readFailure(target, timeout, err)
				continue
			}
			if resp.StatusCode >= http.StatusInternalServerError {
				lastErr = fmt.Errorf("server returned %s: %s", resp.Status, boundBody(body))
				continue
			}
			// Verbatim body: no TrimSpace, because a rejection that pads its message
			// with newlines means the same text to the host as to the operator, and
			// JSON escaping already keeps the event line whole.
			rej := &rejection{status: resp.Status, body: string(body)}
			// Only a 429 names a reset, and only here — a 429 stays a 4xx that
			// returns without a retry, exactly as it always did.
			if resp.StatusCode == http.StatusTooManyRequests {
				if ms, ok := retryAfterMS(resp.Header, nowMS()); ok {
					rej.resetAt = ms
				}
			}
			return chatReply{}, rej
		}

		src := newIdleBody(resp.Body, timeout)
		reply, err := readChatStream(src, resp.Status)
		src.Close()
		if err != nil {
			var read streamReadError
			if errors.As(err, &read) {
				// a stream that dies mid-body is the same kind of trouble as a round
				// trip that never completes, so it reads the same to the operator and
				// is retried the same way
				lastErr = readFailure(target, timeout, read.err)
				continue
			}
			return chatReply{}, err
		}
		if reply.Model == "" {
			reply.Model = cfg.model
		}
		return reply, nil
	}
	return chatReply{}, lastErr
}

// boundBody flattens and caps a server-supplied body: the retry note is one
// line per attempt, so raw newlines would let the server forge extra driver
// log lines and an unbounded body would flood stderr. Rune-wise cap so a cut
// cannot split a multibyte rune.
func boundBody(body []byte) string {
	const maxBodyRunes = 200
	flat := strings.Join(strings.Fields(string(body)), " ")
	if r := []rune(flat); len(r) > maxBodyRunes {
		return string(r[:maxBodyRunes]) + "..."
	}
	return flat
}
