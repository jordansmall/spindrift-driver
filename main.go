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
	"net/http"
	"os"
	"os/exec"
	"sort"
	"strings"
	"syscall"
	"time"
)

const defaultBaseURL = "http://localhost:8080/v1"
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
	// apiKey is env-only by design: a --api-key flag would put the secret in
	// argv, where process listings and shell history expose it.
	apiKey string
}

// usage is a var, not a const: the --timeout default is interpolated from
// chatAttemptTimeout so the text cannot drift from the value in effect.
var usage = `usage: spindrift-driver "prompt" --model <model> [--base-url <url>] [--max-turns <n>] [--timeout <dur>]

The prompt may be given positionally or on stdin. --model is required.
--base-url defaults to $SPINDRIFT_BASE_URL or ` + defaultBaseURL + `.
--max-turns caps the number of agent turns (default 50).
--timeout bounds one chat attempt (dial, response headers, body read), default ` + chatAttemptTimeout.String() + `;
raise it when the server generates slowly — a local model can take minutes per turn.
The API key is read from $` + envAPIKey + ` only, never from argv.
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
	messages := []chatMessage{
		{Role: "system", Content: systemPrompt},
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
	seenModels := map[string]struct{}{}
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
				Tokens:        totals,
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
			plan := planToolCall(tc)
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
			output := plan.run()
			_ = events.emit(event{
				Kind:       "tool_result",
				MessageID:  messageID,
				ToolID:     toolID,
				ToolName:   tc.Function.Name,
				ToolTarget: plan.target,
				ToolOutput: output,
				IsError:    toolResultIsError(output),
			})
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
	_ = events.emit(event{Kind: "error", Text: err.Error()})
	return code
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
	// run produces the exact Tool Result string that goes back into the
	// conversation.
	run func() string
}

// planToolCall parses one tool call without executing it. An unknown tool and
// unparseable arguments are ordinary results rather than crashes, so the plan
// answers with the very error text the Model has always received.
func planToolCall(call toolCall) toolPlan {
	input := decodeToolArguments(call.Function.Arguments)
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
	return toolPlan{input: input, target: args.Command, run: func() string { return runBashTool(args.Command) }}
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
// suffix there and every tool result flips to is_error true silently — the two
// sites have to change together.
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
	maxTurns := fs.Int("max-turns", 50, "maximum agent turns")
	timeout := fs.Duration("timeout", chatAttemptTimeout, "per-attempt timeout for one chat request (dial, headers, body read); accepts Go durations such as 90s or 2m")
	pre, post := splitAtEndOfFlags(args)
	if err := fs.Parse(reorderFlagsFirst(pre)); err != nil {
		return config{}, err
	}
	positional := append(fs.Args(), post...)
	if len(positional) > 1 {
		return config{}, fmt.Errorf("unexpected argument %q: only one prompt is accepted", positional[1])
	}

	if *maxTurns < 1 {
		return config{}, fmt.Errorf("--max-turns must be at least 1, got %d", *maxTurns)
	}
	if *timeout <= 0 {
		return config{}, fmt.Errorf("--timeout must be positive, got %v", *timeout)
	}

	cfg := config{
		model:    *model,
		baseURL:  *baseURL,
		maxTurns: *maxTurns,
		timeout:  *timeout,
	}
	if cfg.model == "" {
		return config{}, errors.New("--model is required")
	}
	if cfg.baseURL == "" {
		if v, ok := lookupEnv(envBaseURL); ok {
			cfg.baseURL = v
		} else {
			cfg.baseURL = defaultBaseURL
		}
	}

	// a set-but-empty key reads as unset, so no Authorization header is sent
	if v, ok := lookupEnv(envAPIKey); ok {
		cfg.apiKey = v
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

// chatAttemptTimeout bounds a single attempt: dial, response headers and
// response body read. It is the fallback default for hand-built configs and
// the --timeout flag default; tests inject a shorter value.
var chatAttemptTimeout = 5 * time.Minute

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
	// keep only the tail: a runaway generator must not flood the model's
	// context, and truncating here — before the exit-code line is appended —
	// guarantees that line always survives.
	if len(result) > bashToolMaxOutput {
		tail := result[len(result)-bashToolMaxOutput:]
		// never split a multi-byte UTF-8 rune: drop continuation bytes at the cut
		for len(tail) > 0 && tail[0]&0xC0 == 0x80 {
			tail = tail[1:]
		}
		result = truncatedOutputMarker(len(tail)) + "\n" + string(tail)
	}
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
			Content   string     `json:"content"`
			ToolCalls []toolCall `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *chatUsage `json:"usage"`
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
// the conversation, plus the two things a usage event needs that the message
// itself does not carry — the model that answered and the token tally.
// CacheWrite5m, CacheWrite1h and CostUSD stay zero: this server family
// reports none of them.
type chatReply struct {
	Message chatMessage
	Model   string
	Tokens  tokenTotals
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
	if errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("POST %s timed out after %s%s: %w", target, timeout, stage, err)
	}
	return err
}

// readFailure rewords a body-read failure the way the round trip's own deadline is
// reworded. Only the non-deadline cause gets the read-response prefix:
// wrapTimeout already names that leg, so wrapping first would say it twice.
func readFailure(target string, timeout time.Duration, err error) error {
	if !errors.Is(err, context.DeadlineExceeded) {
		err = fmt.Errorf("read response: %w", err)
	}
	return wrapTimeout(target, timeout, " while reading the response", err)
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
func readChatStream(body io.Reader, status string) (chatReply, error) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), sseMaxLineBytes)
	var (
		reply   chatReply
		content strings.Builder
		chunks  int
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
			// a tool call delivered whole in one delta needs no reassembly
			reply.Message.ToolCalls = append(reply.Message.ToolCalls, choice.Delta.ToolCalls...)
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
	reqBody, err := json.Marshal(chatRequest{
		Model:         cfg.model,
		Messages:      messages,
		Tools:         []toolSpec{bashTool},
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
	// A hand-built config carries no --timeout, and Client.Timeout 0 means no
	// bound at all rather than no wait, so fall back to the package default.
	timeout := cfg.timeout
	if timeout <= 0 {
		timeout = chatAttemptTimeout
	}
	// Bounded per attempt: http.DefaultClient is process-global and has no
	// timeout, so a server that accepts the connection and never answers would
	// hang the run forever. Client.Timeout covers dial, headers and body read.
	client := &http.Client{Timeout: timeout}
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
			body, err := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil {
				lastErr = readFailure(target, timeout, err)
				continue
			}
			if resp.StatusCode >= http.StatusInternalServerError {
				lastErr = fmt.Errorf("server returned %s: %s", resp.Status, boundBody(body))
				continue
			}
			return chatReply{}, fmt.Errorf("server returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
		}
		reply, err := readChatStream(resp.Body, resp.Status)
		resp.Body.Close()
		if err != nil {
			var read streamReadError
			if errors.As(err, &read) {
				// a stream that dies mid-body trips the same Client.Timeout as a
				// round trip that never completes, so it reads the same to the
				// operator and is retried the same way
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
