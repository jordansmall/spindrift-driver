package main

import (
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
	"strings"
	"syscall"
	"time"
)

const defaultBaseURL = "http://localhost:8080/v1"
const envBaseURL = "SPINDRIFT_BASE_URL"

// systemPrompt is deliberately minimal and hardcoded: slice 2 has no tool
// loop, so the Model just gets repo context via the user prompt.
const systemPrompt = "You are a coding agent working in this repository. Answer with the final response only; it is printed to stdout."

type config struct {
	model    string
	baseURL  string
	prompt   string
	maxTurns int
}

const usage = `usage: spindrift-driver "prompt" --model <model> [--base-url <url>] [--max-turns <n>]

The prompt may be given positionally or on stdin. --model is required.
--base-url defaults to $SPINDRIFT_BASE_URL or ` + defaultBaseURL + `.
--max-turns caps the number of agent turns (default 50).
`

// run is the process-boundary seam: tests drive it in-process, main wraps it.
func run(args []string, stdin io.Reader, stdout, stderr io.Writer, lookupEnv func(string) (string, bool)) int {
	cfg, err := parseArgs(args, stdin, lookupEnv)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) { // -h/--help: usage is the output, not an error
			fmt.Fprint(stdout, usage)
			return 0
		}
		fmt.Fprintf(stderr, "spindrift-driver: %v\n%s", err, usage)
		return 2
	}
	// Autonomous tool loop: each turn posts the whole conversation with the
	// tools parameter; the server decides tool calls (never parsed from text).
	messages := []chatMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: cfg.prompt},
	}
	for turn := 0; turn < cfg.maxTurns; turn++ {
		assistant, err := sendChat(cfg, messages, stderr)
		if err != nil {
			fmt.Fprintf(stderr, "spindrift-driver: %v\n", err)
			return 1
		}
		if len(assistant.ToolCalls) == 0 {
			// No tool calls: the content is the final answer.
			final := assistant.Content
			if !strings.HasSuffix(final, "\n") {
				final += "\n"
			}
			fmt.Fprint(stdout, final)
			return 0
		}
		// OpenAI convention: replay the assistant message verbatim (with its
		// tool_calls), then one tool message per call.
		messages = append(messages, assistant)
		for _, call := range assistant.ToolCalls {
			result := executeToolCall(call)
			messages = append(messages, chatMessage{Role: "tool", Content: result, ToolCallID: call.ID})
		}
	}
	fmt.Fprintf(stderr, "spindrift-driver: reached the turn limit of %d without a final answer\n", cfg.maxTurns)
	return 1
}

// executeToolCall runs one assistant tool call and returns its Tool Result
// content. Unknown tools and unparseable arguments are ordinary results too,
// so the loop can hand them back to the model instead of crashing.
func executeToolCall(call toolCall) string {
	if call.Function.Name != "bash" {
		return fmt.Sprintf("error: unknown tool %q", call.Function.Name)
	}
	var args struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return fmt.Sprintf("error: cannot parse tool arguments: %v", err)
	}
	return runBashTool(args.Command)
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

	cfg := config{
		model:    *model,
		baseURL:  *baseURL,
		maxTurns: *maxTurns,
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

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Tools    []toolSpec    `json:"tools,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
}

// sendChat performs one chat completion round trip against an
// OpenAI-compatible server and returns the full assistant message
// (content and tool calls).
func sendChat(cfg config, messages []chatMessage, stderr io.Writer) (chatMessage, error) {
	// base URL may or may not end in /; join safely
	target := strings.TrimSuffix(cfg.baseURL, "/") + "/chat/completions"
	reqBody, err := json.Marshal(chatRequest{
		Model:    cfg.model,
		Messages: messages,
		Tools:    []toolSpec{bashTool},
	})
	if err != nil {
		return chatMessage{}, err
	}
	req, err := http.NewRequest(http.MethodPost, target, bytes.NewReader(reqBody))
	if err != nil {
		return chatMessage{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	fmt.Fprintf(stderr, "spindrift-driver: POST %s\n", target)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return chatMessage{}, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return chatMessage{}, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return chatMessage{}, fmt.Errorf("server returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	var parsed chatResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return chatMessage{}, fmt.Errorf("unparseable response (status %s): %w", resp.Status, err)
	}
	if len(parsed.Choices) == 0 {
		return chatMessage{}, errors.New("server returned no choices")
	}
	return parsed.Choices[0].Message, nil
}
