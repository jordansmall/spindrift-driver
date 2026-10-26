package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

const defaultBaseURL = "http://localhost:8080/v1"
const envBaseURL = "SPINDRIFT_BASE_URL"

// systemPrompt is deliberately minimal and hardcoded: slice 2 has no tool
// loop, so the Model just gets repo context via the user prompt.
const systemPrompt = "You are a coding agent working in this repository. Answer with the final response only; it is printed to stdout."

type config struct {
	model   string
	baseURL string
	prompt  string
}

const usage = `usage: spindrift-driver "prompt" --model <model> [--base-url <url>]

The prompt may be given positionally or on stdin. --model is required.
--base-url defaults to $SPINDRIFT_BASE_URL or ` + defaultBaseURL + `.
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
	resp, err := sendChat(cfg, stderr)
	if err != nil {
		fmt.Fprintf(stderr, "spindrift-driver: %v\n", err)
		return 1
	}
	if !strings.HasSuffix(resp, "\n") {
		resp += "\n"
	}
	fmt.Fprint(stdout, resp)
	return 0
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr, os.LookupEnv))
}

func parseArgs(args []string, stdin io.Reader, lookupEnv func(string) (string, bool)) (config, error) {
	fs := flag.NewFlagSet("spindrift-driver", flag.ContinueOnError)
	fs.SetOutput(io.Discard) // usage errors are reported by run, keep parse silent
	baseURL := fs.String("base-url", "", "OpenAI-compatible API base URL (default: $SPINDRIFT_BASE_URL or "+defaultBaseURL+")")
	model := fs.String("model", "", "model name (required)")
	pre, post := splitAtEndOfFlags(args)
	if err := fs.Parse(reorderFlagsFirst(pre)); err != nil {
		return config{}, err
	}
	positional := append(fs.Args(), post...)
	if len(positional) > 1 {
		return config{}, fmt.Errorf("unexpected argument %q: only one prompt is accepted", positional[1])
	}

	cfg := config{
		model:   *model,
		baseURL: *baseURL,
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

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
}

// sendChat performs the one-shot chat completion round trip against an
// OpenAI-compatible server and returns the assistant's reply text.
func sendChat(cfg config, stderr io.Writer) (string, error) {
	// base URL may or may not end in /; join safely
	target := strings.TrimSuffix(cfg.baseURL, "/") + "/chat/completions"
	reqBody, err := json.Marshal(chatRequest{
		Model: cfg.model,
		Messages: []chatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: cfg.prompt},
		},
	})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequest(http.MethodPost, target, bytes.NewReader(reqBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	fmt.Fprintf(stderr, "spindrift-driver: POST %s\n", target)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("server returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	var parsed chatResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", fmt.Errorf("unparseable response (status %s): %w", resp.Status, err)
	}
	if len(parsed.Choices) == 0 {
		return "", errors.New("server returned no choices")
	}
	return parsed.Choices[0].Message.Content, nil
}
