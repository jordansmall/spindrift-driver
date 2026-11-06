# spindrift-driver

A **Driver** for the spindrift coordinator: one prompt in, one final answer out — and
the answer arrives as a `result` event on stdout's event stream, not as loose text.
It talks to an OpenAI-compatible **Model** server, offers that Model a single Bash
**Tool**, runs the Tool Calls the Model asks for, and loops until the Model answers.

```
usage: spindrift-driver "prompt" --model <model> [--base-url <url>] [--max-turns <n>] [--timeout <dur>]
```

The prompt is positional or on stdin; `--model` is required; `--max-turns` caps the
agent loop (default 50). `--timeout` is an idle bound on each chat attempt — the dial,
the response headers and the gap between received bytes of the streamed response —
and defaults to 5m0s, so a stream that keeps arriving may run far past it. Raise it
(`--timeout 15m`) only when the server thinks for a long time before its first token, or
when a proxy buffers the response.

## Output: stdout is a JSONL event stream

stdout is a JSONL event stream — one JSON object per line, and nothing else, on every
exit path, a failed run and `-h` included. There is no output-mode flag: these events
are the whole stdout contract ([ADR-0004](docs/adr/0004-events-are-the-only-stdout-contract.md)).
`kind` names the event, `timestamp` is epoch milliseconds, field names are snake_case,
and an event omits the fields its kind does not fill.

The kinds this Driver emits, in the order they appear: per Model call, `start`, then
`reasoning` when that call streamed thinking, then `text` when that call produced
assistant prose, then `usage`, then for each Tool Call a `tool_call` followed by its
`tool_result`; after the last call, `result` on success or `error` before a non-zero
exit.

- `start` — one Model call (turn) beginning. Every event of a call carries that call's
  `message_id`.
- `reasoning` — the Model's streamed thinking, in `text`, emitted once per completed
  message ahead of the `text` event it produced. It is never the answer: only `text` and
  the `result` event carry that.
- `text` — assistant prose, verbatim and untrimmed.
- `usage` — exactly one usage event per Model call, holding that call's `tokens` and
  never a running total. A server that reports no usage still gets one, with zeroes.
- `tool_call` — the Model invoking a Tool: `tool_name`, `tool_input` (the arguments
  decoded as JSON) and `tool_target` (the command, for the Bash Tool).
- `tool_result` — that call's output on its way back to the Model, `is_error` set when
  the command failed. `tool_call` and `tool_result` pair by `tool_id`.
- `result` — the final answer, in `text`, plus `num_turns`, `duration_ms`,
  `duration_api_ms`, the session token totals in `tokens` and the `models` that
  answered. The answer is this event's `text`; it is also emitted as a `text` event,
  because it is assistant prose.
- `error` — every non-zero exit is preceded by one `error` event carrying the raw error
  text, verbatim and unbounded. Exit codes stay authoritative beside it: 0 for a
  `result` (and for `-h`), 1 for a failed run, 2 for a usage error.

Pull the answer out with `jq`, and the failure text the same way:

```
spindrift-driver "…" --model <model> | jq -r 'select(.kind=="result") | .text'
spindrift-driver "…" --model <model> | jq -r 'select(.kind=="error") | .text'
```

Progress stays on stderr and never moves to stdout: each `POST`, each retry, the
one-line driver error. `-h`/`--help` writes its usage to stderr for the same reason, so
stdout parses as events even there.

`testdata/pass.jsonl` is the committed golden Pass log: one scripted run's whole event
stream, byte for byte. `go test -run TestGoldenPassLog -update` regenerates it, and
without `-update` that test fails on any drift — the artifact a host-side decoder is
written against never moves silently.

## Build

```
nix build
./result/bin/spindrift-driver --help
```

`nix build` builds `packages.default` and leaves it at `./result`, so the binary is
`./result/bin/spindrift-driver`. The flake targets `x86_64-linux` and
`aarch64-linux`.

`flake.lock` is committed and pins the nixpkgs input to one reviewed revision, so
every `nix build` builds the same tree instead of following a moving branch. To
refresh the pin deliberately, run `nix flake update` and commit the lock it writes.
A refreshed nixpkgs must provide `go` at or above the `go 1.26.8` line in `go.mod`:
`buildGoModule` pins `GOTOOLCHAIN=local` and cannot download a toolchain, so an
older pin fails the build outright. CI asserts the floor.

Without nix, any Go toolchain at or above the `go` line in `go.mod` builds the same
binary — the driver uses the standard library alone (see [docs/adr/](docs/adr/)):

```
go build ./...
```

## Dev shell

```
nix develop
```

The shell provides `go`, `gopls` and `goimports` from the same pinned nixpkgs that
builds the package (`goimports` ships in nixpkgs' `gotools`), so the editor, the
formatter and the build never disagree about the toolchain.

## Checks

```
nix flake check
```

runs the test suite (`checks.go-test` runs `go test ./...` over the packaged
source). On a host without nix:

```
CGO_ENABLED=0 go test ./...
```

The suite drives real processes, so it needs `bash`, `sleep` and `seq` on `PATH`;
nothing in it opens a socket beyond the loopback `httptest` servers it starts itself.

## Driver Config

A run can take its settings from a JSON file rather than from anything on the command
line ([ADR-0005](docs/adr/0005-driver-config.md)). spindrift writes the file through
`envFiles`, its only run-time channel for putting content where a Box will read it, which
is why the file exists at all.

```
{"base_url": "http://model.example:8080/v1", "api_key": "sk-...", "timeout": "90s"}
```

The file holds exactly three fields: `base_url`, an optional `api_key`, and `timeout` —
the idle bound `--timeout` sets, written as a Go duration string such as `"90s"`. Nothing
else belongs in it: the Model name and `--max-turns` stay flags, so a file can move a
run's traffic but never change what a run does. It lives at
`$HOME/.config/spindrift-driver/config.json`, or at the path `$SPINDRIFT_CONFIG` names,
which is used verbatim.

Precedence is one ladder for every field: the flag, then the environment variable
(`SPINDRIFT_BASE_URL` for the URL, `SPINDRIFT_API_KEY` for the key), then the Driver
Config, then the built-in default — a knob an operator turned by hand always outranks
content baked into a Box. `timeout` has no environment variable, so for it the ladder is
flag, Driver Config, default.

An absent file at the default path is not an error; that is the ordinary state of a
machine with no config, and the run goes on with whatever the flags and defaults give it.
A file you actually named is held to the letter: a `SPINDRIFT_CONFIG` path that does not
exist, malformed JSON, an unknown field, and a field of the wrong type are each a startup
error naming the file and the problem at exit 2, beside the other startup errors — and
`-h` still prints its usage with a broken file in place. A typo such as `base_ur1` never
falls back to the default endpoint in silence.

No flag carries the key: the API key comes from `$SPINDRIFT_API_KEY` or from this file and
never travels on the command line, so it never appears in a process listing where another
user of the host could read it. Nothing is read from the working directory or from the
Target repo — no `spindrift.json`, no `config.json` in the checkout, no dotenv file —
because the repository under repair does not get to configure the run that repairs it.

## Manual live smoke test

Manual by design — it needs a Model server that is actually running, so it is run by
hand and is not part of `nix flake check` or CI.

1. Have an OpenAI-compatible chat completions server reachable from this machine.
2. Point the Driver at it, with `--base-url <url>` or `$SPINDRIFT_BASE_URL` (default
   `http://localhost:8080/v1`). The `/chat/completions` path is appended for you.
3. If the server expects a bearer token, export it: `export SPINDRIFT_API_KEY=…`.
   The key comes from `$SPINDRIFT_API_KEY` or from the Driver Config — there is no flag
   for it, so it never lands in the process listing — and with both unset no
   `Authorization` header is sent at all.
4. Run it:

   ```
   ./result/bin/spindrift-driver "List the Go files here, then say what main.go does." --model <model>
   ```

Expected: exit status 0 and a Pass log on stdout, with the Model's final answer in the
`result` event's `text` — `jq -r 'select(.kind=="result") | .text'` is how you read it.
Progress — each `POST`, each retry — goes to stderr, so stdout stays parseable end to
end. A failure prints one line to stderr, emits an `error` event on stdout, and exits
non-zero: unreachable server, rejected request, a stream that goes silent for longer than
`--timeout` (default 5m0s — a stalled stream, not a long generation: raise it when the
server thinks for a long time before its first token), or the `--max-turns` cap reached
without an answer.

## Not packaged

llama.cpp is deliberately not packaged here. The Model server is always remote: the
Driver's job ends at the OpenAI-compatible endpoint, and packaging an inference
engine would put a large CUDA/GPU dependency tree between this repo and a build.

Decisions behind these boundaries are in [docs/adr/](docs/adr/).
