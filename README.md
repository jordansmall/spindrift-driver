# spindrift-driver

A **Driver** for the spindrift coordinator: one prompt in, one final answer out — and
the answer arrives as a `result` event on stdout's event stream, not as loose text.
It talks to an OpenAI-compatible **Model** server, offers that Model a Bash **Tool**
— and a Subagent **Tool** whenever Agent Files are present — runs the Tool Calls the
Model asks for, and loops until the Model answers.

```
usage: spindrift-driver "prompt" --model <model> [--base-url <url>] [--max-turns <n>] [--timeout <dur>] [--agent <name>]
```

The prompt is positional or on stdin; `--model` is required; `--max-turns` caps the
agent loop (default 300). `--timeout` is an idle bound on each chat attempt — the dial,
the response headers and the gap between received bytes of the streamed response —
and defaults to 5m0s, so a stream that keeps arriving may run far past it. Raise it
(`--timeout 15m`) only when the server thinks for a long time before its first token, or
when a proxy buffers the response.

Two limits end a run before the Model answers. `--max-turns` caps the loop, as above. The
**Repeated-Call Breaker** watches the calls inside it, keyed on the tool name and the
exact arguments string the Model sent: the third identical call in a row is not run, and
the Model gets a Tool Result back asking it to take a different action instead. Any
different call resets the count; three refusals in a row end the run with an `error` event
and exit 1, which says what happened rather than spending the remaining turns repeating
one call. The count is per Driver process, so a child a Subagent Tool spawns starts at
zero. A call a Guard Hook blocks is not a refusal by the breaker: the call ran and the
Model got an answer back, so it resets the streak like any other call.

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
  the command failed. `tool_call` and `tool_result` pair by `tool_id`. A call the
  Repeated-Call Breaker refused never ran, so its `tool_result` is the Driver's own text
  asking the Model to take a different action, and it pairs with its `tool_call` like any
  result does.
- `result` — the final answer, in `text`, plus `num_turns`, `duration_ms`,
  `duration_api_ms`, the session token totals in `tokens` and the `models` that
  answered. The answer is this event's `text`; it is also emitted as a `text` event,
  because it is assistant prose.
- `error` — every non-zero exit is preceded by one `error` event carrying the raw error
  text, verbatim and unbounded — for a rejected request, that text is the server's
  response body exactly as it arrived. `reset_at`, when present, names in epoch
  milliseconds the instant a rate-limiting server said it would take the request again;
  a rejection that names no reset leaves the key off. Exit codes stay authoritative
  beside it: 0 for a `result` (and for `-h`), 1 for a failed run, 2 for a usage error.

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
{"base_url": "http://model.example:8080/v1", "api_key": "sk-...", "timeout": "90s", "agents_dir": "/srv/agents", "hook_manifest": "/srv/hook-manifest.json"}
```

The file holds five fields: `base_url`, an optional `api_key`, `timeout` — the idle bound
`--timeout` sets, written as a Go duration string such as `"90s"` — `agents_dir`, the
directory Agent Files are read from instead of the home-relative default (see
[Subagents](#subagents)), and `hook_manifest`, the Hook Manifest that names the Guard Hooks
every Bash Tool Call is run past (see [Guard Hooks](#guard-hooks)). Nothing else belongs in
it — those five are the whole set: the Model name and `--max-turns` stay flags, so a file
can move a run's traffic, point it at a roster and point it at the guards, but never change
what a run does. It lives at
`$HOME/.config/spindrift-driver/config.json`, or at the path `$SPINDRIFT_CONFIG` names,
which is used verbatim.

Precedence is one ladder for every field: the flag, then the environment variable
(`SPINDRIFT_BASE_URL` for the URL, `SPINDRIFT_API_KEY` for the key), then the Driver
Config, then the built-in default — a knob an operator turned by hand always outranks
content baked into a Box. `timeout` has no environment variable, so for it the ladder is
flag, Driver Config, default; `agents_dir` and `hook_manifest` have neither a flag nor a
variable, so for them the file is the only place they can be set.

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

## Guard Hooks

A Guard Hook is a script the host bakes into the Box that the Driver runs around a Bash Tool
Call — before it, to decide whether the command may run and what command that is, and after
it, to summarise the Tool Result the Model reads. The Hook Manifest is the JSON file that
lists them. Its path comes only from the Driver Config's `hook_manifest` — there is no
default path, and no flag or environment variable names it. With no `hook_manifest` in the
file the run is unconfigured and unguarded: no manifest, no hooks, nothing inspected.

```
{"pre": [{"path": "/usr/lib/spindrift-hooks/deny-credentials.sh", "tools": ["Bash"], "rewrites": false}],
 "post": {"path": "/usr/lib/spindrift-hooks/summarise-output.sh", "tools": ["Bash"]}}
```

The top-level object has two fields. `pre` is an array of hooks run in manifest order — the
file order is the run order, sorted by nothing, and the first refusal stops the chain. Each
entry names `path` (the script to exec), `tools` (the hook-protocol tool names the entry
applies to — `Bash` is the Bash Tool — an empty or absent list meaning every tool, so a hook
that forgets to name its tools guards rather than skips), and `rewrites` (whether the hook
rewrites the tool input). `post` is one object rather than an array — the contract is one
summary hook, not a chain — and its entry names `path` and `tools` only. `rewrites` on it is
refused as an unknown field, because rewriting the Tool Result is what the post-call script
exists for, so the flag would only be a switch nothing reads.

A pre-call hook is handed one JSON object on stdin —
`{"tool_name":"Bash","tool_input":{"command":"…"}}` — and answers on stdout with
`hookSpecificOutput`, whose `permissionDecision` is the verdict, whose
`permissionDecisionReason` is the text the Model is given, and whose
`updatedInput.command` replaces the command the run goes on with. A `deny` blocks the call:
the command is never run, and the Tool Result comes back as an error naming the script and
the reason. Silence passes — an empty stdout on a clean exit, or an absent or null
`hookSpecificOutput`, is a hook with no opinion.

A rewrite is applied to the rest of the chain and to what executes: it is the command the next
hook is asked about, and the rewritten command is the command that runs. One handed back by a
hook that did not declare `rewrites` is applied anyway — the flag is a promise about the
answer a hook owes, never a gate on applying it. A hook that does declare `rewrites: true`
owes that answer for any command worth rewriting, and one that returns no rewrite is refused
by name, since the Driver cannot tell "there was nothing to sanitize" from a hook that stopped
answering halfway; the empty command owes nothing, because there is nothing in it to rewrite.
The `tool_call` event still reports the command the Model asked for: a rewrite is host policy
on what executes, not a claim about what was requested.

Anything else fails closed, because the Driver cannot tell a broken hook from an approving
one: a script that cannot be run, a hook that overruns its per-hook bound of 5s
(`guardHookTimeout`), a hook that writes more than 1 MiB of stdout
(`guardHookOutputLimit`), a hook that exits non-zero, and an answer that is nothing readable
all block the call exactly as a deny does.

`jq` is a precondition of a guarded run. The baked hook scripts read the request above out of
their own stdin with it, so a run that cannot find `jq` refuses every guarded call naming `jq`
rather than letting a hook exit 0 having inspected nothing — that silence reads as a pass.
A run whose Hook Manifest is unconfigured needs no `jq`, because nothing is guarded and so
nothing needs reading.

After the call, the `post` hook is handed the pre-call request with `tool_response` added —
`{"tool_name":"Bash","tool_input":{"command":…},"tool_response":…}` — carrying the command
that actually ran, which after a rewrite is not the one the Model asked for, and the finished
Tool Result exactly as the Model would otherwise receive it. It answers with
`hookSpecificOutput.updatedToolOutput`, which replaces the Tool Result both on the wire and in
what the Model sees next. A summary hook fails open where the
chain fails closed, and each failure leaves the original output unchanged: a call that already
ran has earned its result. The cases are the ones the chain refuses for — a script that cannot
be started, a run past the same 5s (`guardHookTimeout`) and 1 MiB (`guardHookOutputLimit`)
bound, a non-zero exit, a flood of stdout, an answer that is nothing readable, and an empty
replacement, which would trade a real result for the empty string. The Driver's own 10 KiB
Tool Result cap (`bashToolMaxOutput`) stays in place as the floor under a replacement, so a
hook cannot summarise upwards. A refused call is never summarised.

A `hook_manifest` naming a file that cannot be read, or JSON that does not parse, is a
startup error at exit 2 naming the file: the run refuses to start unguarded rather than
discovering the missing file mid-task, and a malformed `post` stops the run exactly as a
malformed `pre` does. A spawned child inherits the manifest exactly as it
inherits the Driver Config, so delegation is no way around the guards.

## Subagents

A subagent is a roster entry the Model delegates to, and the roster is a directory of
Agent Files: one file per entry, `*.md` files only, at `$HOME/.config/spindrift-driver/agents`
or at the directory the Driver Config's `agents_dir` names. An absent directory at the
default path is not an error — it is a run with no subagents — but a directory an
`agents_dir` names has to exist, and an Agent File that will not parse is a startup error
naming it, the way a file the operator wrote always is — as is a repeated frontmatter key,
or two Agent Files claiming the same `name:`, which the driver refuses rather than
resolving to whichever file sorts first. Entry order is the directory's lexical order, so
the offer a Model sees is reproducible.

An Agent File is frontmatter over a body:

```
---
name: scout
description: Maps a repository and reports where a change belongs.
model: vendor/model-x
tools: bash
spawns:
- worker
---
You are a scout. Read, and report paths; never edit.
```

`name` and `model` are required. `model` is the child's Model id, used verbatim — no
effort suffix is parsed off it — and `--model` still overrides it for a run started with
`--agent`, as it does for any run. The body below the frontmatter is the child's system
prompt. `tools` and `spawns` are read and recorded, not enforced: every run gets the Bash
Tool whatever its roster says, and no roster limits whom a run may delegate to.

The Subagent Tool is offered only when the roster holds at least one Agent File — with
none there is nobody for a call to name — and a call takes the `agent` to delegate to and
the `task` to hand it. The child is this same binary re-executed with `--agent <name>` and
the task as its prompt; it inherits the working directory, the environment, the Driver
Config and the Hook Manifest, which is how it reaches the same Model server, the same roster
and the same Guard Hooks as its parent. That inheritance is the environment's and the
config file's, not the command line's: `--base-url`, `--timeout` and `--max-turns` are not
passed on, so a child dials whatever its own environment and Driver Config name and runs
under the default turn cap.
A child is never offered the Subagent Tool, so delegation goes one level deep.

Every event the child writes is relayed onto the parent's stdout as it arrives, with
`role` set to the agent's name and `in_subagent` true, and the child's `usage` events roll
into the parent's `result` token totals — a run's totals include what it delegated. The
task's answer comes back as the Subagent Tool Result, a `tool_result` whose `subagent`
names the agent that produced it. A child that fails — non-zero exit, or no `result` event
at all — yields an error Tool Result and the parent continues: the Model decides whether to
retry, delegate elsewhere, or answer. Spawns run one at a time, so a Model that asks for
three subagents in one turn is served them in order.

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
non-zero: unreachable server, rejected request — its response body verbatim in the error
event, and a rate limit that names a reset puts it in `reset_at` — a stream that goes
silent for longer than `--timeout` (default 5m0s — a stalled stream, not a long
generation: raise it when the server thinks for a long time before its first token), the
Repeated-Call Breaker refusing one call three times in a row, or the `--max-turns` cap
reached without an answer.

## Not packaged

llama.cpp is deliberately not packaged here. The Model server is always remote: the
Driver's job ends at the OpenAI-compatible endpoint, and packaging an inference
engine would put a large CUDA/GPU dependency tree between this repo and a build.

Decisions behind these boundaries are in [docs/adr/](docs/adr/).
