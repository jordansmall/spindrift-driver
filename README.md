# spindrift-driver

A **Driver** for the spindrift coordinator: one prompt in, one final answer out.
It talks to an OpenAI-compatible **Model** server, offers that Model a single Bash
**Tool**, runs the Tool Calls the Model asks for, and loops until the Model answers.

```
usage: spindrift-driver "prompt" --model <model> [--base-url <url>] [--max-turns <n>] [--timeout <dur>]
```

The prompt is positional or on stdin; `--model` is required; `--max-turns` caps the
agent loop (default 50). `--timeout` bounds each chat attempt — dial, response headers
and body read — and defaults to 5m0s; raise it (`--timeout 15m`) when the server
generates slowly, which a local model often does.

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

## Manual live smoke test

Manual by design — it needs a Model server that is actually running, so it is run by
hand and is not part of `nix flake check` or CI.

1. Have an OpenAI-compatible chat completions server reachable from this machine.
2. Point the Driver at it, with `--base-url <url>` or `$SPINDRIFT_BASE_URL` (default
   `http://localhost:8080/v1`). The `/chat/completions` path is appended for you.
3. If the server expects a bearer token, export it: `export SPINDRIFT_API_KEY=…`.
   The key is read from `$SPINDRIFT_API_KEY` only — there is no flag for it, so it
   never lands in the process listing — and with it unset no `Authorization` header
   is sent at all.
4. Run it:

   ```
   ./result/bin/spindrift-driver "List the Go files here, then say what main.go does." --model <model>
   ```

Expected: the Model's final answer on stdout, exit status 0. Progress — each
`POST`, each retry — goes to stderr, so stdout stays clean enough to pipe. A failure
prints one line to stderr and exits non-zero: unreachable server, rejected request, an
attempt that exceeds `--timeout` (default 5m0s — raise it for slow generation), or the
`--max-turns` cap reached without an answer.

## Not packaged

llama.cpp is deliberately not packaged here. The Model server is always remote: the
Driver's job ends at the OpenAI-compatible endpoint, and packaging an inference
engine would put a large CUDA/GPU dependency tree between this repo and a build.

Decisions behind these boundaries are in [docs/adr/](docs/adr/).
