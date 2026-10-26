# Go, standard library only

Spindrift is written in Go, built as far as possible on the standard library. The OpenAI-compatible chat completions endpoint is a single HTTP call with JSON payloads, so `net/http` and `encoding/json` suffice — no LLM SDK, no HTTP client library, no dependency with transititive supply-chain risk. The bash tool uses `os/exec`. Zero third-party dependencies is the goal; any exception must be justified in a future ADR.

## Considered Options

- **Python with httpx** — lowest friction, but pulls a dependency tree.
- **TypeScript/Node** — heavier toolchain and package-manager supply chain.
- **Go, stdlib-only** — chosen: single static binary, `go test` built in, minimal supply-chain surface.
