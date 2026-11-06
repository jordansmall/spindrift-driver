# Standalone first; CLI shaped to the spindrift Driver contract

Superseded by [ADR-0003](0003-spindrift-first.md): spindrift is the primary consumer now, so its interface — not a standalone CLI shape — defines this Driver.

The primary consumer of this Driver is the spindrift coordinator, but full integration (in-box nix row, host-side Strategy, conformance corpus) is deferred. We build a standalone one-shot CLI whose argv shape already matches spindrift's Driver `argvShape` — positional prompt, `--model` flag — so integration later is additive, not a retrofit. The bare minimum is the goal: the harness must be able to run and improve itself before anything else.

## Considered Options

- **Full Driver integration now** — rejected: Strategy + conformance work is a project of its own and premature before the core loop is proven against a real llama.cpp server.
- **Standalone, arbitrary CLI shape** — rejected: cheap now to match the contract, expensive to retrofit later.
