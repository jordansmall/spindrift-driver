# Events are the only stdout contract

stdout carries driverkit events and nothing else — JSONL, one event per line, on every exit path — and there is no output-mode flag. A flag would make stdout two contracts and default to the one the coordinator cannot parse. The text-only stdout of spec #1 (final answer only, read by eye) is retired with it: the final answer is the `result` event's `text`, so a single stream loses nothing. What does not move stays put: progress — each `POST`, each retry, the one-line driver error — remains on stderr, `-h` writes its usage to stderr, and exit codes stay the authoritative signal for success and failure: 0 with a `result`, 1 for a failed run, 2 for a usage error.

driverkit is the grammar's source of truth: `spindrift`'s `cmd/launcher/internal/driver/driverkit` defines the kinds, the fields each kind fills, the snake_case wire names, and timestamps in epoch milliseconds. If driverkit changes, this grammar follows it and an ADR records the lineage shift. This Driver emits `start`, `text`, `usage`, `tool_call`, `tool_result`, `result` and `error`, and nothing else — no speculative fields for kinds it has not implemented.

The committed golden Pass log, `testdata/pass.jsonl`, is what the host-side Strategy consumes to write a decoder against: one scripted Pass at a pinned clock, byte for byte. Because this repo carries no copy of driverkit, the tests here assert the field contract per kind by reading the emitted lines as plain JSON rather than through driverkit types — they pin the wire, not one implementation of it.

## Considered Options

- **Keep text stdout, add a `--json`/`--events` flag** — rejected: two contracts, and the default one drops tool calls, usage and errors, which is precisely what spindrift needs.
- **Events on stdout, plus the answer as trailing plain text** — rejected: one stream with two parsers, and the `result` event already carries the answer verbatim.
- **Progress on stdout too** — rejected: progress is for whoever is watching, and every decoder would have to filter it out before parsing.
- **Assert the contract through a copy of driverkit's types** — rejected: a second source of truth that fails silently; plain JSON keeps driverkit the only one.
