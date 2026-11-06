# Spindrift first; the Driver's contract is what spindrift consumes

Supersedes ADR-0002 (standalone first). That ADR deferred integration and shaped a standalone CLI that merely resembled spindrift's Driver contract; the deferral is now reversed. Spindrift is this Driver's primary consumer, the target is spindrift's default Driver, and an interface exists only if spindrift uses it. Anything on no path from spindrift into a Box is out of scope, however sensible it looks from here.

Standalone runs keep working — flags, environment, a Pass log on stdout, progress on stderr — but only as a side effect of that interface, never as a second contract to design toward. Where the two pull apart, spindrift wins: stdout carries driverkit events rather than whatever a person typing at a shell would rather read.

This repo carries no copy of driverkit: standard library only (ADR-0001), no import, no vendored types. A forked copy of the grammar goes stale silently, and the wire is names and shapes, not Go declarations. So the hand-off between the two repositories is a file — the committed golden Pass log, `testdata/pass.jsonl`, one scripted run's stdout byte for byte, regenerated only by an explicit `-update` flag and failing on any other drift. When driverkit's kinds or fields change, that file drifts first, and an ADR records the lineage (ADR-0004).

## Considered Options

- **Standalone first, integrate later (ADR-0002)** — rejected: the core loop is proven, so the deferral now only maintains a second centre of gravity; a shape nobody integrates against is not a contract.
- **Full Driver integration in this repo** — rejected, as before: the host-side Strategy and the conformance corpus live in spindrift, and this repo's half is the interface and the artifact.
- **Vendor or copy driverkit's types here** — rejected: the copy fails nothing when it goes stale, and the conformance a host needs is on the wire, where only names and shapes exist.
