# The Driver Config reverses spec #1's "no config files" rule

The Driver reads one JSON file at startup for three settings — `base_url`, an optional `api_key`, and the idle `timeout` — and nothing is moved off argv to make room for it. The model name and the turn cap stay flags (`--model`, `--max-turns`), the endpoint and the key keep the environment variables that already carry them, and the file sits underneath both. Those three keys are the whole field set: a file a host writes cannot smuggle in a setting the operator has not approved, because there is no fourth setting in it to smuggle.

spec #1 said no config files, and that rule is reversed here — this file is the exception, and the only one. The reversal is narrow in what it costs: ADR-0001 (standard library only) is untouched, because a JSON file read with `encoding/json` pulls in no dependency, and the flags a run can pass are exactly the flags they were before.

The reason it exists is `envFiles`, spindrift's only run-time configuration channel into a Box, which makes the `driver config` the one thing spindrift can still write where the Driver will read it. Everything else about a Box is settled when the Box is built, and argv is fixed by the Driver row before the run starts — so content written into the Box at run time is the single place a host-side setting can travel. A flag cannot do this job for the same reason: the row has been read before any file could be written, and argv is a poor place to put anything a Box is meant to keep to itself.

The path has one fixed home, `$HOME/.config/spindrift-driver/config.json`, overridable by the `SPINDRIFT_CONFIG` environment variable. A variable rather than a flag, because a variable is already how the base URL and the key reach a Box, and because the path is a fact about the host: the default lives under `$HOME`, never the working directory, so no checkout can put a file where the Driver will read it.

One ladder applies to every field, in this order: the flag, then the environment variable, then the Driver Config, then the built-in default. A knob an operator turned by hand always outranks content baked into a Box.

An absent default file is not an error — it is the ordinary state of a machine with no config. Everything else about a file that was actually named is: a path in `SPINDRIFT_CONFIG` that does not exist, JSON that does not parse, an unknown field, and a field holding the wrong type are each a startup error naming the file and the problem, and the run exits with the code 2 that every other startup error already carries. The reason to be harsh about a typo is what the soft version costs — a key spelled `base_ur1` would fall back to the built-in endpoint, the Box would talk to the wrong server, and nothing on stdout would say so. A typo may never fall back quietly.

The API key comes from the environment or from this file, never from a flag, so it never appears in a process listing where any user on the host can read it. The file is therefore the one place a key sits at rest, which is why the strictness above matters most for it: a silently half-read key is a wrong key that still looks like it was accepted.

The Driver reads no config from its working directory, and that directory holds the Target repo. The repository under repair does not get to configure the run that repairs it: an endpoint, a credential and a timeout written into the checkout would let a Target repo aim its own repair somewhere else.

## Considered Options

- **argv flags only** — rejected: argv is fixed by the Driver row before the run starts and lands in the process listing, so per-run host settings — and the key that goes with them — have nowhere to go.
- **Environment variables only** — rejected: an env map is a poor home for a nested, typed, reviewable document, and `envFiles` writes files, not variables, so spindrift could not deliver one.
- **A project-local config in the Target repo** — rejected: the Target repo would steer its own repair — endpoint, credentials, timeout — and every Box built from a checkout would arrive pre-configured by the code inside it.
- **Tolerant parsing that ignores unknown keys** — rejected: a typo drops one setting silently and the Box quietly talks to the default endpoint, the exact failure this file's strictness exists to rule out.
- **A `--config` flag naming the path** — rejected: the path is part of how spindrift lays out a Box, and a flag would let an invocation disagree with the host that wrote the file.
