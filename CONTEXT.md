# spindrift-driver

A minimal LLM agent harness that runs as a **Driver** in the spindrift coordinator's sense (see the spindrift repo's CONTEXT.md): a headless, one-shot agent CLI that the coordinator bakes into a Box and invokes. Its first customer is itself: the bare minimum needed for the Driver to run inside a repo and improve its own code. It presents tools to a Model, executes the tool calls the Model makes, and loops until the Model produces a final answer.

## Language

**Driver**:
This program — the headless agent CLI that spindrift invokes: one-shot prompt in, final answer and session record out. One of the swappable Drivers beside `claude`, `opencode`, and `pi`.
_Avoid_: harness (that's spindrift itself), agent CLI, backend

**Driver Config**:
The JSON file a run reads its settings from — the Model server's base URL, an optional API key, and `timeout`, the idle timeout on a chat attempt — and no other setting, the Model name and `--max-turns` being flags a file cannot supply. It lives at `$HOME/.config/spindrift-driver/config.json` unless `SPINDRIFT_CONFIG` names another path, which is used verbatim, and spindrift writes it through `envFiles`, its only run-time channel into a Box. For each field the file ranks below the flag and that field's environment variable and above the built-in default, and it is read strictly: a file missing when named, malformed JSON, an unknown field, or a value of the wrong type is a startup error, never a silent fallback, and nothing of the kind is ever read from the working directory or the Target repo.
_Avoid_: dotfile, project config, settings file, argv

**Model**:
The LLM behind the OpenAI-compatible endpoint the Driver talks to. The thing that receives context and chooses tool calls or a final answer.
_Avoid_: agent (a running Driver process, in spindrift's parlance), assistant, LLM

**Tool**:
A capability the Driver offers to the Model, described in the Model's tool schema and executed by the Driver when called.
_Avoid_: function, plugin, skill

**Tool Call**:
A structured tool invocation the Model returns in its response (via the OpenAI `tools` API, parsed server-side) — name, arguments, and an id the Driver pairs with the Tool Result.
_Avoid_: function call, action

**Tool Result**:
The output the Driver sends back to the Model for one Tool Call, in the next request.
_Avoid_: observation, output

**Bash Tool**:
Runs a shell command in the run's working directory and returns its output as the Tool Result.

**Agent File**:
One entry in the subagent roster, kept as a markdown file in a directory that defaults to `$HOME/.config/spindrift-driver/agents` and is redirected only by the Driver Config's `agents_dir` — one file per entry, every `*.md` file in it an entry. Its frontmatter carries `name`, `description`, `model`, `tools` and `spawns`; `name` and `model` are required, and the `model` id is used verbatim as the child's Model, with no effort or vendor suffix parsed out of it. The body below the frontmatter is the child's system prompt. `tools` and `spawns` are read and recorded, not enforced.
_Avoid_: profile, persona, yaml config, subagent definition, agent card

**Subagent Tool**:
The Tool that delegates one task to a roster entry named in the call: the Driver re-executes this same binary as a child run with `--agent <name>`, so the child runs on its own context from the Agent File's model id and system prompt, and inherits the working directory, the environment and the Driver Config. The child's events are relayed into the parent's stream with `role` set to the agent's name and `in_subagent` true, and its usage rolls into the parent's result totals. The child's final answer is the Tool Result, which names the agent in `subagent`; a child that fails is an error Tool Result and the parent goes on. Spawns run one at a time, and a child run is never offered this Tool, so delegation stops at one level.
_Avoid_: subprocess, fork, nested agent, MCP agent, tool forwarding

**Event**:
One canonical driverkit fact the Driver reports on stdout: one JSON object on one line, named by its `kind` and timestamped in epoch milliseconds. These events are the whole stdout contract — a host decodes a run event by event, it does not scrape text.
_Avoid_: log line, record, frame, printout

**Pass log**:
The whole JSONL stream one run writes to stdout — its events in emission order, with the run's stderr interleaved by the recorder — decoded line by line into the session record. `testdata/pass.jsonl` is the committed golden Pass log of one scripted run.
_Avoid_: transcript, log file, stdout dump
