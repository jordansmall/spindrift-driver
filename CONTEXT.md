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

**Edit Tool**:
Replaces one exact string in one file, the call carrying the path, the old string, the new string, and a replace-all switch. Without `replace_all` the old string has to match exactly once, and a call that matches nothing or matches several never touches the file: it comes back as an error Tool Result saying `not found` or `N matches`, which is enough for the Model to correct the call in one step rather than guess at what the file held. A relative path resolves against the run's working directory, and the file's path rather than the strings is what the call reports as `tool_target`, so a Guard Hook sees the call as a read of that file and can refuse it before the file changes.
_Avoid_: sed, patch, diff, in-place edit

**Write Tool**:
Creates a file or overwrites one whole from a path and its content, making any parent directories that are missing on the way; a relative path resolves against the run's working directory, and a Guard Hook can refuse the call as a read of that path before anything is written. Its events name the Tool `write` and report the file's path as `tool_target` — the value the host-side Strategy watches to see that a run changed files rather than described changing them.
_Avoid_: heredoc, redirect, append, touch

**Guard Hook**:
A script the host bakes into the Box that the Driver runs around a Tool Call it guards. The Driver hands it `{tool_name, tool_input}` on stdin and reads its stdout for `hookSpecificOutput`, whose `permissionDecision` is the verdict and whose `permissionDecisionReason` is the text handed back to the Model. A `deny` blocks the call with an error Tool Result naming the script and the reason, and so does any hook that misbehaves — one that cannot be run, one that overruns its bound, one that exits non-zero, one that answers with nothing readable — the Driver cannot tell a broken hook from an approving one. Silence on a clean exit raises no objection: a hook with no opinion lets the call through. A hook may also answer with `updatedInput.command`, which replaces the command the pre-call run goes on with, and a hook the manifest declares a rewriter that answers with no rewrite for a command worth rewriting is refused by name like any other misbehaviour. `jq` has to be runnable for a guarded run at all: the baked scripts read their own request with it, so without it a hook would answer having inspected nothing, and a missing `jq` therefore refuses the guarded calls before any hook runs. The post-call variant is the summary hook: it is handed the finished Tool Result as `tool_response`, and its `updatedToolOutput` replaces that result. It fails open where the pre-call chain fails closed — a summary hook that cannot run leaves the result the command earned unchanged — because by then there is no command left to protect, only a result to lose. The seam exists for policy the Driver must not hold itself, a credential the Model would otherwise read straight out of the environment and the refusal of a backgrounded `bash` call among them; the Driver hardcodes no hook names and no policy, it runs whatever the Hook Manifest lists.
_Avoid_: middleware, interceptor, permission prompt, guardrail, filter

**Hook Manifest**:
The JSON file that lists the Guard Hooks the run applies: a `pre` list of the pre-call hooks in run order, the file order being the run order, each entry naming the script to run, the hook-protocol tool names it applies to, and whether it rewrites the tool input; and beside that `pre` list sits `post`, one entry rather than a list — the one post-call summary hook, whose entry names the script and the tool names only, there being no `rewrites` to declare after the call. Its path comes only from the Driver Config's `hook_manifest` and there is no default path, so an unconfigured run is simply unguarded — no hooks, nothing inspected; a manifest that is configured but cannot be read or parsed stops the run at startup rather than starting it unguarded. A spawned child inherits it the way it inherits the Driver Config, so delegation cannot step outside the guards.
_Avoid_: hook config, plugin list, policy file, pre-flight checklist

**Repeated-Call Breaker**:
The run limit that catches a Model going in circles: it keys on the tool name plus the exact arguments string the Model sent on the wire — the wire string rather than the decoded arguments, whose key order would be the Driver's rather than the Model's — and refuses the third identical Tool Call in a row, which is then not run. A refused call gets a synthetic Tool Result back, an error one worded as a refusal and asking for a different action, so the Model has something to act on rather than another result indistinguishable from the last. Any different call resets the count, a Model that changed course having stopped repeating itself; three refusals in a row end the run with an `error` event and a non-zero exit, which states why the run stopped instead of letting it burn the Pass timeout on a call that will only repeat. The count is per Driver process, so a child a Subagent Tool spawns starts fresh.
_Avoid_: loop detector, rate limiter, retry guard, watchdog

**Agent File**:
One entry in the subagent roster, kept as a markdown file in a directory that defaults to `$HOME/.config/spindrift-driver/agents` and is redirected only by the Driver Config's `agents_dir` — one file per entry, every `*.md` file in it an entry. Its frontmatter carries `name`, `description`, `model`, `tools` and `spawns`; `name` and `model` are required, and the `model` id is used verbatim as the child's Model, with no effort or vendor suffix parsed out of it. The body below the frontmatter is the child's system prompt. `tools` grants the Tools that entry's run is offered, by the Driver's own names (`bash`, `edit`, `write`, `subagent`) and in that order, and an entry listing none is offered the Bash Tool alone; `spawns` limits which roster entries that run may delegate to, and the top-level process, named by no Agent File, is unrestricted.
_Avoid_: profile, persona, yaml config, subagent definition, agent card

**Subagent Tool**:
The Tool that delegates one task to a roster entry named in the call: the Driver re-executes this same binary as a child run with `--agent <name>`, so the child runs on its own context from the Agent File's model id and system prompt, and inherits the working directory, the environment, the Driver Config and the Hook Manifest. The child's events are relayed into the parent's stream with `role` set to the agent's name and `in_subagent` true, and its usage rolls into the parent's result totals. The child's final answer is the Tool Result, which names the agent in `subagent`; a child that fails is an error Tool Result and the parent goes on. Spawns run one at a time. A child is offered this Tool only when its entry grants it and its `spawns` list names somebody it may hand work to, so delegation reaches as far as the roster allows and no further; a spawn outside that list is an error Tool Result naming the agents the caller may spawn.
_Avoid_: subprocess, fork, nested agent, MCP agent, tool forwarding

**Event**:
One canonical driverkit fact the Driver reports on stdout: one JSON object on one line, named by its `kind` and timestamped in epoch milliseconds. These events are the whole stdout contract — a host decodes a run event by event, it does not scrape text.
_Avoid_: log line, record, frame, printout

**Pass log**:
The whole JSONL stream one run writes to stdout — its events in emission order, with the run's stderr interleaved by the recorder — decoded line by line into the session record. `testdata/pass.jsonl` is the committed golden Pass log of one scripted run.
_Avoid_: transcript, log file, stdout dump
