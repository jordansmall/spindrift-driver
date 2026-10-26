# spindrift-driver

A minimal LLM agent harness that runs as a **Driver** in the spindrift coordinator's sense (see the spindrift repo's CONTEXT.md): a headless, one-shot agent CLI that the coordinator bakes into a Box and invokes. Its first customer is itself: the bare minimum needed for the Driver to run inside a repo and improve its own code. It presents tools to a Model, executes the tool calls the Model makes, and loops until the Model produces a final answer.

## Language

**Driver**:
This program — the headless agent CLI that spindrift invokes: one-shot prompt in, final answer and session record out. One of the swappable Drivers beside `claude`, `opencode`, and `pi`.
_Avoid_: harness (that's spindrift itself), agent CLI, backend

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
The one Tool in scope: runs a shell command and returns its output as the tool result.
