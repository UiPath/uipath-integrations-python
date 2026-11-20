# Chat agent (conversational)

A meeting-room assistant you chat with. It looks rooms up with a local tool
(`find_rooms`, no tenant resource behind it), reads files you attach with
`Analyze_Files`, and remembers what you told it earlier in the conversation.

The only thing that makes it a chat agent is `uipath.json`:

```json
{ "runtimeOptions": { "isConversational": true } }
```

`main.py` is an ordinary `UiPathCodexAgent` with `base_instructions` as the
system prompt and no `prompt`, `input_schema` or `output_schema` (a chat
agent's input and output are the UiPath conversation messages).

## How an exchange works

1. The runtime takes the newest user message from the `messages` input (its
   text, plus any uploaded files as job attachments).
2. It resumes the conversation's Codex thread (restored from `state.db`, so
   it works on a fresh robot too) and runs one turn with that message.
3. The reply streams as chat events: text chunks as they arrive, and a tool
   card for each `find_rooms` / `Analyze_Files` call, on the message that
   made it.
4. The thread is saved and the run ends **suspended with no trigger**: the
   job waits for the next message, and the chat host closes the exchange.

## Run locally in the chat UI

```bash
cd samples/chat-agent
uv sync
uv run uipath auth            # pick the tenant, e.g. --alpha
uv run uipath dev
```

In the browser, start a **Chat** run on `agent` and ask:

1. "I need a room for 6 people on Tuesday at 10:00 for the Kingfisher
   review. What fits?" A `find_rooms` card shows `{"people": 6}` and its
   result, then the reply streams in (Beehive).
2. "Remind me which room you suggested and what the meeting was called."
   The answer names Beehive and the Kingfisher review: both come from the
   first exchange, so the thread was resumed.

With a durable tool added (step 5 of "Run on Orchestrator"), an exchange
that waits on a job keeps the run running: `uipath dev` checks the job
every few seconds, and once it has finished the card closes with the
result and the reply continues on its own. Wait for the reply before you
send the next message; one typed while the run is busy is dropped.

## Run locally from the CLI

```bash
uv run uipath run agent --file input.json                  # first message
uv run uipath run agent --file next_message.json --resume  # the next one
```

`--resume` keeps `__uipath/state.db` (the conversation) between the runs;
without it a local run starts a new conversation. The console reports each
run as successful with an empty output (no `__uipath/output.json` is
written): `SUSPENDED` is what a job (with `UIPATH_JOB_KEY` set) writes to
`__uipath/output.json` and what Orchestrator sees. The console shows the
tool calls (state events) but not the reply, which is a chat event. To see
the reply as a chat host gets it, run as a deployed chat job does:

```bash
export UIPATH_JOB_KEY=00000000-0000-0000-0000-000000000001   # any GUID
export CAS_WEBSOCKET_DISABLED=true
```

and add the conversation ids to `uipath.json`:

```json
{
  "runtimeOptions": { "isConversational": true },
  "fpsProperties": {
    "conversationalService.conversationId": "local-conversation",
    "conversationalService.exchangeId": "exchange-1"
  }
}
```

Then `uv run uipath run agent --file input.json` logs every envelope the
chat bridge would send (`startMessage`, the text chunks, the tool calls,
`endMessage`, `endExchange`) to `__uipath/execution.log` instead of sending
it. Change `exchangeId` and add `--resume` for the next message. Remove
`fpsProperties` again before you pack.

## Run on Orchestrator

```bash
uv run uipath pack
uv run uipath publish         # pick a folder (or --my-workspace)
```

The package's `content/operate.json` carries `"isConversational": true`
(copied from `uipath.json`).

1. Create the process from the package in that folder (Orchestrator →
   Automations → Processes), with a robot or serverless runtime.
2. Open it from the Agents / Autopilot chat, which lists conversational
   agents, and start a conversation with the two messages above.
3. Check that:
   * the reply streams and the `find_rooms` card shows its input and result;
   * the job goes to **Suspended** after each exchange, and the same job
     resumes on the next message;
   * the trace has one `Conversational agent run - agent` span per exchange,
     with `isConversational`, `conversationId`, `exchangeId` and `threadId`.
4. Upload a file (e.g. a meeting agenda PDF) and ask "What is on the
   agenda?". An `Analyze_Files` card reads it.
5. Optional, a durable tool inside an exchange: add the
   [`durable-process-tool`](../durable-process-tool) sample's `a_plus_b`
   process tool to `tools` and ask for a sum. Its card stays open and the
   exchange stays open while the job runs; when the job finishes, the job
   resumes, the card closes with the result and the reply continues.

## Trace

```
Conversational agent run - agent     isConversational, conversationId, exchangeId, threadId
├─ LLM call → Model run
├─ Tool call - find_rooms
└─ LLM call → Model run
```
