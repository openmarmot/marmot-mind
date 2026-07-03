# Speak Tool Refactor Plan

**Goal**: Radically improve audio output quality in this voice-first AI agent by making all user-facing speech an explicit, deliberate action the LLM must take via a dedicated `speak` tool.

**Date**: 2026-07-03  
**Context**: This document is intended to be passed to Grok Build (or another coding agent) for implementation. It describes the problem, the proposed architectural change, benefits, and a step-by-step implementation plan.

---

## 1. Problem Statement

The current architecture works as follows:

- User speaks (push-to-talk hotkey) or types → client sends to `/connect`
- Server: STT (whisper.cpp) → transcription + conversation context → LLM (OpenAI-compatible, tool-calling enabled)
- LLM runs a ReAct-style multi-turn loop with available tools (currently `run_terminal`)
- Loop continues while the LLM returns `tool_calls`. It stops when the LLM emits plain `content` (no tool calls).
- The final plain text content is sent to TTS (Kokoro FastAPI `/v1/audio/speech`)
- Audio (base64) + transcription + text is returned to the client (or queued for proactive `/poll` messages)

**Core Problem**: LLMs are trained to produce visually structured output (markdown tables, code blocks, bullet lists, URLs, dense paragraphs, etc.). These formats are excellent for CLI or screens but sound terrible when read aloud by TTS. Examples of bad audio output:
- "Pipe table with column one, column two..."
- Long bullet lists without verbal flow
- Code snippets or JSON dumped verbatim
- Overly verbose or visually organized responses

This results in poor user experience for the primary interface (voice/audio).

The current design has no strong incentive or mechanism forcing the LLM to produce *natural, spoken-language-optimized* text for the final response.

---

## 2. Proposed Solution

**Make spoken output a first-class tool call.**

Instead of the LLM eventually emitting plain text that gets TTS'd, require the LLM to communicate with the user **exclusively** by calling a new `speak(text: str)` tool.

- All internal reasoning, tool use (`run_terminal`, future tools), calculations, memory access, etc. remain private (in the ReAct scratchpad / tool results).
- Only text passed to the `speak` tool is converted to audio and delivered to the user.
- **All user-facing output is unified**: the `speak` tool always performs TTS and enqueues the result via the proactive queue mechanism (`queue_proactive_message`). Delivery to the client (whether for direct user input or server-initiated messages) always happens via `GET /poll` + client-side buffering. `/connect` no longer returns model-generated text or audio.
- After executing `speak`, the server returns a tool result to the LLM (e.g., `{"status": "audio queued for delivery", "text": "what was spoken"}`) so the model knows it has spoken.
- `speak` is **not terminal**: the model may call speak (e.g. "checking..."), continue with other tools, then speak again. The loop continues for any tool_calls.

This is a deliberate architectural inversion: the LLM must *choose* to produce user-facing output, and when it does, it is incentivized (via prompt + tool description) to make it natural spoken language.

This pattern is common in more advanced agent systems (treating final user output as an explicit action) and aligns well with voice-primary design.

---

## 3. Benefits of This Approach

- **Solves the formatting problem at the source**: The model is forced to generate TTS-friendly text (complete sentences, natural flow, verbalized structure like "There are three key points. First...").
- **Cleaner separation of concerns**: Internal agent work vs. what the user actually hears.
- **More agentic behavior**: The model can decide *when* to speak, *what* to say, and can interleave tool use with speech (e.g., "I'm checking that for you..." then later speak the result). It can also choose *not* to speak after completing internal work.
- **Fits existing architecture**:
  - Works for both reactive user inputs (via `/connect` which records the turn and kicks off processing) and proactive messages (cron jobs + `/poll` + human detection via YOLO). All spoken output uses the same queue + poll path.
  - The ReAct loop already exists; we are extending the set of tools and changing the stopping condition (loop continues while any tool_calls are returned; speak is just another tool).
- **Future-proofing**: Easier to add prosody/SSML, multiple voices, emotion, rate limiting on speech, or "thinking out loud" vs. final answer speech later.
- **Better user experience**: Responses will sound more natural and conversational when heard.

---

## 4. Implementation Steps (Recommended Order)

### Step 0: Preparation & Backup
- Review current tool schema, ReAct loop logic, and system prompt in `server/code/server.py` (and `config.json` / prompt handling).
- Confirm TTS endpoint behavior (Kokoro `/v1/audio/speech` returns audio bytes or base64?).

### Step 1: Define the `speak` Tool
- Add a new tool definition (following the existing `run_terminal` pattern) named `speak`.
- Schema (JSON schema style for OpenAI-compatible tool calling):
  - `name`: "speak"
  - `description`: Detailed description emphasizing natural spoken language. Key points to include:
    - "Use this tool to communicate anything to the user via audio. The text must be natural, conversational spoken English optimized for text-to-speech."
    - "Use complete sentences. Short paragraphs are better than long walls of text."
    - "Verbalize structure instead of using visual formatting: say 'There are three important points. First...' instead of bullet lists or tables."
    - "Read numbers, dates, and times naturally (e.g., 'eighty two degrees' not 'eighty two point zero')."
    - "Avoid markdown, code blocks, URLs (say the domain or summarize), and raw data dumps."
    - "Be concise but helpful. One main idea or response per speak call is often best."
  - Parameters:
    - `text` (string, required): The exact text to be spoken to the user.
    - (Future) Optional: `voice`, `speed`, `emotion` if Kokoro or the TTS layer supports it.
- Add the tool to the list of available tools sent to the LLM in each turn.

### Step 2: Update the ReAct / Agent Loop Logic in server.py
Current loop roughly:
```python
while tool_calls and turns < MAX:
    execute tools
    append tool results
    call LLM again
# then TTS on final content
```

New desired behavior (high level):
- Continue the loop as long as the LLM returns `tool_calls` (subject to limits). `speak` is just another tool — the model can speak progress ("checking the files..."), do more work, speak again, etc.
- When `speak` is called:
  - Execute TTS on the `text` parameter (call Kokoro endpoint).
  - Queue it for delivery using the existing proactive queue (`queue_proactive_message`).
  - Append a tool result message to the messages for the LLM, e.g.:
    ```json
    {"role": "tool", "tool_call_id": "...", "content": "{\"status\": \"audio queued for delivery to user\", \"text_spoken\": \"the exact text\"}"}
    ```
- The loop ends only when the LLM emits a message with no tool_calls (or limits reached).
- If the loop ends without any `speak` call: nothing is queued for the user. Log it. No fallback TTS of internal content. For a user turn this simply means the user only sees their own transcription.
- Remove all paths that took "final content" and auto-TTS'd it for the user.
- `process_with_llm` (or equivalent) no longer returns user-facing text. Callers (especially cron) no longer take a return value and queue it — speaks inside the run do the queuing.
- User input path (`/connect`): record the user turn in `conversation_history`, then run the agent loop. Return only the transcription (no model text/audio). The client relies on `/poll` for any replies.
- For **proactive / cron / inject**: unchanged in spirit. Speaks during internal runs enqueue items. Delivery and history append still happen in `/poll`.
- All spoken output (reactive replies and proactives) uses the same queue + poll + client drain path.

### Step 3: Update Conversation Context & History Management
- Ensure that spoken text is properly represented in the persistent conversation context (so the model remembers what it has already told the user aloud).
- Recommended: After a successful `speak`, also append a normal `assistant` message containing the spoken text (or mark it clearly as spoken). This keeps history clean for future turns.
- The existing practice of discarding intermediate tool scratchpad per turn while keeping user + final assistant messages should be adapted: the "final assistant" message should now come from (or include) what was passed to `speak`.
- Keep the rolling context trimming logic (150k tokens, inactivity timeout) intact.

### Step 4: System Prompt Engineering (Critical)
Add or expand a dedicated section in the system prompt (likely in `config.json` or wherever `SYSTEM_PROMPT` is defined/loaded). Suggested content:

```
## VOICE COMMUNICATION PROTOCOL (VERY IMPORTANT)

You are a voice-first AI assistant. The human interacts with you primarily through audio.

- You communicate with the user **exclusively** by calling the `speak` tool. 
- Nothing you output outside of a `speak` tool call is ever heard by the user.
- Internal reasoning, tool results, and chain-of-thought are private scratch space.
- When you call `speak(text=...)`, the text is converted to speech and played to the user.

### Rules for what you put inside `speak()`:
- Use natural, flowing spoken English that sounds good when read aloud.
- Complete sentences and short paragraphs. Avoid walls of text.
- Verbalize structure: "There are three key results. The first is X with value Y. The second is..."
- Read numbers and times naturally ("ninety five degrees", "July third", "three point one four").
- Do NOT include markdown tables, code blocks, raw JSON, long URLs, or bullet points unless you verbalize them.
- Be concise but complete. One focused response per speak call is often ideal.
- You can speak multiple times in a turn if needed (e.g., acknowledge, then later give final answer), but prefer clarity.

### Examples of good speak text:
- "The current temperature in Tucson is 95 degrees and it's sunny."
- "I found three relevant files. The first one is called server.py and it handles the main logic. The second..."

### Examples of text to avoid putting in speak():
- Any markdown table or code fence.
- "Here is the output: ```json ...```"
- Dense bullet lists without verbal connectors.

If you have nothing useful to tell the user after completing internal work, you do not need to call speak at all.
```

Include 2–3 concrete good/bad examples in the prompt. You can also generate variations with another LLM if helpful.

Update any existing instructions about final answers or response style to reference the `speak` tool instead.

### Step 5: Modify API Response & Client-Facing Output
- `/connect` becomes an **input-only** endpoint:
  - Returns primarily `{"transcription": "...", "status": "processing"}` (or similar minimal response).
  - No model `text` or `audio` in the response. The model only "speaks" by queuing via the tool.
- All AI responses (even direct replies to user input) are delivered exclusively via the existing `/poll` mechanism and client proactive handling.
- Update client:
  - After sending to `/connect`, only print the user's transcription.
  - All Marmot speech comes from the poller / local pending queue (same as before for proactives).
  - Remove `-m` (text message) support for now.
  - Remove dashboard chat UI/JS for now (status dashboard can remain).
- `/poll` and queued items stay the same shape.
- Update logging to distinguish internal tools from `speak` calls.

### Step 6: Handle Edge Cases & Robustness
- **No speak called**: Define clear fallback (see Step 2). Consider returning a very short confirmation audio or no audio + a log message.
- **Multiple speak calls in one turn**: Decide on policy (concatenate and play sequentially, or only honor the last one). Start simple (allow and concatenate).
- **Long-running tool sequences**: The model can call `speak` mid-loop to give progress updates ("I'm searching the web for that information now...") before the final answer. This is a nice emergent benefit.
- **Error during TTS**: Catch errors in the `speak` tool execution and return a tool result indicating failure so the model can retry or apologize.
- **Context length**: Speaking adds to history. Keep the existing trimming logic.
- **Proactive gating**: Human detection (`/detect` + YOLO) should still happen *before* playing audio from a proactive `speak`. The gate can remain on the client or be moved earlier.

### Step 7: Testing & Iteration
- Test with simple queries first ("What's the weather like?", "List the files in my home directory").
- Verify that tables/lists from internal tools are *never* spoken raw.
- Test proactive/cron paths.
- Test multi-turn conversations (does the model remember what it previously spoke?).
- Listen to output quality and refine the `speak` tool description + system prompt examples iteratively.
- Add metrics/logging around: number of speak calls per turn, length of spoken text, cases where no speak was called.
- Consider adding a simple eval: after a turn, check whether the spoken text contains forbidden patterns (markdown, code fences, etc.) and log warnings.

### Step 8: Documentation & Future Work
- Update `README.md`, `docs/API.md`, and `AGENTS.md` to describe the new `speak` tool and the voice output contract.
- Document the new turn termination behavior.
- Future enhancements to consider later:
  - Make `speak` support SSML or prosody tags if Kokoro evolves.
  - Add a separate "think_aloud" or internal monologue tool vs. final `speak`.
  - Rate limiting or cooldown on speech for proactive messages.
  - Multiple voices or voice selection per user/session.

---

## 5. Key Design Decisions (Locked)

1. `speak` is **not** terminal — the model is encouraged to speak progress ("checking...", "working on it...") then continue the tool loop before a final speak.
2. Multiple `speak` calls are fully supported. For a single agent run they will produce multiple queued items. Client plays them sequentially via existing drain/poll logic.
3. Zero `speak` calls after a user input or cron run: no output is delivered to the user. Just log on server. This is valid autonomous behavior.
4. Spoken content is persisted when delivered via `/poll` (same as today's proactive append to `conversation_history`). User turns are appended on `/connect` receipt.
5. Client no longer receives direct model responses on `/connect`. All speech (including replies) uses poll path. Clipboard/play happens on delivery. `-m` and dashboard chat temporarily removed.
6. `/connect` is input-only. Everything the AI says to the user goes through `speak` → queue → poll.

---

## 6. Summary for Implementation

We are changing the fundamental output contract of the agent:

**Before**: LLM reasons with tools → eventually emits text → server TTS's whatever text it emitted (returned directly on /connect or queued).

**After**: LLM reasons with tools (including the new `speak` tool) → only text explicitly passed to `speak(...)` ever becomes audio for the user. **All** such audio is queued and delivered exclusively through the `/poll` system (for both user replies and proactives). `/connect` is strictly for input.

This requires:
- New `speak` tool definition + detailed description
- Refactor of agent loop: speak is a normal tool, loop continues on tool_calls (non-terminal), no more "final content" output path
- `/connect` simplified to record user turn + trigger processing + return transcription only
- Client simplified to send input and rely on poller for all output
- Significant system prompt updates with clear rules and examples
- Removal of `-m` mode and dashboard chat (for this phase)
- Updates to cron, history, and delivery paths (largely reuse existing)
- Tool limits (global + per-tool, especially web_search) and fallbacks

The result should be dramatically more listenable, natural voice responses while making the server-side AI more autonomous (it chooses when and what to speak via the tool).

This is a high-leverage refactor for a voice-primary system.

---

**End of Plan**

Implementation target: server/code/server.py (core loop, tools, /connect, cron), client/code/client.py (send + remove -m), config.json (prompt), plus docs updates after core works. 

After implementation, thorough listening tests + prompt iteration will be needed.
