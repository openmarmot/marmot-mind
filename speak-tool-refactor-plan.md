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
- The `speak` tool (when executed by the server) performs TTS and prepares the audio for delivery (either synchronously in the `/connect` response or queued for `/poll` proactive messages).
- After executing `speak`, the server returns a tool result to the LLM (e.g., `{"status": "audio queued for delivery", "text": "what was spoken"}`) so the model knows it has spoken.

This is a deliberate architectural inversion: the LLM must *choose* to produce user-facing output, and when it does, it is incentivized (via prompt + tool description) to make it natural spoken language.

This pattern is common in more advanced agent systems (treating final user output as an explicit action) and aligns well with voice-primary design.

---

## 3. Benefits of This Approach

- **Solves the formatting problem at the source**: The model is forced to generate TTS-friendly text (complete sentences, natural flow, verbalized structure like "There are three key points. First...").
- **Cleaner separation of concerns**: Internal agent work vs. what the user actually hears.
- **More agentic behavior**: The model can decide *when* to speak, *what* to say, and can interleave tool use with speech (e.g., "I'm checking that for you..." then later speak the result). It can also choose *not* to speak after completing internal work.
- **Fits existing architecture**:
  - Works for both reactive turns (`/connect`) and proactive messages (cron jobs + `/poll` + human detection via YOLO).
  - The ReAct loop already exists; we are extending the set of tools and changing the stopping condition.
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
- Continue the loop as long as the LLM returns `tool_calls`.
- When `speak` is among the tool calls:
  - Execute TTS on the `text` parameter (call Kokoro endpoint).
  - Prepare the resulting audio (base64 or reference) for return to client / queue for poll.
  - Append a tool result message to context, e.g.:
    ```json
    {"role": "tool", "tool_call_id": "...", "content": "{\"status\": \"audio queued for delivery to user\", \"text_spoken\": \"the exact text\"}"}
    ```
  - Decide on turn termination policy:
    - Alternative (more advanced): Allow the model to continue after speaking (e.g., speak progress, then do more work, then speak final answer). This requires accumulating spoken audio or playing sequentially.
- If the loop ends without any `speak` call having been made:
  - Fallback behavior: Either return a default audio ("I have completed the task but have nothing additional to say.") or log a warning and return no audio / a short confirmation. Do **not** fall back to TTS'ing arbitrary internal content.
- Update any logic that currently takes the final `content` and TTS's it. That path should be removed or deprecated for user-facing output.
- For **proactive messages** (cron jobs, `/inject`, scheduled prompts):
  - Run the same agent loop.
  - Any `speak` calls during that run populate a queue or pending audio store that `/poll` can serve (still gated by human detection on the client side via `/detect` + YOLO).
  - The `/poll` response format may need minor adjustment if it currently expects different fields.

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
- In the `/connect` success response, continue returning:
  - `transcription`
  - `text` (the exact text that was passed to `speak` — useful for clipboard, logs, and accessibility)
  - `audio` (base64 of the TTS output)
- For `/poll` proactive responses, ensure similar fields are returned when audio is available from a `speak` call.
- Update any logging or debug output to clearly distinguish internal tool activity from spoken content.

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

## 5. Key Design Decisions to Confirm with User (or Decide)

1. Should `speak` be strictly terminal for a turn, or should the model be allowed to continue the ReAct loop after speaking?
2. Policy for multiple `speak` calls in one turn (concatenate vs. last-wins vs. error)?
3. Fallback behavior when the agent loop completes with zero `speak` calls?
4. How much of the spoken text should be persisted in the long-term conversation context vs. only the tool result?
5. Any changes needed to the client-side audio playback or clipboard behavior?

---

## 6. Summary for Grok Build

We are changing the fundamental output contract of the agent:

**Before**: LLM reasons with tools → eventually emits text → server TTS's whatever text it emitted.

**After**: LLM reasons with tools (including the new `speak` tool) → only text explicitly passed to `speak(...)` ever becomes audio for the user.

This requires:
- New tool definition + detailed description
- Changes to the main agent loop control flow and stopping condition in `server.py`
- Significant system prompt updates with clear rules and examples
- Minor adjustments to response formatting, context management, and proactive message handling
- Thoughtful fallback and error handling

The result should be dramatically more listenable, natural voice responses while preserving (and enhancing) the existing powerful ReAct + tool execution capabilities.

This is a high-leverage refactor for a voice-primary system.

---

**End of Plan**

Pass this file to Grok Build along with access to the current `server/code/server.py` (and any prompt/config files) for implementation. Ask it to produce a clean diff or updated file + explanation of changes. 

After implementation, thorough listening tests + prompt iteration will be needed.
