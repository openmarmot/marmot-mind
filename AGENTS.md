# AGENTS.md

Minimal guidance for AI coding agents working on marmot-mind.

## Project Overview
Local voice-first AI agent with tool use.
- Client records audio (hotkey) and sends to server (or use the HTTP API directly for text/audio).
- Server: STT (whisper.cpp) → LLM (OpenAI-compatible + ReAct tools) → optional TTS.
- All conversation state, persistent memory, and context management lives on the **server**.
- Client is intentionally thin (audio + UI + background polling).

## Key Locations
- `client/code/client.py` — Main client logic (recording, hotkey via pynput, background proactive poller, playback).
- `server/code/server.py` — Almost all the interesting code (Flask routes, LLM loop, tools, context trimming, proactive queue, memory extraction).
- `docs/API.md` — API reference (endpoints, curl examples, proactive behavior).
- `server/code/config.json` (client creates `client/code/client_config.json` on first run).
- `agent-data/` — Tool working directory + persistent memory.txt.

## Running
```bash
# Server (first run prompts for whisper/LLM/TTS URLs)
cd server && ./start_server.sh

# Client (interactive hotkey mode)
cd client && ./start_client.sh

# (One-shot text via `-m` has been removed; replies now arrive via /poll.
#  Use the hotkey client or call the HTTP API directly for text input.)
```

Requires external services running:
- whisper.cpp (STT)
- OpenAI-compatible LLM server
- Optional: Kokoro-style TTS at `/v1/audio/speech`

## Architecture Notes (Important)
- **All AI-to-user output (including replies to direct input) goes through `speak` tool → `queue_proactive_message()` → client `/poll`**.
- Client polls `GET /poll` (with optional `?wait=`) when idle. Server queues via the speak tool (or `POST /inject` for manual).
- Spoken messages are appended to server `conversation_history` the moment they are delivered over `/poll`.
- Client maintains a small local buffer (`pending_proactive_queue`) for messages that arrive while busy. They play automatically when free (serialized by `playback_lock`).
- `/poll` request logs are deliberately suppressed on the server (see `QuietPollRequestHandler`) to keep the console clean.
- The `run_terminal` tool has real shell access on the host — be careful.
- Context lives only on the server. The client does not maintain history.

## Development Tips
- Most new features belong on the server.
- Use `POST /inject` heavily when testing proactive behavior.
- Important `log()` statements (timestamped; user turns, Marmot replies, proactive queuing/delivery, tool calls) are the primary way to observe behavior.
- Keep changes minimal and focused — this is a small, simple codebase.

See `README.md` for user-facing documentation and `docs/API.md` for endpoint details.