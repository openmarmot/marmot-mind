# AGENTS.md

Minimal guidance for AI coding agents working on marmot-mind.

## Project Overview

Local multi-participant setup:

1. **Chat server** (`server/`) — single-room Discord/Slack-style chat. Web UI at `/`. SQLite persistence.
2. **Mind** (`mind/`) — fully independent AI client process. Connects to the chat server, posts/reads messages, runs one self-scheduling think loop. Status + config UI on a random free port.

Humans interact via the chat website. There is no voice client.

## Key Locations

| Path | What |
|------|------|
| `server/code/server.py` | Flask chat API + serves web UI |
| `server/code/db.py` | Users + messages (SQLite) |
| `server/code/templates/index.html` | Chat web UI |
| `server/data/chat.db` | Server DB (created at runtime) |
| `mind/code/mind.py` | Mind process entry: status server + loop control |
| `mind/code/agent.py` | Think loop + LLM ReAct tool use |
| `mind/code/chat_client.py` | HTTP client for chat API |
| `mind/code/storage.py` | Per-username SQLite under `mind/data/{username}/` |
| `mind/code/tools/` | post_message, run_terminal, web_search, mind tools |
| `docs/API.md` | Chat API reference |

## Running

```bash
# Chat server (default port 5000, override with MARMOT_PORT)
cd server && ./start_server.sh

# Mind (status UI on random port; configure via web or CLI flags)
cd mind && ./start_mind.sh
cd mind && ./start_mind.sh --create alice --start-loop \
  --chat-server http://127.0.0.1:5000 \
  --llm-url http://HOST:8000/v1 --llm-model MODEL
```

Requires an external OpenAI-compatible LLM for minds. Chat server needs no LLM.

## Architecture Notes

- **Signup required** before posting. Username identifies all messages. Token auth (`Authorization: Bearer …`).
- **Message ids** are monotonic integers. Incremental sync: `GET /api/messages?after=N`.
- **Tags**: list of usernames and/or `everyone`. Minds treat tags on their username or everyone as directed.
- **One mind process = one username.** Concurrent minds = multiple `mind.py` processes (separate data dirs + ports).
- **Mind config** (chat URL, LLM URL/model) is per-username SQLite + editable on the mind status page.
- **Personality** is generated on create and persisted.
- **All mind state** (focus, goals, next_steps, observations, memory, last_seen_message_id, loop_enabled) survives restart.
- **Single think loop** — no separate user-response vs background agents. Chat is the only I/O channel to humans/other minds.
- Mind communicates **only** via `post_message` tool (not TTS/speak).
- `run_terminal` has real shell access in that mind’s `tool-calls/` workspace — be careful.

## Development Tips

- Prefer small, focused changes.
- Test chat API with curl (see `docs/API.md`) without needing an LLM.
- Test minds with `Run one loop now` on the status page.
- Keep the chat server dumb (transport + storage). Intelligence lives in `mind/`.

See `README.md` for user-facing docs and `docs/API.md` for endpoints.
