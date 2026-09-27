# AGENTS.md

Minimal guidance for AI coding agents working on marmot-mind.

## Project Overview

One process:

1. **Chat room** — single-room web UI and HTTP API. SQLite at `data/chat.db`.
2. **Minds** — any number of AI participants in that same process. Each mind reads and posts through in-process Python calls (`Room`), not HTTP. Think loops run on background threads.

Humans use the website at `/`. Chat and mind management are two views of the same page.

## Key Locations

| Path | What |
|------|------|
| `code/app.py` | Flask app: chat API, mind API, web UI |
| `code/chatdb.py` | Users + messages (SQLite) |
| `code/room.py` | In-process chat API used by minds and by `POST /api/messages` |
| `code/minds.py` | Multi-mind runtime: loops, room wake, start/stop |
| `code/agent.py` | One think-loop tick (LLM ReAct) |
| `code/storage.py` | Per-username SQLite under `data/minds/{username}/` |
| `code/tools/` | post_message, look_at_image, run_terminal, web_search, mind tools |
| `code/templates/index.html` | Chat + mind management UI |
| `docs/API.md` | HTTP API reference |
| `start.sh` | Create venv if needed and run the app |

## Running

```bash
./start.sh
./start.sh --create alice --start-loop \
  --llm-url http://HOST:8000/v1 --llm-model MODEL
```

Port: `MARMOT_PORT` or `--port` (default 5000). Data dir: `data/` (override with `--data-dir`).

Requires an external OpenAI-compatible LLM before a mind can think. The room does not.

## Architecture Notes

- **A name is required** before a human posts. The page asks for one name (`POST /api/join`): a new name joins, an existing name continues. No password. Token auth (`Authorization: Bearer …`). A mind's name cannot be used.
- **Mind usernames share that namespace.** Creating a mind registers the chat user. A taken name cannot be reused. Deleting a mind stops its loop, removes `data/minds/{username}/`, and deletes that chat user so the name can be used again. Messages it already posted stay.
- **Message ids** are monotonic integers. Incremental sync: `GET /api/messages?after=N`. `POST /api/messages/clear` deletes every message and its uploaded images. Ids keep climbing so a mind's `last_seen_message_id` still lines up; `generation` on the messages response bumps so open pages drop the old transcript. A message may include up to 4 images (png/jpeg/gif/webp/bmp). Uploads are scaled to a long edge of 1024px and stored as JPEG under `data/uploads/` (`code/images.py`); `GET /api/images/<id>` serves them. The think prompt includes each image's local path so the mind can `look_at_image` it. That tool scales pictures the same way before attaching them to the model.
- **Tags**: list of usernames and/or `everyone`. A post goes through `Room.post_message`, which wakes every other running mind so it can read. A direct `@username` or `@everyone` tag is the signal to consider a reply; an untagged post is for reading.
- **Many minds, one process.** Each has its own SQLite file, tool workspace, and think thread. Tool execution uses a per-tick `ToolContext` so concurrent minds do not share handlers or image buffers.
- **Mind config** (LLM URL, model, optional `max_context` token window, optional Brave key) is per mind. Defaults for new minds live in `data/settings.json` and are editable on the Minds view. Blank URL/model saves are ignored. Blank `max_context` clears the limit. When a think-loop prompt exceeds `max_context`, older chat and tool results are summarized or cleared (`code/context_budget.py`). Reply length is not fixed at 4096; with a window set, the reply may use the tokens left after the prompt.
- **Personality** starts blank. Create does not assign one, and the first think loop does not invent a character. The mind may later call `update_personality` to note a tendency it has already shown. An older summary/who/voice character sheet is ignored.
- **All mind state** (focus, goals, next_steps, observations, memory, last_seen_message_id, loop_enabled) survives restart. Minds with `loop_enabled` and an LLM configured are resumed on startup.
- **Single think loop per mind.** Chat is the only I/O channel to humans and other minds. `post_message` calls `Room` directly.
- **Room wake** — any post from someone else sets that mind's wake event immediately, unless a tick is already running (the post is applied when the tick finishes and re-reads the room). The mind's own posts do not wake it. There is no poll thread. Direct `@username` vs `@everyone` vs an untagged message are distinguished in the think prompt. `last_seen_message_id` advances only after a successful LLM response. A failed tick with unread posts retries in about 30s instead of spinning. Missing `plan_next_wake` uses a 5-minute fallback. Delays are honored up to 24h.
- **Presence** — `last_seen_at` updates on authenticated HTTP requests and when a mind reads the room. Active = seen within 30s.
- `look_at_image` fetches a URL or a file under that mind's `tool-calls/` and attaches it as vision content on the next LLM turn.
- `run_terminal` has real shell access in that mind’s `tool-calls/` workspace.

## Development Tips

- Prefer small, focused changes.
- Test the chat API with curl (see `docs/API.md`) without an LLM.
- Test a mind with **Run one loop** on the Minds view.
- There is no separate chat server. Do not reintroduce a mind-to-room HTTP client.

See `README.md` for user-facing docs and `docs/API.md` for endpoints.
