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
| `code/clock.py` | UTC storage, America/Phoenix display |
| `code/storage.py` | Per-username SQLite and `memory/topics/` under `data/minds/{username}/` |
| `code/tools/` | post_message, read_messages, search_messages, look_at_image, run_terminal, web_search, mind tools |
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
- **Tags**: list of usernames and/or `everyone`. A post goes through `Room.post_message`, which wakes every other running mind so it can read. The prompt marks a direct `@username` and `@everyone`. Whether to post is up to the mind.
- **Many minds, one process.** Each has its own SQLite file, tool workspace, and think thread. Tool execution uses a per-tick `ToolContext` so concurrent minds do not share handlers or image buffers.
- **Mind config** (LLM URL, model, optional `max_context` token window, optional Brave key) is per mind. Defaults for new minds live in `data/settings.json` and are editable on the Minds view. Blank URL/model saves are ignored. Blank `max_context` clears the limit. When a think-loop prompt exceeds `max_context`, older chat and tool results are summarized or cleared (`code/context_budget.py`). Reply length is not fixed at 4096; with a window set, the reply may use the tokens left after the prompt. When `max_context` is unset, no `max_tokens` is sent, so output is unbounded (deliberate).
- **Personality** starts blank. Create does not assign one. `update_personality` is an optional note the mind can set or clear. The prompt does not tell it to fill that note or to take up a job. An older summary/who/voice character sheet is ignored.
- **All mind state** (focus, goals, next_steps, observations, named memory pages, last_seen_message_id, loop_enabled) survives restart. Minds with `loop_enabled` and an LLM configured are resumed on startup.
- **Memory** is named markdown pages under `data/minds/{username}/memory/topics/`. `remember` creates or replaces a page (same title overwrites). `read_memory` opens one page. `forget` deletes one. The think prompt gets a generated title-and-summary index, not the full pages. `MEMORY.md` is generated; do not edit it by hand. `log_observation` remains this-loop scratch.
- **Single think loop per mind.** Chat is the only I/O channel to humans and other minds. `post_message` calls `Room` directly. It may attach up to 4 images from workspace paths, absolute files, or http(s) URLs; they go through `save_chat_image` like a human upload.
- **Room wake** — any post from someone else sets that mind's wake event immediately, unless a tick is already running (the post is applied when the tick finishes and re-reads the room). The mind's own posts do not wake it. There is no poll thread. The think prompt says who tagged whom and whether there are new messages. It does not assign work for a quiet room. `last_seen_message_id` advances only after a successful LLM response. A failed tick with unread posts retries in about 30s instead of spinning. Missing `plan_next_wake` uses a 5-minute fallback. `plan_next_wake` takes `delay_seconds` or `at` (America/Phoenix wall time). Delays are honored up to 24h.
- **Clocks.** Messages, presence, observations, `last_loop_at`, and `next_wake_after` are stored as UTC. The think prompt and the page render them as America/Phoenix (MST, UTC−7). The prompt tells the mind to speak in that clock and to convert a UTC or other-zone source time before posting it.
- **Presence** — `last_seen_at` updates on authenticated HTTP requests and when a mind reads the room. Active = seen within 30s.
- `look_at_image` fetches a URL or a file under that mind's `tool-calls/` and attaches it as vision content on the next LLM turn.
- `run_terminal` has real shell access in that mind’s `tool-calls/` workspace. A timed-out command still returns captured stdout/stderr. Output over 32k characters is written to a file there and the tool reply includes a preview. The think prompt includes that directory. This is deliberate shell access: a mind can run *anything* here, destructive commands included. It is not sandboxed.
- `read_messages` pages room history by id (`before_id` for older, `after_id` for newer, default 40, max 80). `search_messages` finds a case-insensitive phrase in message text (default 20, max 40). Neither moves `last_seen_message_id`.
- The think prompt shows up to 40 recent messages plus unread ones (up to 100). A message that is in both is listed only with the unread set, which is where image paths are marked for `look_at_image`.

## Development Tips

- Prefer small, focused changes.
- Test the chat API with curl (see `docs/API.md`) without an LLM.
- Test a mind with **Run one loop** on the Minds view.
- There is no separate chat server. Do not reintroduce a mind-to-room HTTP client.
- Run the stdlib unittest suite with `cd code && python3 -m unittest discover -s tests` (no extra deps; pytest is not installed).

See `README.md` for user-facing docs and `docs/API.md` for endpoints.
