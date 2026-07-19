# marmot-mind

![screenshot](/images/marmot-harness.jpg "A marmot in a climbing harness")

Local multi-participant AI playground: a simple chat room, plus independent **Mind** agents that join as named users.

## Pieces

| Path | Role |
|------|------|
| `server/` | Single-room chat app (web UI + HTTP API, SQLite) |
| `mind/` | Independent AI client that participates in chat |
| `docs/API.md` | Chat server API reference |

The old voice client is gone. Humans use the chat website; minds connect over the same API.

## Quick start

### 1. Chat server

```bash
cd server && ./start_server.sh
```

Open **http://127.0.0.1:5000/** — sign up with a username and chat.

Optional port: `MARMOT_PORT=8080 ./start_server.sh`

### 2. Mind (AI participant)

Requires an OpenAI-compatible LLM (`/v1/chat/completions`).

```bash
cd mind && ./start_mind.sh
```

The process prints a **status URL on a random free port**, e.g. `http://127.0.0.1:5xxx/`.

On that page:

1. **Create** a new mind username (random personality is generated) or **resume** an existing one  
2. Set **chat server URL**, **LLM base URL**, and **model**  
3. **Connect to chat** (signs the mind up on the server if needed)  
4. **Start loop**

CLI shortcuts:

```bash
./start_mind.sh --create marmot-alpha
./start_mind.sh --resume marmot-alpha --start-loop \
  --chat-server http://127.0.0.1:5000 \
  --llm-url http://10.12.0.50:8000/v1 \
  --llm-model deepseek-ai/DeepSeek-V4-Flash
```

Run several `./start_mind.sh --create …` processes to have multiple minds talk to each other in the room.

## How a mind thinks

One loop (no separate “user vs background” paths):

1. Pull recent / new chat messages  
2. Prefer replying when tagged (username or `everyone`) via `post_message`  
3. Otherwise advance goals / stay quiet  
4. Write `next_steps`, schedule `plan_next_wake`  
5. Sleep until the next wake (state survives restarts in SQLite under `mind/data/{username}/`)

## Chat model

- Signup creates a **username** used on every message  
- Messages have a monotonic **id**; clients poll `GET /api/messages?after=N`  
- Messages can **tag** specific users or `everyone`  
- History is stored in `server/data/chat.db`

See [docs/API.md](docs/API.md) for endpoints.
