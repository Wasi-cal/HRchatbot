# HR chatbot API

```bash
python3 -m vectorstore.setup_db   # idempotent; creates query_logs
./run_api.sh                      # http://localhost:8000/docs
python3 -m eval.api_smoke_test    # end-to-end check (server running)
```

```bash
B=http://localhost:8000
# fresh query
curl -s $B/api/chat -H 'content-type: application/json' \
  -d '{"query":"How many days of leave do I get?"}'
# -> {"type":"needs_clarification","blocking_tag_type":"country","original_query":...}
# resolve it (send back original_query + blocking_tag_type, and any user_attributes returned)
curl -s $B/api/chat -H 'content-type: application/json' \
  -d '{"query":"I work in India","pending_clarification":{"original_query":"How many days of leave do I get?","blocking_tag_type":"country"}}'
# upload / list / restrict
curl -s -F file=@"hr docs/Travel Policy_Ver1.0.pdf" $B/api/admin/documents/upload
curl -s $B/api/admin/documents
curl -s -X PATCH $B/api/admin/documents/<id>/restrict -H 'content-type: application/json' -d '{"is_restricted":true}'
# usage (days=0 means all time)
curl -s "$B/api/admin/usage/summary?days=7"
curl -s "$B/api/admin/usage/recent?limit=50"
```

Response `type` is `answer`, `refusal_shortcut` (low-confidence canned reply), or
`needs_clarification`. All include `user_attributes` - send them back on the next turn.
There is no auth (see the security note in `api/main.py`).

## Voice (Deepgram Voice Agent, bring-your-own-LLM)

The browser connects straight to Deepgram (`wss://agent.deepgram.com/v1/agent/converse`) and
sends a `Settings` message. Deepgram's *cloud* then calls our think endpoint,
`POST /api/voice/v1/chat/completions` (OpenAI Chat Completions-style, SSE streamed), each turn.

**Deepgram's servers cannot reach `localhost`.** For local dev, expose the backend with a tunnel script:

```bash
./run_api.sh                          # terminal 1 (FastAPI, port $PORT, default 8000)
python3 scripts/start_tunnel.py       # terminal 2 - leave it open
```

It starts a **cloudflared** quick tunnel (default; no account; `brew install cloudflared`), captures the
`https://*.trycloudflare.com` URL, writes it to `.env` as `PUBLIC_BASE_URL` (the variable `api/voice_settings.py`
reads), prints the regenerated Settings JSON, and stops the tunnel on Ctrl+C. If the binary is missing it prints the
install command. Switch providers with `--provider ngrok` (needs `ngrok config add-authtoken <token>` once) or
`--provider localtunnel` (npm; may show an interstitial page that can block Deepgram's calls). `--port N` overrides the
port; `--verbose` echoes the tunnel's own logs. Start `./run_api.sh` on the same port - the script warns if nothing is answering.

**The URL changes every time the tunnel starts**, so always restart via this script - not the tunnel binary directly -
or `.env` and the Settings go stale. Re-send the new Settings to Deepgram on your next session.

Send that JSON as the first message after `Welcome`; wait for `SettingsApplied`, then stream audio.
Optionally set
`THINK_ENDPOINT_SECRET`: the builder adds it as the `authorization: Bearer ...` header Deepgram sends
back, and the endpoint rejects calls without it. Model/voice/greeting live in `provider_configs` (see the runtime config section).

Test without Deepgram or audio: `python3 -m eval.voice_think_test` (simulates Deepgram's calls with a full message history).

How it works: the endpoint is stateless and re-derives everything from the history. A clarifying question is
spoken as `"Quick check before I answer - <question>"` using a fixed wording per attribute (no hidden token,
since TTS would read it out); the next user turn is then treated as the reply. Country defaults to India.
The request body Deepgram sends isn't fully documented, so the endpoint reads only `messages` and
should be checked against a real session. Voice turns are logged with `source = 'voice'`
(`/api/admin/usage/summary?source=voice` filters on it).

## Runtime config, cache, pooling

- `docker compose up -d` starts Postgres (5434) and Redis (6380). Redis is a cache only; if it's down, reads fall back to Postgres.
- `GET /api/admin/config` lists `app_config` rows and the active generation/stt/tts `provider_configs`.
  `PATCH /api/admin/config/{key}` with `{"value": ...}` updates one setting (validated) and refreshes the cache.
  Keys: `retrieval.top_k|candidate_n|rrf_k`, `generation.model|confidence_threshold`, `stt.model`, `tts.voice|greeting`.
  Other servers/workers see DB-side edits within 30s (cache TTL).
- The env vars `GENERATION_MODEL_NAME`, `GENERATION_CONFIDENCE_THRESHOLD`, `DEEPGRAM_LISTEN_MODEL`, `DEEPGRAM_TTS_MODEL`,
  `VOICE_GREETING` only seed the DB the first time. API keys, `THINK_ENDPOINT_SECRET` and DB credentials stay env-only.
- Tests: `python3 -m eval.config_test` (API running), `python3 -m eval.run_regression`.

## Live voice test (real Deepgram session)

Extra deps: `uv sync --group voice-test` (installs `deepgram-sdk` and `sounddevice`). Microphone/speaker access
uses PortAudio: the macOS `sounddevice` wheel bundles it (just allow the terminal under System Settings > Privacy >
Microphone); on Linux install it first (`sudo apt install libportaudio2`). `DEEPGRAM_API_KEY` must be in `.env`
(Member role or higher is enough to mint tokens).

```bash
docker compose up -d                                           # postgres + redis
VOICE_DEBUG_LOG_RAW_REQUEST=1 ./run_api.sh                     # terminal 1 (debug flag optional, see below)
python3 scripts/start_tunnel.py                                # terminal 2
python3 scripts/voice_live_test.py                             # terminal 3: speak, Ctrl+C to stop
python3 scripts/voice_live_test.py --text "How many days of leave do I get?"   # no mic: typed turn(s), repeatable
```

The client fetches `GET /api/voice/settings` (rebuilt on every call from the latest `.env` and `provider_configs`, so a
fresh tunnel URL is picked up without restarting the server), gets a short-lived token from `POST /api/voice/token` (falls
back to the raw API key with a loud warning if that fails), sends `Settings`, then streams mic audio and plays the agent's
audio. Every event (`SettingsApplied`, `UserStartedSpeaking`, `ConversationText`, `Error`, ...) is printed with a timestamp.
Speaking over the agent drops its queued audio (barge-in). `--text` mode streams silence, because Deepgram closes sessions
that receive no audio.

**`VOICE_DEBUG_LOG_RAW_REQUEST=1`** (debug only, off by default - the log contains users' transcribed speech) prints the full
raw think-endpoint request body, a one-line summary (top-level keys, whether `stream` was present, message roles, tools) and
header *names*; `VOICE_DEBUG_LOG_FILE=path` also appends it to a file.

What it showed in the first live session (Oct 2026):
- Body is exactly `{"model", "messages", "stream": true, "stream_options": {"include_usage": true}}`: **`stream` is sent
  explicitly as `true`**, no `tools`/system message (we configure no prompt or functions).
- `messages` is the full history, starting with a synthetic `{"role":"user","content":"-"}` followed by the greeting, then the
  real turns. Our spoken clarifying questions come back verbatim, so history-based clarification detection works.
- Deepgram sends our `authorization` header from `think.endpoint.headers`.

Token notes: `ttl_seconds` must be an **integer** (a float such as `60.0` is silently ignored and you get 30s). An idle session
(`--no-mic --seconds 110`) stayed connected well past a 60s token's expiry, so a token appears to be needed only at
handshake time; a long mic conversation hasn't been tried.
