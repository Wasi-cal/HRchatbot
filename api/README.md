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

**Deepgram's servers cannot reach `localhost`.** For local dev, expose the backend with a tunnel:

```bash
./run_api.sh                         # terminal 1
brew install ngrok && ngrok config add-authtoken <token>   # once
ngrok http 8000                      # terminal 2 -> note the https://xxxx.ngrok-free.app URL
echo 'PUBLIC_BASE_URL=https://xxxx.ngrok-free.app' >> .env
python3 -m api.voice_settings        # prints the Settings JSON with think.endpoint.url = <tunnel>/api/voice/v1/chat/completions
```

Send that JSON as the first message after `Welcome`; wait for `SettingsApplied`, then stream audio.
The URL changes on each ngrok restart (free tier), so regenerate the settings. Optionally set
`THINK_ENDPOINT_SECRET`: the builder adds it as the `authorization: Bearer ...` header Deepgram sends
back, and the endpoint rejects calls without it. Model/voice/greeting are env-configurable (see `.env.example`).

Test without Deepgram or audio: `python3 -m eval.voice_think_test` (simulates Deepgram's calls with a full message history).

How it works: the endpoint is stateless and re-derives everything from the history. A clarifying question is
spoken as `"Quick check before I answer - <question>"` using a fixed wording per attribute (no hidden token,
since TTS would read it out); the next user turn is then treated as the reply. Country defaults to India.
The request body Deepgram sends isn't fully documented, so the endpoint reads only `messages` and
should be checked against a real session. Voice turns are logged with `source = 'voice'`
(`/api/admin/usage/summary?source=voice` filters on it).
