# Tessa web (Next.js)

Chat + voice UI for the HR policy assistant. The browser talks only to this Next app; `/api/*` is proxied to the FastAPI backend (`API_URL`, default `http://localhost:8000`, see `next.config.mjs`). Voice connects from the browser straight to Deepgram using a short-lived token from `/api/voice/token`.

```
docker compose up -d postgres redis   # from the repo root
./run_api.sh                          # backend on :8000
cd web && npm install && npm run dev  # UI on :3000
```

Voice also needs `PUBLIC_BASE_URL` in the backend `.env` to be a tunnel Deepgram's cloud can reach.
Conversation history is kept in the browser (localStorage); there is no per-user history API yet.
