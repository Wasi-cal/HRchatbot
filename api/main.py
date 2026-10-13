"""FastAPI app for the HR RAG chatbot serving layer.

Run locally:  uvicorn api.main:app --reload   (apply the schema first: python3 -m vectorstore.setup_db)
Docs:         http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from api import admin, chat, voice  # noqa: E402  (after load_dotenv so env is set at import)
from cache import redis_cache  # noqa: E402
from config.runtime import seed_runtime_config  # noqa: E402
from vectorstore.db import close_pool, get_pool  # noqa: E402
from vectorstore.retrieve import _get_embedder  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One shared psycopg_pool pool for the whole app (vectorstore/db.py),
    # opened here and closed on shutdown; every module checks connections
    # out of it via get_pool().connection().
    pool = get_pool()
    pool.wait()
    seed_runtime_config()  # idempotent: fills any missing app_config/provider_configs rows
    from vectorstore.duplicates import backfill_fingerprints

    with pool.connection() as conn:
        backfill_fingerprints(conn)  # idempotent: only documents without a content_fingerprint
    redis_cache.configure()
    _get_embedder()  # load BGE-M3 now rather than on the first request
    yield
    close_pool()


app = FastAPI(title="HR Chatbot API", lifespan=lifespan)

# SECURITY GAP - DEV ONLY: there is NO authentication or authorization
# anywhere in this API, and CORS is wide open. That means anyone who can
# reach the server can call /api/admin/* (upload or restrict documents,
# read all query logs and answers). Before any real deployment: add auth,
# put /api/admin behind an admin role, and restrict allow_origins to the
# real frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(admin.router)
app.include_router(voice.router)


# Optional dummy frontend (plain HTML/CSS/JS, no build step): http://localhost:8000/app/
# The frontend/ folder is a LOCAL-ONLY dev tool (gitignored, not in the repo), so the mount only exists when the
# folder is present; the API itself never depends on it.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/app/")


@app.get("/health")
def health():
    return {"status": "ok"}
