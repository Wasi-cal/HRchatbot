"""FastAPI app for the HR RAG chatbot serving layer.

Run locally:  ./run_api.sh   (or: uvicorn api.main:app --reload)
Docs:         http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    redis_cache.configure()
    _get_embedder()  # load BGE-M3 now rather than on the first request
    yield
    close_pool()


app = FastAPI(title="HR Chatbot API", lifespan=lifespan)

# SECURITY GAP - DEV ONLY: there is NO authentication or authorization
# anywhere in this API, and CORS is wide open. That means anyone who can
# reach the server can call /api/admin/* (upload or restrict documents,
# read all query logs, including user attributes and answers) and can
# supply arbitrary user_attributes to /api/chat. Before any real
# deployment: add auth (and derive user_attributes server-side from the
# session rather than trusting the client), put /api/admin behind an
# admin role, and restrict allow_origins to the real frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(admin.router)
app.include_router(voice.router)


@app.get("/health")
def health():
    return {"status": "ok"}
