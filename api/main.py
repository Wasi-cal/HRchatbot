"""FastAPI app for the HR RAG chatbot serving layer.

Run locally:  ./run_api.sh   (or: uvicorn api.main:app --reload)
Docs:         http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from api import admin, chat  # noqa: E402  (after load_dotenv so env is set at import)
from vectorstore.db import close_pool  # noqa: E402
from vectorstore.retrieve import _get_embedder  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
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


@app.get("/health")
def health():
    return {"status": "ok"}
