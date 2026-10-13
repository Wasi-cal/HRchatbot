"""POST /api/chat - the text channel: small talk, or a source lookup, else retrieve once + generate once."""
from typing import Annotated, Literal, Union

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from api.query_log import log_query
from channels import text as text_channel

router = APIRouter(prefix="/api/chat", tags=["chat"])


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    # Prior visible turns, oldest first. Untrusted (the client can edit it): the server keeps
    # the last conversation.history_messages, truncates each to conversation.history_max_chars
    # and drops anything else (channels/common.py clean_history). Optional.
    # Unknown fields (e.g. the retired user_attributes / pending_clarification) are ignored.
    history: list[HistoryMessage] = Field(default_factory=list, max_length=100)


class AnswerResponse(BaseModel):
    type: Literal["answer"] = "answer"
    text: str
    timing: dict


class RefusalResponse(BaseModel):
    """The low-confidence retrieval shortcut: a canned "I couldn't find that" reply
    (with a few real topics), no generation call made."""
    type: Literal["refusal_shortcut"] = "refusal_shortcut"
    text: str
    timing: dict


class SourceLookupResponse(BaseModel):
    """"Where is this from": a templated reply naming the documents the previous answer drew on, from a
    deterministic re-retrieval - no LLM call."""
    type: Literal["source_lookup"] = "source_lookup"
    text: str


class SmalltalkResponse(BaseModel):
    """A greeting / thanks / goodbye etc. answered by channels/text.py with no
    retrieval or LLM call."""
    type: Literal["smalltalk"] = "smalltalk"
    text: str


ChatResponse = Annotated[
    Union[AnswerResponse, RefusalResponse, SmalltalkResponse, SourceLookupResponse], Field(discriminator="type")
]


@router.post("", response_model=ChatResponse)
@router.post("/", response_model=ChatResponse, include_in_schema=False)
def chat(req: ChatRequest, background: BackgroundTasks):
    history = [m.model_dump() for m in req.history]

    # Small talk first: before retrieval, DB reads or any LLM work. (A bare "yes" after the bot's
    # question or offer is not small talk: it falls through so the offer is followed up.)
    category = text_channel.smalltalk_match(req.query, history)
    if category:
        text = text_channel.smalltalk_reply(category)
        background.add_task(
            log_query,
            query_text=req.query,
            response_type="smalltalk",
            answer_text=text,
            contextualized=False,
            advisory=False,
        )
        return SmalltalkResponse(text=text)

    # "Where is this from": 0 LLM calls, one re-retrieval of the previous answer's turn.
    source = text_channel.source_reply(req.query, history)
    if source:
        background.add_task(
            log_query,
            query_text=req.query,
            response_type="source_lookup",
            answer_text=source["text"],
            chunk_ids_used=source["chunk_ids"],
            timing=source["timing"],
            contextualized=False,
            advisory=False,
        )
        return SourceLookupResponse(text=source["text"])

    result = text_channel.answer(req.query, history=history)
    shortcut = result["shortcut"]
    background.add_task(
        log_query,
        query_text=req.query,
        response_type="refusal_shortcut" if shortcut else "answer",
        answer_text=result["answer"],
        shortcut_fired=shortcut,
        timing=result["timing"],
        chunk_ids_used=result.get("chunk_ids", []),
        prompt_version=result.get("prompt_version"),
        contextualized=result.get("contextualized"),
        advisory=result.get("advisory"),
    )
    cls = RefusalResponse if shortcut else AnswerResponse
    return cls(text=result["answer"], timing=result["timing"])
