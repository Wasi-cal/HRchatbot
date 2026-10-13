"""POST /api/chat - wraps ask_and_answer() with clarification resolution."""
from typing import Annotated, Literal, Union

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from access_control.ask_and_answer import CLARIFYING_QUESTIONS
from access_control.resolve_clarification import resolve_clarification
from channels import text as text_channel
from api.query_log import log_query

router = APIRouter(prefix="/api/chat", tags=["chat"])


class PendingClarification(BaseModel):
    original_query: str
    blocking_tag_type: str


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    user_attributes: dict | None = None
    pending_clarification: PendingClarification | None = None
    # Prior visible turns, oldest first. Untrusted (the client can edit it): the server keeps
    # the last conversation.history_messages, truncates each to conversation.history_max_chars
    # and drops anything else (channels/common.py clean_history). Optional.
    history: list[HistoryMessage] = Field(default_factory=list, max_length=100)


class AnswerResponse(BaseModel):
    type: Literal["answer"] = "answer"
    text: str
    timing: dict
    # The attributes the answer was produced with (incl. any resolved
    # from a clarification) - the client should send these back on the
    # next turn so it isn't re-asked.
    user_attributes: dict | None = None


class RefusalResponse(BaseModel):
    """The generate_answer() low-confidence shortcut path: a canned
    "I don't have that information" reply, no generation call made."""
    type: Literal["refusal_shortcut"] = "refusal_shortcut"
    text: str
    timing: dict
    # The attributes the answer was produced with (incl. any resolved
    # from a clarification) - the client should send these back on the
    # next turn so it isn't re-asked.
    user_attributes: dict | None = None


class SmalltalkResponse(BaseModel):
    """A greeting / thanks / goodbye etc. answered by channels/text.py with no
    retrieval or LLM call."""
    type: Literal["smalltalk"] = "smalltalk"
    text: str
    user_attributes: dict | None = None


class ClarificationResponse(BaseModel):
    type: Literal["needs_clarification"] = "needs_clarification"
    question: str
    blocking_tag_type: str
    original_query: str
    user_attributes: dict | None = None  # attributes known so far; send back with the reply


ChatResponse = Annotated[
    Union[AnswerResponse, RefusalResponse, ClarificationResponse, SmalltalkResponse], Field(discriminator="type")
]


@router.post("", response_model=ChatResponse)
@router.post("/", response_model=ChatResponse, include_in_schema=False)
def chat(req: ChatRequest, background: BackgroundTasks):
    # Small talk first: before retrieval, DB reads or any LLM work. Never
    # applies to a reply to a pending clarifying question.
    category = text_channel.smalltalk_match(req.query, after_clarification=req.pending_clarification is not None)
    if category:
        text = text_channel.smalltalk_reply(category)
        background.add_task(
            log_query,
            query_text=req.query,
            response_type="smalltalk",
            answer_text=text,
            user_attributes=req.user_attributes,
            contextualized=False,
            advisory=False,
        )
        return SmalltalkResponse(text=text, user_attributes=req.user_attributes)

    if req.pending_clarification:
        pc = req.pending_clarification
        attributes = resolve_clarification(
            pc.original_query, pc.blocking_tag_type, req.query, req.user_attributes
        )
        if attributes is None:
            question = CLARIFYING_QUESTIONS.get(
                pc.blocking_tag_type,
                f"Could you tell me your {pc.blocking_tag_type.replace('_', ' ')}?",
            )
            background.add_task(
                log_query,
                query_text=pc.original_query,
                response_type="clarification",
                user_attributes=req.user_attributes,
                blocking_tag_type=pc.blocking_tag_type,
            )
            return ClarificationResponse(
                question=f"Sorry, I didn't catch that. {question}",
                blocking_tag_type=pc.blocking_tag_type,
                original_query=pc.original_query,
                user_attributes=req.user_attributes,
            )
        query = pc.original_query
    else:
        query, attributes = req.query, req.user_attributes

    result = text_channel.answer(query, attributes, history=[m.model_dump() for m in req.history])

    if result["type"] == "needs_clarification":
        background.add_task(
            log_query,
            query_text=query,
            response_type="clarification",
            user_attributes=attributes,
            blocking_tag_type=result["blocking_tag_type"],
            contextualized=result.get("contextualized"),
            advisory=result.get("advisory"),
        )
        return ClarificationResponse(
            question=result["question"],
            blocking_tag_type=result["blocking_tag_type"],
            original_query=query,
            user_attributes=attributes,
        )

    shortcut = result["shortcut"]
    background.add_task(
        log_query,
        query_text=query,
        response_type="refusal_shortcut" if shortcut else "answer",
        answer_text=result["answer"],
        user_attributes=attributes,
        shortcut_fired=shortcut,
        timing=result["timing"],
        chunk_ids_used=result.get("chunk_ids", []),
        prompt_version=result.get("prompt_version"),
        contextualized=result.get("contextualized"),
        advisory=result.get("advisory"),
    )
    cls = RefusalResponse if shortcut else AnswerResponse
    return cls(text=result["answer"], timing=result["timing"], user_attributes=attributes)
