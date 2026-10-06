"""POST /api/chat - wraps ask_and_answer() with clarification resolution."""
from typing import Annotated, Literal, Union

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from access_control.ask_and_answer import CLARIFYING_QUESTIONS, ask_and_answer
from access_control.resolve_clarification import resolve_clarification
from api.query_log import log_query

router = APIRouter(prefix="/api/chat", tags=["chat"])


class PendingClarification(BaseModel):
    original_query: str
    blocking_tag_type: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    user_attributes: dict | None = None
    pending_clarification: PendingClarification | None = None


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


class ClarificationResponse(BaseModel):
    type: Literal["needs_clarification"] = "needs_clarification"
    question: str
    blocking_tag_type: str
    original_query: str
    user_attributes: dict | None = None  # attributes known so far; send back with the reply


ChatResponse = Annotated[
    Union[AnswerResponse, RefusalResponse, ClarificationResponse], Field(discriminator="type")
]


@router.post("", response_model=ChatResponse)
@router.post("/", response_model=ChatResponse, include_in_schema=False)
def chat(req: ChatRequest, background: BackgroundTasks):
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

    result = ask_and_answer(query, attributes)

    if result["type"] == "needs_clarification":
        background.add_task(
            log_query,
            query_text=query,
            response_type="clarification",
            user_attributes=attributes,
            blocking_tag_type=result["blocking_tag_type"],
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
    )
    cls = RefusalResponse if shortcut else AnswerResponse
    return cls(text=result["answer"], timing=result["timing"], user_attributes=attributes)
