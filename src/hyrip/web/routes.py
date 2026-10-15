"""The /api router: articles, sessions, answers, tutor SSE, history, stats, models."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..config import Config
from ..errors import ConfigError, DataError, GenerationError, HyripError, LlmError, StateError
from ..models.article import Article
from ..models.grading import UserAnswer
from ..service.session import SessionService
from ..store.repo import load_answers, load_article, load_session
from .schemas import AnswerIn, ArticleIn, FollowupIn, SessionIn
from .sse import sse_headers, stream_sse

router = APIRouter(prefix="/api")

_ERROR_STATUS = {
    StateError: 409,
    LlmError: 502,
    GenerationError: 502,
    DataError: 400,
    ConfigError: 422,
}


def _http_error(exc: HyripError) -> HTTPException:
    status = _ERROR_STATUS.get(type(exc), 500)
    return HTTPException(status_code=status, detail=exc.user_message())


async def _call(fn, *args, **kwargs):
    """Run a sync service call off the event loop, mapping HyripError to HTTP errors."""
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except HyripError as exc:
        raise _http_error(exc) from exc


def _service(request: Request) -> SessionService:
    return request.app.state.service


def _config(request: Request) -> Config:
    return request.app.state.config


# ---- articles ------------------------------------------------------------------------


def article_dto(article_id: int, article: Article) -> dict:
    return {
        "id": article_id,
        "title": article.title,
        "source": article.source.value,
        "source_url": article.source_url,
        "word_count": article.word_count,
        "paragraphs": article.paragraphs,
        "paragraph_labels": article.paragraph_labels,
    }


def _load_article(service: SessionService, article_id: int) -> Article:
    return load_article(service.store, article_id)


@router.post("/articles", status_code=201)
async def create_article(payload: ArticleIn, request: Request) -> dict:
    service = _service(request)
    if payload.source == "text":
        article_id = await _call(
            service.create_article_from_text, payload.text, title=payload.title
        )
    else:
        article_id = await _call(service.create_article_from_url, payload.url)
    article = await _call(_load_article, service, article_id)
    return article_dto(article_id, article)


@router.get("/articles/{article_id}")
async def get_article(article_id: int, request: Request) -> dict:
    service = _service(request)
    try:
        article = await asyncio.to_thread(_load_article, service, article_id)
    except StateError:
        raise HTTPException(status_code=404, detail="no such article") from None
    except HyripError as exc:
        raise _http_error(exc) from exc
    return article_dto(article_id, article)


# ---- generation + session lifecycle ----------------------------------------------------


async def _generate_background(service: SessionService, session_id: int, limiter) -> None:
    """One in-flight generation at a time; failures stay visible via the failed status."""
    async with limiter.semaphore:
        try:
            await asyncio.to_thread(service.generate_for_session, session_id)
        except Exception:
            return


@router.post("/sessions", status_code=202)
async def create_session(payload: SessionIn, request: Request, background: BackgroundTasks) -> dict:
    service = _service(request)
    session_id = await _call(
        service.create_session,
        payload.article_id,
        mode=payload.mode,
        n_questions=payload.n_questions,
        model=payload.model,
        claims=payload.claims,
        time_budget_min=payload.time_budget_min,
    )
    background.add_task(_generate_background, service, session_id, request.app.state.limiter)
    snapshot = await _call(service.snapshot, session_id)
    return {"session_id": session_id, "status": snapshot.status}


@router.get("/sessions/{session_id}")
async def get_session(session_id: int, request: Request) -> dict:
    service = _service(request)
    try:
        snapshot = await asyncio.to_thread(service.snapshot, session_id)
    except StateError:
        raise HTTPException(status_code=404, detail="no such session") from None
    except HyripError as exc:
        raise _http_error(exc) from exc
    payload = snapshot.__dict__.copy()
    payload["session_id"] = snapshot.id
    if snapshot.status in ("reading", "active"):
        payload["view"] = await _call(service.view, session_id)
    if snapshot.status == "report":
        payload["report"] = (await _call(service.report_of, session_id)).model_dump()
    return payload


@router.post("/sessions/{session_id}/start")
async def start_session(session_id: int, request: Request) -> dict:
    await _call(_service(request).start, session_id)
    return {"ok": True}


@router.post("/sessions/{session_id}/abandon")
async def abandon_session(session_id: int, request: Request) -> dict:
    await _call(_service(request).abandon, session_id)
    return {"ok": True}


# ---- answers ----------------------------------------------------------------------------


@router.post("/sessions/{session_id}/answers")
async def post_answer(session_id: int, payload: AnswerIn, request: Request) -> dict:
    service = _service(request)
    raw = None if payload.skipped else payload.raw
    graded = await _call(
        service.record_answer,
        session_id,
        UserAnswer(number=payload.number, raw=raw, skipped=payload.skipped),
    )
    return {"recorded": True, "verdict": graded.model_dump() if graded is not None else None}


@router.get("/sessions/{session_id}/answers")
async def get_answers(session_id: int, request: Request) -> dict:
    service = _service(request)
    tally = await _call(service.tally, session_id)

    def rows() -> list[dict]:
        row = load_session(service.store, session_id)
        if row.paper_id is None:
            return []
        return [
            {
                "number": answer.number,
                "raw": answer.raw,
                "skipped": answer.skipped,
                "verdict": answer.verdict,
                "graded_by": answer.graded_by,
            }
            for answer in load_answers(service.store, session_id)
        ]

    return {"tally": tally.model_dump(), "answers": await _call(rows)}


# ---- finishing ------------------------------------------------------------------------


@router.post("/sessions/{session_id}/finish")
async def finish_session(session_id: int, request: Request) -> dict:
    report = await _call(_service(request).finish, session_id)
    return report.model_dump()


@router.get("/sessions/{session_id}/report")
async def get_report(session_id: int, request: Request) -> dict:
    report = await _call(_service(request).report_of, session_id)
    return report.model_dump()


# ---- tutor SSE -------------------------------------------------------------------------


@router.get("/sessions/{session_id}/tutor-feedback/{number}")
async def tutor_feedback(session_id: int, number: int, request: Request):
    service = _service(request)
    graded = await _call(service.answer_verdict, session_id, number)
    if graded is None:
        raise HTTPException(
            status_code=409, detail="this answer has no verdict yet (tutor sessions only)"
        )

    async def event_stream():
        async for frame in stream_sse(
            lambda: service.tutor_feedback_stream(session_id, number, graded)
        ):
            yield frame

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=sse_headers())


@router.post("/sessions/{session_id}/tutor-followup")
async def tutor_followup(session_id: int, payload: FollowupIn, request: Request):
    service = _service(request)

    async def event_stream():
        async for frame in stream_sse(
            lambda: service.tutor_followup_stream(session_id, payload.number, payload.text)
        ):
            yield frame

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=sse_headers())


# ---- catalog + history + stats ---------------------------------------------------------


@router.get("/models")
async def list_models(request: Request) -> list[dict]:
    try:
        models = await asyncio.to_thread(_service(request).list_models)
    except LlmError as exc:
        raise HTTPException(status_code=502, detail=exc.user_message()) from exc
    from dataclasses import asdict

    return [asdict(m) for m in models]


@router.get("/history")
async def history(request: Request):
    return await _call(_service(request).list_history)


@router.get("/stats")
async def stats(request: Request):
    return (await _call(_service(request).stats)).model_dump()


@router.get("/health")
async def health(request: Request) -> dict:
    config = _config(request)
    try:
        models = await asyncio.to_thread(_service(request).list_models)
    except Exception:
        models = []
    return {
        "status": "ok" if models else "degraded",
        "api_key_set": bool(config.api_key),
        "catalog": len(models),
    }
