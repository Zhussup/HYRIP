"""Web API tests on the ASGI transport with an injected FakeLLM."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from hyrip.llm.client import ChatResult
from hyrip.service.session import SessionService
from hyrip.store.repo import save_article
from hyrip.web.app import build_app


@pytest.fixture
def app(store, config, fake_llm):
    config_obj = config
    service = SessionService(store, fake_llm, config_obj)
    return build_app(service, config_obj)


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client_value:
        yield client_value


@pytest.fixture
def saved_article(store, article) -> int:
    return save_article(store, article)


def queue_paper(fake_llm, raw_paper):
    fake_llm.responses.append(
        ChatResult(content=raw_paper.model_dump_json(), finish_reason="stop", latency_s=0.1)
    )


async def make_session(client, fake_llm, raw_paper, saved_article, **extra):
    queue_paper(fake_llm, raw_paper)
    response = await client.post(
        "/api/sessions",
        json={"article_id": saved_article, "mode": "exam", **extra},
    )
    assert response.status_code == 202, response.text
    body = response.json()
    session_id = body["session_id"]
    # the 202 response arrived before generation; the poll must reach a settled state
    for _ in range(50):
        state = (await client.get(f"/api/sessions/{session_id}")).json()
        if state["status"] != "generating":
            break
        await asyncio.sleep(0)
    assert state["status"] == "reading", state
    return session_id


async def test_article_from_paste_returns_dto(client, config):
    text = (
        "Title\n\n"
        + " ".join("Bees live in cities." for _ in range(40))
        + "\n\n"
        + " ".join("More paragraph text here." for _ in range(40))
    )
    response = await client.post("/api/articles", json={"source": "text", "text": text})
    assert response.status_code == 201
    dto = response.json()
    assert dto["id"] >= 1
    assert set(dto) >= {"title", "source", "word_count", "paragraphs", "paragraph_labels"}
    assert dto["paragraph_labels"][0] == "A"


async def test_article_empty_paste_rejected(client):
    response = await client.post("/api/articles", json={"source": "text", "text": "  "})
    assert response.status_code == 422  # pydantic validator


async def test_article_missing_404(client):
    assert (await client.get("/api/articles/999")).status_code == 404


async def test_full_exam_flow(client, fake_llm, raw_paper, saved_article):
    session_id = await make_session(client, fake_llm, raw_paper, saved_article)

    # answer keys are stripped while active
    view = (await client.get(f"/api/sessions/{session_id}")).json()["view"]
    assert "answer_key" not in json.dumps(view)
    assert (await client.post(f"/api/sessions/{session_id}/start")).status_code == 200

    post = await client.post(f"/api/sessions/{session_id}/answers", json={"number": 1, "raw": "A"})
    assert post.json() == {"recorded": True, "verdict": None}  # exam hides verdicts

    answers = (await client.get(f"/api/sessions/{session_id}/answers")).json()
    assert answers["answers"][0]["number"] == 1

    marker = json.dumps({"results": [{"number": 10, "verdict": "correct"}]})
    fake_llm.responses.append(ChatResult(content=marker, finish_reason="stop"))
    finished = await client.post(f"/api/sessions/{session_id}/finish")
    assert finished.status_code == 200
    report = finished.json()
    assert report["score"]["total"] == 12
    assert (await client.get(f"/api/sessions/{session_id}/report")).json()["score"] == report[
        "score"
    ]


async def test_generation_failure_surfaces_in_poll(client, fake_llm, saved_article):
    fake_llm.responses.append(ChatResult(content="junk", finish_reason="stop"))
    created = (await client.post("/api/sessions", json={"article_id": saved_article})).json()
    session_id = created["session_id"]
    for _ in range(50):
        state = (await client.get(f"/api/sessions/{session_id}")).json()
        if state["status"] != "generating":
            break
        await asyncio.sleep(0)
    assert state["status"] == "failed"


async def test_answers_out_of_range_409(client, fake_llm, raw_paper, saved_article):
    session_id = await make_session(client, fake_llm, raw_paper, saved_article)
    await client.post(f"/api/sessions/{session_id}/start")
    response = await client.post(
        f"/api/sessions/{session_id}/answers", json={"number": 99, "raw": "A"}
    )
    assert response.status_code == 409


async def test_finish_before_start_409(client, fake_llm, raw_paper, saved_article):
    session_id = await make_session(client, fake_llm, raw_paper, saved_article)
    response = await client.post(f"/api/sessions/{session_id}/finish")
    assert response.status_code == 409


async def test_tutor_streams_over_sse(client, fake_llm, raw_paper, saved_article):
    session_id = await make_session(client, fake_llm, raw_paper, saved_article, mode="tutor")
    await client.post(f"/api/sessions/{session_id}/start")
    answer = await client.post(
        f"/api/sessions/{session_id}/answers", json={"number": 1, "raw": "A"}
    )
    verdict = answer.json()["verdict"]
    assert verdict is not None and verdict["verdict"] == "correct"

    fake_llm.streams.append(["Good read: ", "Q1 checks the opening claim."])
    frames = ""
    async with client.stream("GET", f"/api/sessions/{session_id}/tutor-feedback/1") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            frames += line + "\n"
    assert "Good read" in frames and '"done": true' in frames

    fake_llm.streams.append(["It is wrong because ", "the passage is silent."])
    async with client.stream(
        "POST",
        f"/api/sessions/{session_id}/tutor-followup",
        json={"number": 1, "text": "why?"},
    ) as response:
        body = ""
        async for chunk in response.aiter_text():
            body += chunk
    assert "passage is silent" in body and '"done": true' in body


async def test_tutor_feedback_without_verdict_409(client, fake_llm, raw_paper, saved_article):
    session_id = await make_session(client, fake_llm, raw_paper, saved_article, mode="exam")
    await client.post(f"/api/sessions/{session_id}/start")
    await client.post(f"/api/sessions/{session_id}/answers", json={"number": 1, "raw": "A"})
    response = await client.get(f"/api/sessions/{session_id}/tutor-feedback/1")
    assert response.status_code == 409


async def test_models_and_history_routes(client, fake_llm, raw_paper, saved_article):
    models = await client.get("/api/models")
    assert models.status_code == 200 and models.json()[0]["name"] == "test-model"

    session_id = await make_session(client, fake_llm, raw_paper, saved_article)
    marker = json.dumps({"results": [{"number": 10, "verdict": "correct"}]})
    fake_llm.responses.append(ChatResult(content=marker, finish_reason="stop"))
    await client.post(f"/api/sessions/{session_id}/start")
    await client.post(f"/api/sessions/{session_id}/finish")

    history = (await client.get("/api/history")).json()
    assert history[0]["id"] == session_id and history[0]["status"] == "report"
    stats = (await client.get("/api/stats")).json()
    assert stats["total_sessions"] == 1


async def test_health_reports_key_and_catalog(client):
    health = (await client.get("/api/health")).json()
    assert health["api_key_set"] is True and health["catalog"] == 2


async def test_spa_index_served():
    from pathlib import Path

    from hyrip.web.app import build_app  # noqa: F401 — package import sanity

    static_dir = (
        Path(__import__("hyrip.web.routes", fromlist=["__file__"]).__file__).parent / "static"
    )
    assert (static_dir / "index.html").exists()
