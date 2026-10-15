"""Shared fixtures: config, store, articles, papers, and a programmable FakeLLM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyrip.config import Config
from hyrip.ingest.normalize import article_from_source
from hyrip.llm.client import ChatResult, LLMClient, ModelInfo
from hyrip.models.article import Article, ArticleSource
from hyrip.models.llm_contracts import RawGeneratedPaper
from hyrip.store.db import Store

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        api_key="test-key",
        base_url="http://llm.test/v1",
        tags_url="http://llm.test/api/tags",
        model="test-model",
        timeout_s=5.0,
        max_tokens=6000,
        temperature=0.4,
        coach_notes=False,
        data_dir=tmp_path,
        n_questions_default=12,
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "test.db")


@pytest.fixture
def article_fixture_text() -> str:
    return (FIXTURES / "article_bbc.txt").read_text(encoding="utf-8")


@pytest.fixture
def article(config: Config, article_fixture_text: str) -> Article:
    title, _, body = article_fixture_text.partition("\n\n")
    return article_from_source(body, source=ArticleSource.FILE, config=config, title=title)


@pytest.fixture
def raw_paper() -> RawGeneratedPaper:
    data = json.loads((FIXTURES / "paper_full_12.json").read_text(encoding="utf-8"))
    return RawGeneratedPaper.model_validate(data)


class FakeLLM(LLMClient):
    """Chat responses queued ahead of the call; every call recorded for assertions."""

    def __init__(self):
        self.responses: list[ChatResult] = []
        self.streams: list[list[str]] = []
        self.calls: list[dict] = []

    def _record(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})

    def chat(self, messages, *, json_mode=False, temperature=None, max_tokens=None):
        self._record(messages, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens)
        if self.responses:
            return self.responses.pop(0)
        return ChatResult(content="{}", finish_reason="stop", latency_s=0.01)

    def chat_stream(self, messages, *, temperature=None, max_tokens=None):
        self._record(messages, temperature=temperature, max_tokens=max_tokens)
        yield from self.streams.pop(0) if self.streams else []

    def list_models(self):
        self.calls.append({"messages": None, "list_models": True})
        return [ModelInfo(name="test-model"), ModelInfo(name="other-model")]

    @property
    def last_user_message(self) -> str:
        for message in reversed(self.calls[-1]["messages"]):
            if message["role"] == "user":
                return message["content"]
        return ""


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
