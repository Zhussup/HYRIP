"""URL extraction: mocked HTTP only — trafilatura, metadata, naive fallback, errors."""

from __future__ import annotations

import httpx
import pytest
import respx

from hyrip.config import Config
from hyrip.errors import DataError
from hyrip.ingest.url_source import article_from_url

PARAS = [
    "Rooftop hives have appeared in the capital since the ban was lifted ten years ago, and now "
    "the city counts more bee colonies than its surrounding countryside, a study suggests. "
    "Beekeepers describe steady demand, and scientists note that city colonies seem to cope "
    "better than cousins in the countryside, and some keepers sell jars at weekend stalls.",
    "Ecologists caution that too many honeybees in one block can starve the wild insects that "
    "pollinate the parks' native plants, and a city hall has capped new permits. Beekeepers "
    "have responded with restraint: their network keeps a registry, and members pledge not to "
    "expand their number beyond a shared limit they set together a few years ago, and the registry of hives grows slowly through the year.",
]


def html(paragraphs: list[str]) -> str:
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return (
        '<html><head><meta property="og:title" content="Rooftop bees boom"/>'
        f"</head><body>{body}</body></html>"
    )


@pytest.fixture
def article_html() -> str:
    return html(PARAS)


@respx.mock
def test_url_happy_path_with_og_title(config: Config, article_html: str):
    respx.get("https://news.test/bees").mock(return_value=httpx.Response(200, text=article_html))
    article = article_from_url("https://news.test/bees", config)
    assert article.source.value == "url"
    assert article.title == "Rooftop bees boom"
    assert article.source_url == "https://news.test/bees"
    assert article.word_count == sum(len(p.split()) for p in article.paragraphs)
    assert article.paragraph_labels[0] == "A"


@respx.mock
def test_naive_fallback_when_extractor_returns_none(config: Config, article_html: str, monkeypatch):
    import trafilatura

    monkeypatch.setattr(trafilatura, "extract", lambda *a, **k: None)
    respx.get("https://news.test/bees2").mock(return_value=httpx.Response(200, text=article_html))
    article = article_from_url("https://news.test/bees2", config)
    assert article.paragraphs == PARAS


@respx.mock
def test_http_error_is_DataError(config: Config):
    respx.get("https://news.test/404").mock(return_value=httpx.Response(404))
    with pytest.raises(DataError, match="404"):
        article_from_url("https://news.test/404", config)


def test_non_http_scheme_rejected(config: Config):
    with pytest.raises(DataError, match="http"):
        article_from_url("ftp://nope.example", config)


@respx.mock
def test_no_extractable_text_is_DataError(config: Config):
    respx.get("https://news.test/empty").mock(
        return_value=httpx.Response(200, text="<html><body></body></html>")
    )
    with pytest.raises(DataError):
        article_from_url("https://news.test/empty", config)


@respx.mock
def test_word_cap_still_enforced_for_url(config: Config):
    long_text = " ".join(["city honey"] * 3000)
    respx.get("https://news.test/long").mock(
        return_value=httpx.Response(200, text=html([long_text]))
    )
    with pytest.raises(DataError, match="limit"):
        article_from_url("https://news.test/long", config)
