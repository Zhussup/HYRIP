"""URL input: trafilatura extraction with a naive <p> fallback, then the normal pipeline."""

from __future__ import annotations

import re

import httpx

from ..config import Config
from ..errors import DataError
from ..models.article import Article, ArticleSource
from .normalize import article_from_source

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def article_from_url(url: str, config: Config, *, timeout_s: float | None = None) -> Article:
    if not url.lower().startswith(("http://", "https://")):
        raise DataError("URL must start with http:// or https://")
    html = _fetch(url, timeout_s or config.timeout_s)
    title, author, published = _metadata(html)
    errors: list[str] = []
    for candidate in _candidates(html):
        try:
            return article_from_source(
                candidate,
                ArticleSource.URL,
                config,
                title=title,
                source_url=url,
                author=author,
                published=published,
            )
        except DataError as exc:
            errors.append(str(exc))
    raise DataError(errors[0] if errors else "the page is empty")


def _fetch(url: str, timeout_s: float) -> str:
    try:
        response = httpx.get(
            url,
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DataError(f"Could not fetch {url}: {exc}") from exc
    return response.text


def _candidates(html: str) -> list[str]:
    candidates: list[str] = []
    try:
        import trafilatura

        text = (
            trafilatura.extract(html, include_comments=False, include_tables=False) or ""
        ).strip()
        if text:
            candidates.append(text)
    except Exception:
        pass
    naive = _naive_paragraphs(html)
    if naive:
        candidates.append("\n\n".join(naive))
    return candidates


def _metadata(html: str) -> tuple[str | None, str | None, str | None]:
    title = author = published = None
    try:
        import trafilatura

        meta = trafilatura.extract_metadata(html)
        if meta is not None:
            title = (meta.title or "").strip() or None
            author = str(meta.author or "").strip("[]") or None
            published = (meta.date or "").strip() or None
    except Exception:
        pass
    if title is None:  # og:title fallback
        match = re.search(
            r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"', html, flags=re.IGNORECASE
        )
        title = match.group(1).strip() if match else None
    return title, author, published


def _naive_paragraphs(html: str) -> list[str]:
    """Fallback: the page's <p> elements, tags stripped, headings/nav noise dropped by length."""
    paragraphs: list[str] = []
    for match in re.finditer(r"<p[^>]*>(.*?)</p>", html, flags=re.DOTALL | re.IGNORECASE):
        text = re.sub(r"<[^>]+>", " ", match.group(1))
        text = " ".join(text.replace("&amp;", "&").replace("&#39;", "'").split())
        if len(text) >= 40 and not text.count("{") > 2:
            paragraphs.append(text)
    return paragraphs
