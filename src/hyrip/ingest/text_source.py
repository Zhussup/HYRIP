"""Paste input: normalize free text into an Article (the title may be user-supplied)."""

from __future__ import annotations

from ..config import Config
from ..errors import DataError
from ..models.article import Article, ArticleSource
from .normalize import article_from_source


def article_from_text(raw: str, config: Config, *, title: str | None = None) -> Article:
    if not raw or not raw.strip():
        raise DataError("The pasted text is empty — send the article body and nothing else.")
    if title is not None and not title.strip():
        title = None
    return article_from_source(raw, ArticleSource.PASTE, config, title=title or "Pasted article")
