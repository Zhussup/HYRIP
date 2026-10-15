"""Text → Article: paragraph splitting, markdown cleanup, labeling, word counts."""

from __future__ import annotations

import re
from datetime import datetime

from ..config import Config
from ..errors import DataError
from ..models.article import Article, ArticleSource, now_utc

_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MARKDOWN_EMPH = re.compile(r"\*{1,3}([^*]+?)\*{1,3}|_{1,3}([^_]+?)_{1,3}")
_MARKDOWN_CODE = re.compile(r"`{1,3}([^`]*)`{1,3}")


def _strip_markdown(text: str) -> str:
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _MARKDOWN_HEADING.sub("", text)
    text = _MARKDOWN_EMPH.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _MARKDOWN_CODE.sub(r"\1", text)
    return text


def split_paragraphs(text: str) -> list[str]:
    """Blank-line-separated paragraphs; hard-wrapped lines inside one paragraph are joined."""
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        merged = " ".join(line.strip() for line in block.splitlines())
        merged = re.sub(r"\s+", " ", merged).strip()
        if merged:
            paragraphs.append(merged)
    return paragraphs


def word_count(text: str) -> int:
    return len(text.split())


def labels_for(count: int) -> list[str]:
    from ..models.questions import paragraph_label as _label

    return [_label(i) for i in range(count)]


_QUOTE_NOISE = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
    }
)
_QUOTE_PUNCT = re.compile(r"[.,;:!?(){}\[\]\"'‘’“”\-]")


def quote_normalized(text: str) -> str:
    """Lowercase, punctuation-free, whitespace-collapsed form for verbatim quote checks."""
    text = text.translate(_QUOTE_NOISE)
    text = _QUOTE_PUNCT.sub("", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_finds(quote: str, haystack: str) -> bool:
    needle = quote_normalized(quote)
    if not needle:
        return False
    return needle in quote_normalized(haystack)


def build_article(
    text: str,
    *,
    source: ArticleSource,
    title: str = "Untitled article",
    source_url: str | None = None,
    author: str | None = None,
    published: str | None = None,
    fetched_at: datetime | None = None,
    config: Config,
) -> Article:
    """Normalize raw article text into an Article, enforcing the word-count caps."""
    text = _strip_markdown(text or "")
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        raise DataError("No article text found — nothing to build a paper from.")
    body = "\n\n".join(paragraphs)
    total = word_count(body)
    if total > config.max_article_words:
        raise DataError(
            f"Article is {total} words; the limit is {config.max_article_words}."
            " Longer passages produce low-quality questions."
        )
    if total < config.min_article_words:
        raise DataError(
            f"Article is only {total} words; at least {config.min_article_words}"
            " are needed for meaningful IELTS questions."
        )
    article = Article(
        source=source,
        title=title,
        source_url=source_url,
        author=author,
        published=published,
        text=body,
        paragraphs=paragraphs,
        paragraph_labels=labels_for(len(paragraphs)),
        word_count=total,
        fetched_at=fetched_at or now_utc(),
    )
    return article


def article_from_source(raw: str, source: ArticleSource, config: Config, **meta) -> Article:
    if not raw or not raw.strip():
        raise DataError("Empty article text.")
    return build_article(raw, source=source, config=config, **meta)
