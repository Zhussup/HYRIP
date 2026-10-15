"""Local file input (.txt / .md): the first short line is treated as the title."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..errors import DataError
from ..models.article import Article, ArticleSource
from .normalize import article_from_source

_TITLE_MAX_CHARS = 100


def article_from_file(path: str | Path, config: Config) -> Article:
    file = Path(path)
    if not file.is_file():
        raise DataError(f"No such file: {file}")
    try:
        text = file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise DataError(f"Cannot read {file}: {exc}") from exc
    text, title = _lift_first_title_line(text)
    if title is None:
        title = file.stem.replace("-", " ").replace("_", " ")
    return article_from_source(text, ArticleSource.FILE, config, title=title, source_url=str(file))


def _lift_first_title_line(text: str) -> tuple[str, str | None]:
    lines = text.splitlines()
    if not lines:
        return text, None
    first = lines[0].strip()
    if not (0 < len(first) <= _TITLE_MAX_CHARS):
        return text, None
    if text[len(lines[0]) :].lstrip(" \t").startswith("\n\n") or (
        len(lines) > 1 and not lines[1].strip()
    ):
        # "a title line" then a blank line: classic title + body
        rest = "\n".join(lines[1:]).lstrip("\n")
        cleaned = first.removeprefix("# ").removeprefix("## ").strip()
        return rest, cleaned
    return text, None
