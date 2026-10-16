"""Runtime configuration, resolved from environment variables only (no config file in v1)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

DEFAULT_BASE_URL = "https://ollama.com/v1"
DEFAULT_TAGS_URL = "https://ollama.com/api/tags"
DEFAULT_MODEL = "gpt-oss:120b"


def _as_bool(value: str) -> bool:
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


@dataclass(frozen=True)
class Config:
    """Everything a session needs to run; frontends pass it around as-is."""

    api_key: str | None
    base_url: str
    tags_url: str
    model: str
    timeout_s: float
    max_tokens: int
    temperature: float
    coach_notes: bool
    data_dir: Path
    n_questions_default: int
    time_budget_min: int = 20
    max_article_words: int = 4000
    warn_article_words: int = 1800
    min_article_words: int = 120
    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Config:
        env = dict(os.environ if env is None else env)
        return cls(
            api_key=env.get("OLLAMA_API_KEY") or None,
            base_url=env.get("HYRIP_BASE_URL") or DEFAULT_BASE_URL,
            tags_url=env.get("HYRIP_TAGS_URL") or DEFAULT_TAGS_URL,
            model=env.get("HYRIP_MODEL") or DEFAULT_MODEL,
            timeout_s=float(env.get("HYRIP_TIMEOUT") or 180),
            # 12-item Cambridge-style papers from a verbose model run 4-8k tokens; 6000 was
            # cut mid-list in observed runs ("Invalid JSON: EOF while parsing a list").
            max_tokens=int(env.get("HYRIP_MAX_TOKENS") or 12000),
            temperature=float(env.get("HYRIP_TEMPERATURE") or 0.4),
            coach_notes=_as_bool(env.get("COACH_NOTES") or "1"),
            data_dir=paths.data_dir(),
            n_questions_default=int(env.get("HYRIP_QUESTIONS") or 12),
            extra=env,
        )
