"""Typos, drift and omissions in hyrip raise these; frontends render them, never tracebacks."""

from __future__ import annotations


class HyripError(Exception):
    """Base class for all hyrip errors."""

    hint: str | None = None

    def user_message(self) -> str:
        return str(self)


class ConfigError(HyripError):
    """Missing or invalid configuration (usually OLLAMA_API_KEY)."""

    hint = "export OLLAMA_API_KEY='...'  # from ollama.com → Settings → API keys"


class DataError(HyripError):
    """Article ingestion failed (unreadable file, blocked URL, empty text)."""

    hint = "Copy the article text directly and use `hyrip read` instead."


class GenerationError(HyripError):
    """Question-paper generation failed even after validation + repair."""

    hint = "Generation is retried automatically once; try again, or try another model via --model."


class LlmError(HyripError):
    """Ollama Cloud call failed (network, rate limit, bad model name)."""


class StateError(HyripError):
    """Invalid session state transition (e.g. answering after the session was finished)."""


class StoreError(HyripError):
    """Local database cannot be opened or written."""
