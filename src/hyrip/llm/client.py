"""The one door to Ollama Cloud.

Chat goes to the OpenAI-compatible /v1/chat/completions with a Bearer key; the model catalog
to the keyless GET /api/tags. Plain sync httpx (no openai SDK): we need both hosts under one
key and nothing more than completions; using the SDK would hide drift-prone details.

A stream is never retried mid-flight — only opening it is. Interrupted streams surface the
partial output and the error, so consumers never see duplicated tokens.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from ..config import Config
from ..errors import ConfigError, LlmError

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRY_DELAYS = (2.0, 8.0)
_ATTEMPTS = 3


@dataclass
class ChatResult:
    content: str
    finish_reason: str | None = None
    latency_s: float = 0.0


@dataclass
class ModelInfo:
    name: str


class LLMClient:
    """Interface; tests inject a fake."""

    def chat(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:  # pragma: no cover - interface
        raise NotImplementedError

    def chat_stream(  # pragma: no cover - interface
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        raise NotImplementedError

    def list_models(self) -> list[ModelInfo]:  # pragma: no cover - interface
        raise NotImplementedError


class OllamaCloudClient(LLMClient):
    def __init__(self, config: Config):
        self.config = config

    # ---- internals ---------------------------------------------------------

    def _headers(self, *, need_key: bool) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Content-Encoding": "identity"}
        if need_key:
            if not self.config.api_key:
                raise ConfigError("OLLAMA_API_KEY is not set.")
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _payload(
        self,
        messages: list[dict],
        *,
        json_mode: bool,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict:
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    @staticmethod
    def _sleep_for_retry(status: int, headers: dict | httpx.Headers | None) -> None:
        retry_after = (headers or {}).get("Retry-After") if headers else None
        if isinstance(retry_after, str) and retry_after.isdigit():
            time.sleep(min(float(retry_after), 30.0))
        else:
            time.sleep(4.0)

    def chat(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        payload = self._payload(
            messages,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        url = self.config.base_url + "/chat/completions"
        started = time.monotonic()
        last_error: Exception | None = None
        result: ChatResult | None = None
        for attempt in range(_ATTEMPTS):
            try:
                with httpx.Client(
                    timeout=httpx.Timeout(10.0, read=self.config.timeout_s, write=30.0, pool=10.0)
                ) as client:
                    response = client.post(url, json=payload, headers=self._headers(need_key=True))
                if response.status_code in RETRYABLE_STATUS:
                    self._sleep_for_retry(response.status_code, response.headers)
                    raise LlmRetry(str(response.status_code))
                response.raise_for_status()
                body = response.json()
            except httpx.ConnectError as exc:
                last_error = exc
            except httpx.ReadTimeout as exc:
                last_error = exc
            except LlmRetry:
                last_error = LlmError("Ollama Cloud rate-limited or overloaded.")
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                detail = exc.response.text[:300]
                raise LlmError(f"Ollama Cloud error: HTTP {status}: {detail}") from exc
            except json.JSONDecodeError as exc:
                raise LlmError("Ollama Cloud returned a non-JSON body.") from exc
            else:
                try:
                    choice = body["choices"][0]
                    content = choice["message"].get("content") or ""
                    finish_reason = choice.get("finish_reason")
                except (KeyError, IndexError, TypeError) as exc:
                    raise LlmError("Unexpected response shape from Ollama Cloud.") from exc
                if content.strip():
                    result = ChatResult(
                        content=content,
                        finish_reason=finish_reason,
                        latency_s=time.monotonic() - started,
                    )
                    break
                # an empty body on HTTP 200 is usually a transient cloud hiccup — retry once more
                last_error = LlmError("Ollama Cloud returned an empty completion.")
            if attempt < len(_RETRY_DELAYS):
                time.sleep(_RETRY_DELAYS[attempt])
        else:
            raise LlmError(f"Ollama Cloud unreachable after {attempt + 1} attempts: {last_error}")
        assert result is not None  # loop breaks only on a result
        return result

    def chat_stream(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        self._headers(need_key=True)  # fail fast on a missing key before opening a stream
        payload = self._payload(
            messages, json_mode=False, temperature=temperature, max_tokens=max_tokens, stream=True
        )
        url = self.config.base_url + "/chat/completions"
        last_error: Exception | None = None
        yielded = False
        for attempt in range(_ATTEMPTS):
            try:
                with httpx.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=self._headers(need_key=True),
                    timeout=httpx.Timeout(10.0, read=self.config.timeout_s, write=30.0, pool=10.0),
                ) as response:
                    if response.status_code in RETRYABLE_STATUS:
                        self._sleep_for_retry(response.status_code, response.headers)
                        raise _OpenRetry(str(response.status_code))
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        body_text = line[len("data: ") :].strip()
                        if body_text == "[DONE]":
                            break
                        try:
                            obj = json.loads(body_text)
                        except json.JSONDecodeError:
                            continue
                        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                        if content := delta.get(
                            "content"
                        ):  # reasoner delta has separate `reasoning`
                            yielded = True
                            yield content
                    return
            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_error = exc
            except _OpenRetry as exc:
                last_error = exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                detail = exc.response.text[:300]
                raise LlmError(f"Ollama Cloud error: HTTP {status}: {detail}") from exc
            if yielded:  # never replay tokens we already sent
                raise LlmError(
                    f"Stream interrupted after emitting output: {last_error}"
                ) from last_error
            if attempt < len(_RETRY_DELAYS):
                time.sleep(_RETRY_DELAYS[attempt])
        raise LlmError(f"Ollama Cloud unreachable after {attempt + 1} attempts: {last_error}")

    def list_models(self) -> list[ModelInfo]:
        """Keyless: lets users verify connectivity before sorting out their API key."""
        try:
            response = httpx.get(self.config.tags_url, timeout=20.0)
            response.raise_for_status()
            names = [m["name"] for m in response.json().get("models", [])]
        except (httpx.HTTPError, ValueError) as exc:
            raise LlmError(f"Cannot reach the Ollama model catalog: {exc}") from exc
        return [ModelInfo(name=name) for name in names]


class LlmRetry(Exception):
    """Internal: a retryable HTTP status from a finished (non-stream) call."""


class _OpenRetry(Exception):
    """Internal: a retryable HTTP status that blocked a stream from even opening."""
