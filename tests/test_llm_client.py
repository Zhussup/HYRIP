"""The Ollama Cloud HTTP door: Bearer auth, JSON bodies, retries, streaming, catalog."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from hyrip.config import Config
from hyrip.errors import ConfigError, LlmError
from hyrip.llm import client as client_module
from hyrip.llm.client import OllamaCloudClient

CHAT_URL = "http://llm.test/v1/chat/completions"
TAGS_URL = "http://llm.test/api/tags"

COMPLETION_BODY = {
    "choices": [
        {"message": {"content": "hello there"}, "finish_reason": "stop"},
    ]
}


def _route(method: str = "POST", url: str = CHAT_URL):
    return respx.route(method=method, url=url)


@pytest.fixture
def client(config: Config) -> OllamaCloudClient:
    return OllamaCloudClient(config)


@respx.mock
def test_chat_sends_bearer_json_and_parses_content(client: OllamaCloudClient):
    route = _route().respond(json=COMPLETION_BODY)
    result = client.chat([{"role": "user", "content": "hi"}])
    assert route.called
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer test-key"
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert payload["stream"] is False
    assert payload["temperature"] == pytest.approx(0.4)
    assert "response_format" not in payload  # not a JSON-mode call
    assert result.content == "hello there"
    assert result.finish_reason == "stop"


@respx.mock
def test_json_mode_sets_response_format(client: OllamaCloudClient):
    route = _route().respond(json=COMPLETION_BODY)
    result = client.chat([{"role": "user", "content": "hi"}], json_mode=True, temperature=0.2)
    assert result.content == "hello there"
    payload = json.loads(route.calls.last.request.content)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == pytest.approx(0.2)


@respx.mock
def test_missing_key_raises_ConfigError_before_any_request(config: Config):
    import dataclasses

    client = OllamaCloudClient(dataclasses.replace(config, api_key=""))
    route = _route()
    with pytest.raises(ConfigError, match="OLLAMA_API_KEY"):
        client.chat([{"role": "user", "content": "hi"}])
    assert not route.called


@respx.mock
def test_retryable_status_then_success(client: OllamaCloudClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("hyrip.llm.client.time.sleep", lambda _: None)
    route = _route()
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "1"}, request=httpx.Request("POST", CHAT_URL)),
        httpx.Response(200, json=COMPLETION_BODY, request=httpx.Request("POST", CHAT_URL)),
    ]
    result = client.chat([{"role": "user", "content": "hi"}])
    assert result.content == "hello there"
    assert route.call_count == 2


@respx.mock
def test_retries_exhausted_raises_LlmError(
    client: OllamaCloudClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("hyrip.llm.client.time.sleep", lambda _: None)
    _route().mock(
        return_value=httpx.Response(
            503,
            request=httpx.Request("POST", CHAT_URL),
        )
    )
    with pytest.raises(LlmError, match=r"unreachable|rate-limited|503"):
        client.chat([{"role": "user", "content": "hi"}])


@respx.mock
def test_http_error_status_carries_detail(client: OllamaCloudClient):
    _route().mock(
        return_value=httpx.Response(
            401,
            text='{"error": "invalid api key"}',
            request=httpx.Request("POST", CHAT_URL),
        )
    )
    with pytest.raises(LlmError, match="401"):
        client.chat([{"role": "user", "content": "hi"}])


@respx.mock
def test_empty_completion_retries_then_raises(client: OllamaCloudClient, monkeypatch):
    monkeypatch.setattr(client_module, "_RETRY_DELAYS", (0.0, 0.0))
    route = _route().mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
        )
    )
    with pytest.raises(LlmError, match="empty completion"):
        client.chat([{"role": "user", "content": "hi"}])
    assert route.call_count == client_module._ATTEMPTS


@respx.mock
def test_empty_completion_recovers_after_retry(client: OllamaCloudClient, monkeypatch):
    monkeypatch.setattr(client_module, "_RETRY_DELAYS", (0.0, 0.0))
    route = _route().mock(
        side_effect=[
            httpx.Response(
                200, json={"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
            ),
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": "recovered"}, "finish_reason": "stop"}]},
            ),
        ]
    )
    result = client.chat([{"role": "user", "content": "hi"}])
    assert result.content == "recovered"
    assert route.call_count == 2


@respx.mock
def test_unexpected_shape_raises(client: OllamaCloudClient):
    _route().respond(json={"foo": "bar"})
    with pytest.raises(LlmError, match="Unexpected response shape"):
        client.chat([{"role": "user", "content": "hi"}])


@respx.mock
def test_stream_yields_content_deltas_and_skips_reasoning(client: OllamaCloudClient):
    def sse(request: httpx.Request) -> httpx.Response:
        events = [
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"reasoning": "hidden thought"}}]},
            {"choices": [{"delta": {"content": "lo!"}}]},
            "[DONE]",
        ]
        body = "".join(f"data: {ev if ev == '[DONE]' else json.dumps(ev)}\n\n" for ev in events)
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}, request=request
        )

    _route().mock(side_effect=sse)
    chunks = list(client.chat_stream([{"role": "user", "content": "hi"}]))
    assert chunks == ["Hel", "lo!"]
    payload = json.loads(_route().calls.last.request.content)
    assert payload["stream"] is True


@respx.mock
def test_stream_retryable_open_then_success(
    client: OllamaCloudClient, monkeypatch: pytest.MonkeyPatch
):
    """500s on every open: retries exhaust into a LlmError with nothing yielded."""
    monkeypatch.setattr("hyrip.llm.client.time.sleep", lambda _: None)
    _route().mock(return_value=httpx.Response(500, request=httpx.Request("POST", CHAT_URL)))
    chunks: list[str] = []
    with pytest.raises(LlmError, match="unreachable"):
        for chunk in client.chat_stream([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)
    assert chunks == []


@respx.mock
def test_stream_mid_stream_failure_raises_LlmError(
    client: OllamaCloudClient, monkeypatch: pytest.MonkeyPatch
):
    """A ReadTimeout after deltas have flown raises LlmError; no retry replays them."""

    def flaky_sse(request: httpx.Request) -> httpx.Response:
        def generator():
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'par'}}]})}\n\n".encode()
            raise httpx.ReadTimeout("cut")

        return httpx.Response(
            200,
            content=generator(),  # type: ignore[arg-type]
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    _route().mock(side_effect=flaky_sse)
    chunks = []
    with pytest.raises(LlmError, match=r"interrupted|Stream"):
        for chunk in client.chat_stream([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)
    assert chunks == ["par"]


@respx.mock
def test_stream_429_opens_retry_then_success(
    client: OllamaCloudClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("hyrip.llm.client.time.sleep", lambda _: None)
    _route().mock(
        side_effect=[
            httpx.Response(429, request=httpx.Request("POST", CHAT_URL)),
            httpx.Response(
                200,
                content=f"data: {json.dumps({'choices': [{'delta': {'content': 'ok'}}]})}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
                request=httpx.Request("POST", CHAT_URL),
            ),
        ]
    )
    assert list(client.chat_stream([{"role": "user", "content": "hi"}])) == ["ok"]


@respx.mock
def test_list_models_is_keyless_and_parses_names(config: Config):
    import dataclasses

    client = OllamaCloudClient(dataclasses.replace(config, api_key=""))
    route = respx.get(TAGS_URL).respond(
        json={"models": [{"name": "gpt-oss:120b"}, {"name": "kimi-k3"}]}
    )
    models = client.list_models()
    assert [m.name for m in models] == ["gpt-oss:120b", "kimi-k3"]
    sent = route.calls.last.request
    assert "Authorization" not in sent.headers


@respx.mock
def test_list_models_catalog_down_raises_LlmError(config: Config):
    respx.get(TAGS_URL).mock(
        return_value=httpx.Response(500, request=httpx.Request("GET", TAGS_URL))
    )
    with pytest.raises(LlmError, match="catalog"):
        OllamaCloudClient(config).list_models()
