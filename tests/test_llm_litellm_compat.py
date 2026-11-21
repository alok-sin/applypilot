"""Hermetic streaming-mode compatibility tests for the litellm-backed LLMClient.

No network: ``litellm.completion`` is replaced with fakes that mimic the
chunked streaming response shape (``choices[0].delta.content``) that
litellm 1.110.2 produces for ``stream=True`` calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import applypilot.llm as llm_module
from applypilot.llm import LLMClient, LLMConfig


def _streaming_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def _streaming_client() -> LLMClient:
    return LLMClient(
        LLMConfig(
            provider="openai",
            api_base=None,
            model="openai/gpt-4o-mini",
            api_key="test-key",
            use_streaming=True,
        )
    )


def test_streaming_chat_accumulates_chunks_and_requests_stream(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_completion(**kwargs: object):
        captured.update(kwargs)
        return iter([_streaming_chunk("hello "), _streaming_chunk("stream"), _streaming_chunk("")])

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)

    response = _streaming_client().chat([{"role": "user", "content": "hi"}], max_output_tokens=64)

    assert response == "hello stream"
    assert captured["stream"] is True
    assert captured["model"] == "openai/gpt-4o-mini"
    assert captured["max_tokens"] == 64


def test_streaming_chat_skips_chunks_without_delta_content(monkeypatch) -> None:
    """Chunks lacking a non-empty delta.content must not corrupt the output."""

    class _Empty:
        pass

    def _fake_completion(**kwargs: object):
        return iter(
            [
                SimpleNamespace(choices=[]),  # no choices
                SimpleNamespace(choices=[SimpleNamespace(delta=_Empty())]),  # no content attr
                _streaming_chunk("ok"),
            ]
        )

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)

    response = _streaming_client().chat([{"role": "user", "content": "hi"}], max_output_tokens=64)

    assert response == "ok"


def test_streaming_chat_empty_stream_raises_runtime_error(monkeypatch) -> None:
    def _fake_completion(**kwargs: object):
        return iter([_streaming_chunk(""), _streaming_chunk("")])

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)

    with pytest.raises(RuntimeError, match="no text content"):
        _streaming_client().chat([{"role": "user", "content": "hi"}], max_output_tokens=64)
