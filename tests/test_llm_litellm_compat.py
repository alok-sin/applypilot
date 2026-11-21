"""Hermetic streaming-mode compatibility tests for the litellm-backed LLMClient.

No network: ``litellm.completion`` is replaced with fakes that mimic the
chunked streaming response shape (``choices[0].delta.content``) that
litellm 1.110.2 produces for ``stream=True`` calls, plus one round trip
against the real litellm using its offline ``mock_response`` hook.
"""

from __future__ import annotations

import warnings
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


def test_streaming_chat_round_trips_real_litellm_completion(monkeypatch) -> None:
    """Hermetic streaming round trip against the REAL litellm.completion(stream=True).

    The wrapper injects only litellm's offline ``mock_response`` hook and
    forwards every other kwarg to the installed litellm unchanged. This
    validates that the client accumulates litellm 1.110.2's real streaming
    chunk shape (``choices[0].delta.content``), not just hand-rolled fakes.
    As with the non-streaming round trip, drop_params=True means unknown
    kwargs may be silently dropped — this is a compatibility/response-shape
    check, not a strict signature check.
    """
    real_completion = llm_module.litellm.completion
    captured: dict[str, object] = {}

    def _mock_response_completion(**kwargs: object):
        captured.update(kwargs)
        return real_completion(**{**kwargs, "mock_response": "hello stream"})

    monkeypatch.setattr(llm_module.litellm, "completion", _mock_response_completion)

    # litellm 1.110.2 builds pydantic response schemas lazily on the first
    # real completion() call, and pydantic emits ReadOnly-qualifier
    # UserWarnings for litellm's TypedDicts. That is third-party noise
    # (src/applypilot/llm.py already ignores it at runtime; pytest resets
    # warning filters per test), so scope the same suppression here instead
    # of letting it fail strict -W error runs.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
        response = _streaming_client().chat([{"role": "user", "content": "hi"}], max_output_tokens=64)

    assert response == "hello stream"
    assert captured["stream"] is True
    assert captured["model"] == "openai/gpt-4o-mini"
    assert captured["max_tokens"] == 64
