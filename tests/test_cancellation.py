"""Tests for cooperative cancellation + SIGINT escape hatch."""

from __future__ import annotations

import signal
from types import SimpleNamespace

import litellm
import pytest

import applypilot.cancellation as cancellation
import applypilot.llm as llm_module
from applypilot.llm import LLMClient, LLMConfig


@pytest.fixture(autouse=True)
def _reset_cancellation_state(monkeypatch):
    """Each test runs with a clean stop_event, counter, and original SIGINT handler."""
    original_handler = signal.getsignal(signal.SIGINT)
    cancellation.stop_event.clear()
    monkeypatch.setattr(cancellation, "_sigint_count", 0)
    monkeypatch.setattr(cancellation, "_installed", False)
    yield
    cancellation.stop_event.clear()
    signal.signal(signal.SIGINT, original_handler)


# --- Handler installation ---------------------------------------------------


def test_install_sigint_handler_is_idempotent():
    """Calling install_sigint_handler twice must not stack handlers."""
    cancellation.install_sigint_handler()
    first = signal.getsignal(signal.SIGINT)
    cancellation.install_sigint_handler()
    second = signal.getsignal(signal.SIGINT)
    assert first is second


def test_first_sigint_sets_stop_event_and_raises_keyboard_interrupt():
    cancellation.install_sigint_handler()
    handler = signal.getsignal(signal.SIGINT)

    assert not cancellation.stop_event.is_set()
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGINT, None)
    assert cancellation.stop_event.is_set()


def test_second_sigint_calls_os_exit_130(monkeypatch):
    """Second Ctrl+C must bypass every `except` via os._exit(130)."""
    cancellation.install_sigint_handler()
    handler = signal.getsignal(signal.SIGINT)

    exit_codes: list[int] = []

    def _fake_exit(code: int) -> None:
        exit_codes.append(code)
        raise SystemExit(code)  # stop execution in the test

    monkeypatch.setattr(cancellation.os, "_exit", _fake_exit)

    # First press — cooperative
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGINT, None)

    # Second press — hard kill
    with pytest.raises(SystemExit):
        handler(signal.SIGINT, None)

    assert exit_codes == [130]


# --- Rate-limit classification ----------------------------------------------


def _rate_limit_error() -> litellm.RateLimitError:
    return litellm.RateLimitError("429 quota exceeded", llm_provider="openai", model="gpt-4o-mini")


def test_is_rate_limit_error_recognizes_real_litellm_rate_limit_error() -> None:
    assert llm_module._is_rate_limit_error(_rate_limit_error()) is True


def test_is_rate_limit_error_recognizes_status_code_429_attribute() -> None:
    exc = RuntimeError("too many requests")
    exc.status_code = 429  # type: ignore[attr-defined]
    assert llm_module._is_rate_limit_error(exc) is True


def test_is_rate_limit_error_follows_nested_cause_chain() -> None:
    inner = RuntimeError("upstream rejected")
    inner.status_code = 429  # type: ignore[attr-defined]
    outer = RuntimeError("wrapped failure")
    outer.__cause__ = inner
    assert llm_module._is_rate_limit_error(outer) is True


def test_is_rate_limit_error_ignores_generic_runtime_error_mentioning_rate_limit() -> None:
    """A message that merely says "rate limit 429" must not be classified as one."""
    assert llm_module._is_rate_limit_error(RuntimeError("rate limit 429")) is False


# --- LLM client integration -------------------------------------------------


def _client() -> LLMClient:
    return LLMClient(
        LLMConfig(
            provider="openai",
            api_base=None,
            model="openai/gpt-4o-mini",
            api_key="test-key",
            fallback_model="gemini/gemini-2.5-flash",
            fallback_api_key="fallback-key",
        )
    )


def test_chat_short_circuits_when_stop_event_already_set(monkeypatch):
    """If cancellation is requested before the call, don't touch litellm."""
    called = {"count": 0}

    def _fake_completion(**kwargs):
        called["count"] += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="x"))])

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)

    cancellation.stop_event.set()
    with pytest.raises(KeyboardInterrupt):
        _client().chat([{"role": "user", "content": "hi"}])

    assert called["count"] == 0


def test_chat_does_not_fall_back_when_cancelled_mid_request(monkeypatch):
    """Primary raises after cancellation — fallback must not be attempted."""
    calls: list[str] = []

    def _fake_completion(**kwargs):
        calls.append(kwargs["model"])
        # Simulate litellm wrapping a KeyboardInterrupt as a generic exception
        # after the user presses Ctrl+C.
        cancellation.stop_event.set()
        raise RuntimeError("connection aborted")

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)

    with pytest.raises(KeyboardInterrupt):
        _client().chat([{"role": "user", "content": "hi"}])

    assert calls == ["openai/gpt-4o-mini"]


def test_chat_re_raises_keyboard_interrupt_from_primary(monkeypatch):
    """A genuine KeyboardInterrupt from litellm must propagate, not fall back."""
    calls: list[str] = []

    def _fake_completion(**kwargs):
        calls.append(kwargs["model"])
        raise KeyboardInterrupt()

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)

    with pytest.raises(KeyboardInterrupt):
        _client().chat([{"role": "user", "content": "hi"}])

    assert calls == ["openai/gpt-4o-mini"]


def test_chat_still_falls_back_on_normal_error(monkeypatch):
    """When not cancelled, a normal primary failure should still trigger fallback."""
    calls: list[str] = []

    def _fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "openai/gpt-4o-mini":
            raise RuntimeError("boom")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)

    result = _client().chat([{"role": "user", "content": "hi"}])
    assert result == "ok"
    assert calls == ["openai/gpt-4o-mini", "gemini/gemini-2.5-flash"]


def test_chat_falls_back_on_real_litellm_rate_limit_error(monkeypatch):
    """A genuine litellm.RateLimitError from the primary must trigger fallback."""
    calls: list[str] = []

    def _fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "openai/gpt-4o-mini":
            raise _rate_limit_error()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="fallback ok"))])

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)

    result = _client().chat([{"role": "user", "content": "hi"}])
    assert result == "fallback ok"
    assert calls == ["openai/gpt-4o-mini", "gemini/gemini-2.5-flash"]
