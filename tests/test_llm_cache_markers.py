from __future__ import annotations

from applypilot.llm import _apply_cache_markers


def test_anthropic_wraps_content_in_cache_control_block() -> None:
    messages = [
        {"role": "system", "content": "sys prompt", "cache": "ephemeral"},
        {"role": "user", "content": "resume text", "cache": "ephemeral"},
        {"role": "user", "content": "job body"},
    ]

    out = _apply_cache_markers("anthropic", messages)

    assert out[0] == {
        "role": "system",
        "content": [{"type": "text", "text": "sys prompt", "cache_control": {"type": "ephemeral"}}],
    }
    assert out[1] == {
        "role": "user",
        "content": [{"type": "text", "text": "resume text", "cache_control": {"type": "ephemeral"}}],
    }
    assert out[2] == {"role": "user", "content": "job body"}


def test_non_anthropic_strips_cache_key_and_keeps_string_content() -> None:
    messages = [
        {"role": "system", "content": "sys", "cache": "ephemeral"},
        {"role": "user", "content": "body"},
    ]

    for provider in ("openai", "gemini", "lightning", "local"):
        out = _apply_cache_markers(provider, messages)
        assert out[0] == {"role": "system", "content": "sys"}
        assert "cache" not in out[0]
        assert out[1] == {"role": "user", "content": "body"}


def test_message_without_cache_key_passes_through() -> None:
    messages = [{"role": "user", "content": "hi"}]
    assert _apply_cache_markers("anthropic", messages) == messages
    assert _apply_cache_markers("openai", messages) == messages


def test_apply_does_not_mutate_input() -> None:
    messages = [{"role": "system", "content": "sys", "cache": "ephemeral"}]
    original = [dict(m) for m in messages]
    _apply_cache_markers("anthropic", messages)
    assert messages == original
