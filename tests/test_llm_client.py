import os
import warnings
from types import SimpleNamespace

import pytest

import applypilot.llm as llm_module
from applypilot.llm import LLMClient, LLMConfig


def test_client_init_does_not_mutate_provider_env(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Seed a known value so monkeypatch restores litellm's original
    # suppress_debug_info state after the test instead of leaking ours.
    monkeypatch.setattr(llm_module.litellm, "suppress_debug_info", False, raising=False)
    LLMClient(
        LLMConfig(
            provider="openai",
            api_base=None,
            model="openai/gpt-4o-mini",
            api_key="test-key",
        )
    )
    assert "OPENAI_API_KEY" not in os.environ
    assert llm_module.litellm.suppress_debug_info is True


def _mock_response(content: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


def test_chat_passes_defaults_without_temperature(monkeypatch) -> None:
    client = LLMClient(
        LLMConfig(
            provider="openai",
            api_base=None,
            model="openai/gpt-4o-mini",
            api_key="test-key",
        )
    )
    captured: dict[str, object] = {}

    def _fake_completion(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return _mock_response()

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)
    response = client.chat([{"role": "user", "content": "hello"}], max_output_tokens=128)

    assert response == "hello"
    assert captured["model"] == "openai/gpt-4o-mini"
    assert captured["max_tokens"] == 128
    assert captured["timeout"] == 120
    assert captured["num_retries"] == 5
    assert captured["drop_params"] is True
    assert captured["api_key"] == "test-key"
    assert captured["api_base"] is None
    assert "temperature" not in captured


def test_chat_supports_temperature_and_typed_extra(monkeypatch) -> None:
    client = LLMClient(
        LLMConfig(
            provider="gemini",
            api_base=None,
            model="gemini/gemini-3.0-flash",
            api_key="g-key",
        )
    )
    captured: dict[str, object] = {}

    def _fake_completion(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return _mock_response("ok")

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)
    response = client.chat(
        [{"role": "user", "content": "hello"}],
        max_output_tokens=64,
        temperature=0.2,
        top_p=0.9,
        stop=["\n\n"],
        response_format={"type": "json_object"},
    )

    assert response == "ok"
    assert captured["model"] == "gemini/gemini-3.0-flash"
    assert captured["api_key"] == "g-key"
    assert captured["temperature"] == 0.2
    assert captured["top_p"] == 0.9
    assert captured["stop"] == ["\n\n"]
    assert captured["response_format"] == {"type": "json_object"}


def test_chat_sets_local_api_base_and_api_key(monkeypatch) -> None:
    client = LLMClient(
        LLMConfig(
            provider="openai",
            api_base="http://127.0.0.1:8080/v1",
            model="openai/local-model",
            api_key="local-key",
        )
    )
    captured: dict[str, object] = {}

    def _fake_completion(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return _mock_response()

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)
    _ = client.chat([{"role": "user", "content": "hello"}], max_output_tokens=64)

    assert captured["api_base"] == "http://127.0.0.1:8080/v1"
    assert captured["api_key"] == "local-key"


@pytest.mark.parametrize("model", ["openai/gpt-4o-mini", "gemini/gemini-2.5-flash"])
def test_chat_round_trips_real_litellm_completion(monkeypatch, model: str) -> None:
    """Hermetic integration against the REAL litellm.completion().

    The wrapper only injects litellm's offline ``mock_response`` hook and
    forwards every other kwarg to the installed litellm unchanged, so the
    round trip fails loudly if the client's kwarg set drifts out of sync
    with the installed litellm's completion() signature.
    """
    real_completion = llm_module.litellm.completion
    captured: dict[str, object] = {}

    def _mock_response_completion(**kwargs: object) -> object:
        captured.update(kwargs)
        return real_completion(**{**kwargs, "mock_response": "hello from litellm"})

    monkeypatch.setattr(llm_module.litellm, "completion", _mock_response_completion)

    client = LLMClient(
        LLMConfig(
            provider=model.split("/", 1)[0],
            api_base=None,
            model=model,
            api_key="test-key",
        )
    )
    # litellm 1.110.2 builds pydantic response schemas lazily on the first
    # real completion() call, and pydantic emits ReadOnly-qualifier
    # UserWarnings for litellm's TypedDicts. That is third-party noise
    # (src/applypilot/llm.py already ignores it at runtime; pytest resets
    # warning filters per test), so scope the same suppression here instead
    # of letting it fail strict -W error runs.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
        response = client.chat([{"role": "user", "content": "hello"}], max_output_tokens=64)

    assert response == "hello from litellm"
    assert captured["model"] == model
    assert captured["max_tokens"] == 64


def test_chat_translates_lightning_model_to_openai(monkeypatch) -> None:
    """provider=lightning must route through litellm's openai/ compatibility layer."""
    client = LLMClient(
        LLMConfig(
            provider="lightning",
            api_base="https://lightning.ai/api/v1",
            model="lightning/lightning-ai/gemma-4-31B-it",
            api_key="k",
        )
    )
    captured: dict[str, object] = {}

    def _fake_completion(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return _mock_response()

    monkeypatch.setattr(llm_module.litellm, "completion", _fake_completion)
    response = client.chat([{"role": "user", "content": "hello"}], max_output_tokens=64)

    assert response == "hello"
    assert captured["model"] == "openai/lightning-ai/gemma-4-31B-it"
    assert captured["api_base"] == "https://lightning.ai/api/v1"
    assert captured["api_key"] == "k"
