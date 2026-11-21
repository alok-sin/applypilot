import pytest

from applypilot.llm import _litellm_model, resolve_llm_config


def test_infers_provider_from_first_configured_source() -> None:
    cfg = resolve_llm_config(
        {
            "GEMINI_API_KEY": "g-key",
            "OPENAI_API_KEY": "o-key",
            "ANTHROPIC_API_KEY": "a-key",
            "LLM_URL": "http://127.0.0.1:8080/v1",
        }
    )
    assert cfg.provider == "gemini"
    assert cfg.model == "gemini/gemini-3.0-flash"
    assert cfg.api_key == "g-key"


def test_unprefixed_model_uses_inferred_provider() -> None:
    cfg = resolve_llm_config({"LLM_MODEL": "gpt-4o-mini", "OPENAI_API_KEY": "o-key"})
    assert cfg.provider == "openai"
    assert cfg.model == "openai/gpt-4o-mini"


def test_requires_model_provider_prefix_without_inferable_provider() -> None:
    with pytest.raises(RuntimeError, match="must include a provider prefix"):
        resolve_llm_config({"LLM_MODEL": "gpt-4o-mini", "LLM_API_KEY": "generic"})


def test_provider_and_api_key_come_from_model_contract() -> None:
    cfg = resolve_llm_config({"LLM_MODEL": "gemini/gemini-3.0-flash", "GEMINI_API_KEY": "g-key"})
    assert cfg.provider == "gemini"
    assert cfg.api_base is None
    assert cfg.model == "gemini/gemini-3.0-flash"
    assert cfg.api_key == "g-key"


def test_uses_generic_api_key_for_unmapped_provider() -> None:
    cfg = resolve_llm_config({"LLM_MODEL": "vertex_ai/gemini-3.0-flash", "LLM_API_KEY": "v-key"})
    assert cfg.provider == "vertex_ai"
    assert cfg.api_key == "v-key"


def test_llm_url_infers_local_default_model_and_allows_missing_api_key() -> None:
    cfg = resolve_llm_config(
        {
            "LLM_URL": "http://127.0.0.1:8080/v1/",
        }
    )
    assert cfg.provider == "openai"
    assert cfg.model == "openai/local-model"
    assert cfg.api_base == "http://127.0.0.1:8080/v1"
    assert cfg.api_key == ""


def test_missing_everything_raises_clear_error() -> None:
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        resolve_llm_config({})


def test_infers_lightning_provider_with_default_model_and_api_base() -> None:
    cfg = resolve_llm_config({"LIGHTNING_API_KEY": "l-key"})
    assert cfg.provider == "lightning"
    assert cfg.model == "lightning/lightning-ai/gemma-4-31B-it"
    assert cfg.api_base == "https://lightning.ai/api/v1"
    assert cfg.api_key == "l-key"


def test_litellm_model_passes_non_lightning_models_through() -> None:
    assert _litellm_model("openai/gpt-4o-mini") == "openai/gpt-4o-mini"
    assert _litellm_model("gemini/gemini-2.5-flash") == "gemini/gemini-2.5-flash"
    assert _litellm_model("anthropic/claude-haiku-4-5") == "anthropic/claude-haiku-4-5"


def test_litellm_model_translates_lightning_prefix_to_openai() -> None:
    assert (
        _litellm_model("lightning/lightning-ai/gemma-4-31B-it")
        == "openai/lightning-ai/gemma-4-31B-it"
    )


@pytest.mark.parametrize("raw", ["true", "TRUE", "True", "1", "yes", "YES", "Yes", " true "])
def test_llm_streaming_mode_truthy_values(raw: str) -> None:
    cfg = resolve_llm_config({"GEMINI_API_KEY": "g-key", "LLM_STREAMING_MODE": raw})
    assert cfg.use_streaming is True


@pytest.mark.parametrize("raw", ["false", "FALSE", "0", "no", ""])
def test_llm_streaming_mode_falsy_values(raw: str) -> None:
    cfg = resolve_llm_config({"GEMINI_API_KEY": "g-key", "LLM_STREAMING_MODE": raw})
    assert cfg.use_streaming is False


def test_llm_streaming_mode_defaults_to_false_when_unset() -> None:
    cfg = resolve_llm_config({"GEMINI_API_KEY": "g-key"})
    assert cfg.use_streaming is False


def test_explicit_fallback_resolves_provider_scoped_key() -> None:
    cfg = resolve_llm_config(
        {
            "OPENAI_API_KEY": "o-key",
            "GEMINI_API_KEY": "g-key",
            "LLM_FALLBACK_MODEL": "gemini/gemini-2.5-flash",
        }
    )
    assert cfg.fallback_model == "gemini/gemini-2.5-flash"
    assert cfg.fallback_api_key == "g-key"
    assert cfg.fallback_api_base is None
