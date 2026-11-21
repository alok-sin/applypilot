"""Live Gemini smoke test through the real ``LLMClient``.

Requires ``GEMINI_API_KEY`` and is skipped otherwise. Run manually:

    pytest tests/test_gemini_smoke.py -m smoke

The default smoke model must exist in the installed litellm's
``model_cost`` catalog; the offline test below guards that without
needing any credentials.
"""

import os

import litellm
import pytest

from applypilot.llm import LLMClient, LLMConfig

# Verified present in litellm 1.110.2's model_cost catalog (unlike
# gemini-3.0-flash, which only exists on the pre-upgrade branch default).
DEFAULT_SMOKE_MODEL = "gemini/gemini-2.5-flash"


def _smoke_model() -> str:
    """Return a litellm-qualified model name, honoring GEMINI_SMOKE_MODEL."""
    raw = os.getenv("GEMINI_SMOKE_MODEL", DEFAULT_SMOKE_MODEL).strip()
    if raw.startswith("gemini/"):
        return raw
    if raw.startswith("models/"):
        raw = raw.split("/", 1)[1]
    return f"gemini/{raw}"


def test_default_smoke_model_exists_in_litellm_catalog() -> None:
    """Offline guard: the smoke default must be a known litellm model."""
    assert DEFAULT_SMOKE_MODEL in litellm.model_cost, (
        f"{DEFAULT_SMOKE_MODEL} is missing from litellm's model_cost catalog; "
        "pick a model the installed litellm recognizes."
    )


@pytest.mark.smoke
def test_gemini_smoke_chat_returns_non_empty_content() -> None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("Set GEMINI_API_KEY to run Gemini smoke tests.")

    prompt = os.getenv("GEMINI_SMOKE_PROMPT", "Reply with a single word: ready.")
    client = LLMClient(
        LLMConfig(
            provider="gemini",
            api_base=None,
            model=_smoke_model(),
            api_key=api_key,
        )
    )
    response = client.chat(
        [{"role": "user", "content": prompt}],
        max_output_tokens=32,
        timeout=60,
        num_retries=1,
    )

    assert response.strip(), "Gemini smoke call returned empty content."
