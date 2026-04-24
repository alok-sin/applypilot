"""Ctx-aware LLM config resolution.

Locks the precedence contract for ``resolve_config_for_ctx``:
  task override → user llm.yaml (via ctx.user.llm_config) → user secrets /
  process env.

Two users in one process must never cross-contaminate keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from applypilot.core import build_local_run_context
from applypilot.llm import (
    LLMConfig,
    _task_config_from_yaml,
    get_client_for_ctx,
    resolve_config_for_ctx,
)


def _seed(root: Path, *, llm_yaml: str | None = None, env: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "profile.json").write_text("{}", encoding="utf-8")
    if llm_yaml is not None:
        (root / "llm.yaml").write_text(llm_yaml, encoding="utf-8")
    if env:
        (root / ".env").write_text(
            "\n".join(f"{k}={v}" for k, v in env.items()),
            encoding="utf-8",
        )
    return root


# ---------------------------------------------------------------------------
# Pure resolver
# ---------------------------------------------------------------------------


def test_task_config_from_yaml_returns_none_when_empty() -> None:
    assert _task_config_from_yaml({}, {}, "score") is None
    assert _task_config_from_yaml(None, {}, "score") is None


def test_task_config_from_yaml_resolves_task_entry() -> None:
    yaml_cfg = {
        "tasks": {
            "score": {"model": "gemini/gemini-3.0-flash"},
        }
    }
    env = {"GEMINI_API_KEY": "g-key"}
    cfg = _task_config_from_yaml(yaml_cfg, env, "score")
    assert cfg is not None
    assert cfg.provider == "gemini"
    assert cfg.model == "gemini/gemini-3.0-flash"
    assert cfg.api_key == "g-key"


def test_task_config_from_yaml_falls_through_to_default_section() -> None:
    yaml_cfg = {"default": {"model": "openai/gpt-4o-mini"}}
    env = {"OPENAI_API_KEY": "o-key"}
    cfg = _task_config_from_yaml(yaml_cfg, env, "tailor")
    assert cfg is not None
    assert cfg.model == "openai/gpt-4o-mini"
    assert cfg.api_key == "o-key"


# ---------------------------------------------------------------------------
# Ctx-aware resolution
# ---------------------------------------------------------------------------


def test_task_override_beats_yaml_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _seed(
        tmp_path / "u",
        llm_yaml="tasks:\n  score:\n    model: gemini/gemini-3.0-flash\n",
        env={"GEMINI_API_KEY": "yaml-wins"},
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ctx = build_local_run_context(root)

    override = LLMConfig(
        provider="openai",
        api_base=None,
        model="openai/gpt-4o-mini",
        api_key="byo-key",
    )
    ctx.task.llm_overrides["score"] = override

    resolved = resolve_config_for_ctx(ctx, "score")
    assert resolved is override  # exact instance, no rebuild


def test_user_yaml_beats_process_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ctx.user.llm_config (from llm.yaml) takes precedence over env-only resolution."""
    root = _seed(
        tmp_path / "u",
        llm_yaml="tasks:\n  score:\n    model: anthropic/claude-haiku-4-5\n",
        env={"ANTHROPIC_API_KEY": "anthro-key"},
    )
    # Process env has an unrelated key that would otherwise win resolve_llm_config.
    monkeypatch.setenv("OPENAI_API_KEY", "from-process-env")
    ctx = build_local_run_context(root)

    cfg = resolve_config_for_ctx(ctx, "score")
    assert cfg.provider == "anthropic"
    assert cfg.api_key == "anthro-key"


def test_env_fallback_when_no_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No llm.yaml → use resolve_llm_config against user secrets + os.environ."""
    root = _seed(tmp_path / "u", env={"GEMINI_API_KEY": "from-dotenv"})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = build_local_run_context(root)

    cfg = resolve_config_for_ctx(ctx, "score")
    assert cfg.provider == "gemini"
    assert cfg.api_key == "from-dotenv"


def test_two_ctxs_resolve_to_their_own_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two users in one process must see their own ``.env`` values, not each other's."""
    alice = _seed(
        tmp_path / "alice",
        llm_yaml="tasks:\n  score:\n    model: gemini/gemini-3.0-flash\n",
        env={"GEMINI_API_KEY": "alice-key"},
    )
    bob = _seed(
        tmp_path / "bob",
        llm_yaml="tasks:\n  score:\n    model: gemini/gemini-3.0-flash\n",
        env={"GEMINI_API_KEY": "bob-key"},
    )
    # Even with a wrong ambient value, user secrets dominate.
    monkeypatch.setenv("GEMINI_API_KEY", "ambient-never-wins")

    a = build_local_run_context(alice, user_id="alice")
    b = build_local_run_context(bob, user_id="bob")

    assert resolve_config_for_ctx(a, "score").api_key == "alice-key"
    assert resolve_config_for_ctx(b, "score").api_key == "bob-key"


def test_get_client_for_ctx_builds_fresh_per_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-ctx clients must not share identity — prevents cached credentials leaking."""
    root = _seed(
        tmp_path / "u",
        llm_yaml="tasks:\n  score:\n    model: gemini/gemini-3.0-flash\n",
        env={"GEMINI_API_KEY": "k"},
    )
    ctx = build_local_run_context(root)

    c1 = get_client_for_ctx(ctx, "score")
    c2 = get_client_for_ctx(ctx, "score")
    assert c1 is not c2  # intentional: no shared cache at the core level
    assert c1.config.api_key == "k"
    assert c2.config.api_key == "k"
