"""Phase-0 smoke: pipeline honors the injected RunContext.

Locks two contracts the spoke depends on:
- ``run_pipeline`` accepts an external ``RunContext`` and forwards it.
- Cancellation flows through ``ctx.task.cancellation`` — a set token
  short-circuits the sequential driver before stages run.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

import applypilot.pipeline as pipeline
from applypilot.core import (
    EventCancellationToken,
    LocalCancellationToken,
    RunContext,
    build_local_run_context,
)


@pytest.fixture
def silent_console(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "console",
        type("DummyConsole", (), {"print": staticmethod(lambda *a, **k: None)})(),
    )


@pytest.fixture
def seed_app_dir(tmp_path: Path) -> Path:
    root = tmp_path / "app"
    root.mkdir()
    (root / "profile.json").write_text("{}", encoding="utf-8")
    return root


def test_event_cancellation_token_wraps_threading_event() -> None:
    event = threading.Event()
    token = EventCancellationToken(event)

    assert not token.is_set()
    event.set()
    assert token.is_set()
    token.clear()
    assert not event.is_set()
    token.set()
    assert event.is_set()


def test_run_pipeline_uses_injected_ctx_cancellation(monkeypatch, silent_console, seed_app_dir):
    """A pre-cancelled ctx must stop the pipeline before any stage runs."""
    ctx = build_local_run_context(seed_app_dir, user_id="u", run_id="r")
    ctx.task.cancellation.set()

    # Neutralize bootstrap + DB side effects.
    monkeypatch.setattr(pipeline, "load_env", lambda: None)
    monkeypatch.setattr(pipeline, "ensure_dirs", lambda: None)
    monkeypatch.setattr(pipeline, "init_db_for_ctx", lambda ctx: None)
    monkeypatch.setattr(pipeline, "get_stats_for_ctx", lambda ctx: {
        "total": 0, "pending_detail": 0, "with_description": 0, "scored": 0,
        "tailored": 0, "with_cover_letter": 0, "ready_to_apply": 0, "applied": 0,
    })

    calls: list[str] = []

    def _fail_stage(*args, **kwargs):
        calls.append("ran")
        return {"status": "ok"}

    for stage_name in pipeline._STAGE_RUNNERS:
        monkeypatch.setitem(pipeline._STAGE_RUNNERS, stage_name, _fail_stage)

    result = pipeline.run_pipeline(stages=["discover"], ctx=ctx)

    assert calls == []
    assert result["stages"] == []


def test_run_pipeline_default_ctx_wires_to_stop_event(monkeypatch, silent_console, tmp_path: Path):
    """Without an injected ctx, the CLI default wires cancellation to the
    process-global stop_event — SIGINT still aborts the run."""
    from applypilot import cancellation

    # Point APP_DIR at an empty scratch dir so build_default_run_context
    # doesn't read the real user's home.
    root = tmp_path / "app"
    root.mkdir()
    (root / "profile.json").write_text("{}", encoding="utf-8")
    import applypilot.config as config_mod
    monkeypatch.setattr(config_mod, "APP_DIR", root)

    monkeypatch.setattr(pipeline, "load_env", lambda: None)
    monkeypatch.setattr(pipeline, "ensure_dirs", lambda: None)
    monkeypatch.setattr(pipeline, "init_db_for_ctx", lambda ctx: None)
    monkeypatch.setattr(pipeline, "get_stats_for_ctx", lambda ctx: {
        "total": 0, "pending_detail": 0, "with_description": 0, "scored": 0,
        "tailored": 0, "with_cover_letter": 0, "ready_to_apply": 0, "applied": 0,
    })

    seen_ctx: dict[str, RunContext] = {}

    def _capture_sequential(ctx, ordered, *args, **kwargs):
        seen_ctx["ctx"] = ctx
        return {"stages": [], "errors": {}, "elapsed": 0.0}

    monkeypatch.setattr(pipeline, "_run_sequential", _capture_sequential)

    cancellation.stop_event.clear()
    try:
        pipeline.run_pipeline(stages=["discover"])
        ctx = seen_ctx["ctx"]
        assert isinstance(ctx.task.cancellation, EventCancellationToken)
        assert not ctx.task.cancellation.is_set()

        cancellation.stop_event.set()
        assert ctx.task.cancellation.is_set()
    finally:
        cancellation.stop_event.clear()


def test_two_pipeline_ctxs_cancel_independently(tmp_path: Path) -> None:
    """Two RunContexts with their own LocalCancellationToken — cancelling
    one must not affect the other. This is the multi-user contract."""
    alice_root = tmp_path / "alice"
    alice_root.mkdir()
    (alice_root / "profile.json").write_text("{}", encoding="utf-8")
    bob_root = tmp_path / "bob"
    bob_root.mkdir()
    (bob_root / "profile.json").write_text("{}", encoding="utf-8")

    alice = build_local_run_context(alice_root, user_id="alice")
    bob = build_local_run_context(bob_root, user_id="bob")

    assert isinstance(alice.task.cancellation, LocalCancellationToken)
    assert isinstance(bob.task.cancellation, LocalCancellationToken)

    alice.task.cancellation.set()
    assert alice.task.cancellation.is_set()
    assert not bob.task.cancellation.is_set()
