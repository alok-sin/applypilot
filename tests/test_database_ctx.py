"""Ctx-aware database layer.

Locks the contract that two ctxs with independent ``Database`` providers
see each other's data as empty — the row-level isolation Phase 1 depends
on.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from applypilot.core import build_local_run_context
from applypilot.database import (
    get_stats_for_ctx,
    init_db_for_ctx,
    init_schema,
    store_jobs,
)


def _seed(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "profile.json").write_text("{}", encoding="utf-8")
    return root


def test_init_schema_is_idempotent(tmp_path: Path) -> None:
    root = _seed(tmp_path / "u")
    ctx = build_local_run_context(root)

    init_db_for_ctx(ctx)
    init_db_for_ctx(ctx)  # second call must not crash or duplicate

    conn = ctx.user.db.connection()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "jobs" in tables
    ctx.user.db.close()


def test_init_schema_accepts_bare_connection(tmp_path: Path) -> None:
    """Pure helper works on any sqlite3 connection — in-memory included."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)

    # Schema present + ensure_columns ran.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    assert "url" in cols
    assert "filter_reason" in cols  # added by the migration registry
    assert "tailored_resume_json_path" in cols


def test_get_stats_for_ctx_reads_from_ctx_db(tmp_path: Path) -> None:
    root = _seed(tmp_path / "u")
    ctx = build_local_run_context(root)
    init_db_for_ctx(ctx)

    conn = ctx.user.db.connection()
    store_jobs(
        conn,
        [
            {"url": "https://x.test/1", "title": "Eng", "location": "Remote"},
            {"url": "https://x.test/2", "title": "SRE", "location": "NY"},
        ],
        site="test",
        strategy="direct",
    )

    stats = get_stats_for_ctx(ctx)
    assert stats["total"] == 2
    assert stats["scored"] == 0
    ctx.user.db.close()


def test_two_ctxs_have_row_level_isolation(tmp_path: Path) -> None:
    alice = build_local_run_context(_seed(tmp_path / "alice"), user_id="alice")
    bob = build_local_run_context(_seed(tmp_path / "bob"), user_id="bob")

    init_db_for_ctx(alice)
    init_db_for_ctx(bob)

    store_jobs(
        alice.user.db.connection(),
        [{"url": "https://x.test/a", "title": "Alice Role"}],
        site="t",
        strategy="d",
    )
    store_jobs(
        bob.user.db.connection(),
        [
            {"url": "https://x.test/b1", "title": "Bob Role 1"},
            {"url": "https://x.test/b2", "title": "Bob Role 2"},
        ],
        site="t",
        strategy="d",
    )

    assert get_stats_for_ctx(alice)["total"] == 1
    assert get_stats_for_ctx(bob)["total"] == 2

    # Alice's row is not visible in Bob's DB.
    bob_urls = [
        r[0] for r in bob.user.db.connection().execute("SELECT url FROM jobs").fetchall()
    ]
    assert "https://x.test/a" not in bob_urls

    alice.user.db.close()
    bob.user.db.close()


def test_ctx_db_does_not_touch_module_global_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """init_db_for_ctx must not create/use the process-wide get_connection cache.

    If Phase 1 runs pipelines for N users in a loop and each run touched
    DB_PATH via the module thread-local, the wrong file would get written.
    """
    import applypilot.database as db_mod

    root = _seed(tmp_path / "u")
    ctx = build_local_run_context(root)

    calls: list[object] = []
    real_get_connection = db_mod.get_connection

    def _tracked(db_path=None):
        calls.append(db_path)
        return real_get_connection(db_path)

    monkeypatch.setattr(db_mod, "get_connection", _tracked)

    init_db_for_ctx(ctx)
    get_stats_for_ctx(ctx)

    assert calls == []  # ctx path bypassed the module-global entirely
    ctx.user.db.close()
