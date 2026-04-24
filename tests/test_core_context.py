"""Phase-0 smoke test for applypilot.core.

Exists to lock the API contract that the spoke repo depends on:
multiple independent ``RunContext``s in one process must not share state.
If this ever fails, Phase 1 (multi-user admin mode) is broken.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from applypilot.core import (
    CancellationToken,
    Database,
    LocalCancellationToken,
    LocalSecretsProvider,
    LocalStorageBackend,
    RunContext,
    SecretsProvider,
    SqliteDatabase,
    StorageBackend,
    build_local_run_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_profile_dir(root: Path, user: str, llm_key: str) -> Path:
    """Write a minimal single-user profile layout at ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "profile.json").write_text(json.dumps({"name": user}), encoding="utf-8")
    (root / "resume.txt").write_text(f"resume for {user}\n", encoding="utf-8")
    (root / "searches.yaml").write_text(f"discover_tags: [{user}]\n", encoding="utf-8")
    (root / "llm.yaml").write_text("model: gemini/gemini-3.0-flash\n", encoding="utf-8")
    (root / ".env").write_text(f"GEMINI_API_KEY={llm_key}\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_local_providers_satisfy_protocols(tmp_path: Path) -> None:
    """runtime_checkable Protocol membership — makes refactors loud not silent."""
    root = _seed_profile_dir(tmp_path / "alice", "alice", "k-alice")
    ctx = build_local_run_context(root, user_id="alice")

    assert isinstance(ctx.user.secrets, SecretsProvider)
    assert isinstance(ctx.user.storage, StorageBackend)
    assert isinstance(ctx.user.db, Database)
    assert isinstance(ctx.task.cancellation, CancellationToken)

    # The concrete classes should also match (sanity).
    assert isinstance(ctx.user.secrets, LocalSecretsProvider)
    assert isinstance(ctx.user.storage, LocalStorageBackend)
    assert isinstance(ctx.user.db, SqliteDatabase)
    assert isinstance(ctx.task.cancellation, LocalCancellationToken)


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_two_contexts_have_independent_data_roots(tmp_path: Path) -> None:
    alice_root = _seed_profile_dir(tmp_path / "alice", "alice", "k-alice")
    bob_root = _seed_profile_dir(tmp_path / "bob", "bob", "k-bob")

    alice = build_local_run_context(alice_root, user_id="alice")
    bob = build_local_run_context(bob_root, user_id="bob")

    assert alice.user.data_dir != bob.user.data_dir
    assert alice.user.profile["name"] == "alice"
    assert bob.user.profile["name"] == "bob"

    # Secrets don't leak.
    assert alice.user.secrets.get("GEMINI_API_KEY") == "k-alice"
    assert bob.user.secrets.get("GEMINI_API_KEY") == "k-bob"

    # Storage roots differ.
    assert alice.user.storage.tailored_dir() != bob.user.storage.tailored_dir()
    assert alice.user.storage.tailored_dir().is_relative_to(alice_root)
    assert bob.user.storage.tailored_dir().is_relative_to(bob_root)


def test_cancellation_tokens_are_independent(tmp_path: Path) -> None:
    """Cancelling one task must not affect any other task in the same process."""
    a = build_local_run_context(_seed_profile_dir(tmp_path / "a", "a", "ka"), user_id="a")
    b = build_local_run_context(_seed_profile_dir(tmp_path / "b", "b", "kb"), user_id="b")

    assert not a.task.cancellation.is_set()
    assert not b.task.cancellation.is_set()

    a.task.cancellation.set()
    assert a.task.cancellation.is_set()
    assert not b.task.cancellation.is_set()


def test_databases_are_separate_files(tmp_path: Path) -> None:
    """Writes to one user's DB are not visible to the other."""
    a = build_local_run_context(_seed_profile_dir(tmp_path / "a", "a", "ka"), user_id="a")
    b = build_local_run_context(_seed_profile_dir(tmp_path / "b", "b", "kb"), user_id="b")

    conn_a = a.user.db.connection()
    conn_a.execute("CREATE TABLE marker (who TEXT)")
    conn_a.execute("INSERT INTO marker VALUES ('alice')")
    conn_a.commit()

    conn_b = b.user.db.connection()
    tables = conn_b.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    assert [row[0] for row in tables] == []  # b's DB is empty

    a.user.db.close()
    b.user.db.close()


def test_concurrent_pipelines_do_not_cross_contaminate(tmp_path: Path) -> None:
    """Two threads, two contexts, two DBs — no interleaved writes.

    This is the contract Phase 1 depends on: the admin CLI will run the
    pipeline for N users in parallel (or rapid succession) and expect each
    user's data to land in its own files without a module-global holding
    anything.
    """
    contexts = [
        build_local_run_context(
            _seed_profile_dir(tmp_path / name, name, f"k-{name}"),
            user_id=name,
        )
        for name in ("alice", "bob", "carol")
    ]

    barrier = threading.Barrier(len(contexts))
    errors: list[BaseException] = []

    def _work(ctx: RunContext) -> None:
        try:
            barrier.wait(timeout=5)
            conn = ctx.user.db.connection()
            conn.execute("CREATE TABLE IF NOT EXISTS scoreboard (who TEXT, n INTEGER)")
            for i in range(25):
                conn.execute(
                    "INSERT INTO scoreboard VALUES (?, ?)", (ctx.user.user_id, i)
                )
            conn.commit()
        except BaseException as exc:  # pragma: no cover - defensive
            errors.append(exc)

    threads = [threading.Thread(target=_work, args=(ctx,)) for ctx in contexts]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors, errors

    # Each DB holds exactly its own user's rows.
    for ctx in contexts:
        conn = ctx.user.db.connection()
        rows = conn.execute("SELECT DISTINCT who FROM scoreboard").fetchall()
        assert [r[0] for r in rows] == [ctx.user.user_id]
        count = conn.execute("SELECT COUNT(*) FROM scoreboard").fetchone()[0]
        assert count == 25
        ctx.user.db.close()


# ---------------------------------------------------------------------------
# Directory-hydration edge cases
# ---------------------------------------------------------------------------


def test_missing_optional_files_do_not_crash_build(tmp_path: Path) -> None:
    """A data dir with only profile.json should still build a RunContext."""
    root = tmp_path / "sparse"
    root.mkdir()
    (root / "profile.json").write_text("{}", encoding="utf-8")

    ctx = build_local_run_context(root)
    assert ctx.user.profile == {}
    assert ctx.user.resume_text == ""
    assert ctx.user.resume_pdf is None
    assert ctx.user.search_config == {}
    assert ctx.user.llm_config == {}
    # Secrets provider exists even if .env is absent; lookups return None.
    assert ctx.user.secrets.get("NOPE") is None


def test_run_id_defaults_to_random_and_is_overridable(tmp_path: Path) -> None:
    root = _seed_profile_dir(tmp_path / "r", "r", "kr")
    a = build_local_run_context(root)
    b = build_local_run_context(root)
    assert a.task.run_id != b.task.run_id  # random by default

    c = build_local_run_context(root, run_id="fixed-id")
    assert c.task.run_id == "fixed-id"


# ---------------------------------------------------------------------------
# Secrets resolution
# ---------------------------------------------------------------------------


def test_secrets_env_file_wins_over_process_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-user .env values beat ambient os.environ, so running user A's
    pipeline never accidentally picks up user B's leaked process env."""
    root = _seed_profile_dir(tmp_path / "u", "u", "from-file")
    monkeypatch.setenv("GEMINI_API_KEY", "from-process-env")
    ctx = build_local_run_context(root)
    assert ctx.user.secrets.get("GEMINI_API_KEY") == "from-file"


def test_secrets_fall_back_to_process_env_when_dotenv_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "noenv"
    root.mkdir()
    (root / "profile.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SOME_KEY", "from-env")
    ctx = build_local_run_context(root)
    assert ctx.user.secrets.get("SOME_KEY") == "from-env"
    assert ctx.user.secrets.get("MISSING", "fallback") == "fallback"
