"""Default local-disk providers for single-user use.

These wrap today's layout under the user's data dir (``APP_DIR`` today =
``~/.applypilot-plus/``) so the existing CLI works unchanged after the
context refactor. The spoke repo ships Postgres/S3/KMS equivalents when
multi-tenant mode arrives — same protocols, different storage.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from pathlib import Path

from dotenv import dotenv_values

from applypilot.cancellation import make_cancellation_token
from applypilot.core.context import RunContext, TaskContext, UserContext


class LocalSecretsProvider:
    """Reads ``.env`` from the user's data dir, with ``os.environ`` fallback.

    Kept deliberately dumb: load on init, read on access. The ``.env`` file
    is authoritative — ``os.environ`` is consulted only when the key is
    absent from the file (matches today's behavior where the CLI loads
    ``.env`` into the process env before stages run).
    """

    def __init__(self, data_dir: Path | str) -> None:
        self._env_path = Path(data_dir) / ".env"
        self._values: dict[str, str | None] = dict(dotenv_values(self._env_path)) if self._env_path.exists() else {}

    def get(self, key: str, default: str | None = None) -> str | None:
        value = self._values.get(key)
        if value:
            return value
        env_value = os.environ.get(key)
        if env_value:
            return env_value
        return default


class LocalStorageBackend:
    """Filesystem storage under the user's data dir."""

    def __init__(self, data_dir: Path | str) -> None:
        self._root = Path(data_dir)

    def tailored_dir(self) -> Path:
        return self._root / "tailored_resumes"

    def cover_letter_dir(self) -> Path:
        return self._root / "cover_letters"

    def log_dir(self) -> Path:
        return self._root / "logs"


class SqliteDatabase:
    """Thin wrapper around the core SQLite DB for one user's data dir.

    Per-thread connection cache — SQLite connections are pinned to their
    creating thread by default, and the pipeline runs parallel workers.
    Mirrors the existing pattern in [applypilot.database].

    Core stays single-table (same schema as today's [database.py]); the
    jobs/user_jobs split is a spoke-side concern.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._local = threading.local()

    def connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.execute("SELECT 1")
                return conn
            except sqlite3.ProgrammingError:
                pass  # closed — fall through and reopen

        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.row_factory = sqlite3.Row
        self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None


class LocalCancellationToken:
    """Process-local cancellation token backed by a fresh ``threading.Event``.

    Explicit class (rather than just aliasing the Event) so callers can
    :func:`isinstance` check in tests without importing ``threading``.
    """

    def __init__(self) -> None:
        self._event = make_cancellation_token()

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()


class EventCancellationToken:
    """Adapter that wraps an existing :class:`threading.Event` in the
    :class:`CancellationToken` protocol.

    Used by the CLI to bridge the module-global SIGINT-driven
    ``cancellation.stop_event`` into per-task contexts without orphaning
    any callsite that still reads the module-global.
    """

    def __init__(self, event) -> None:
        self._event = event

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()


# ---------------------------------------------------------------------------
# Convenience builder — the single-user CLI uses this to hydrate a
# RunContext from the existing APP_DIR layout.
# ---------------------------------------------------------------------------


def build_default_run_context(*, run_id: str | None = None) -> RunContext:
    """Build the CLI's default :class:`RunContext`.

    Wraps today's ``APP_DIR`` layout and bridges cancellation to the
    module-global ``stop_event`` so SIGINT continues to abort the run.

    Import-time dependencies live inside the function so module import
    stays cheap (no ``config`` or ``cancellation`` import cost on
    ``applypilot.core`` consumers that never call this).
    """
    from applypilot.cancellation import stop_event
    from applypilot.config import APP_DIR

    ctx = build_local_run_context(APP_DIR, run_id=run_id)
    ctx.task.cancellation = EventCancellationToken(stop_event)
    return ctx


def build_local_run_context(
    data_dir: Path | str,
    *,
    user_id: str = "local",
    run_id: str | None = None,
) -> RunContext:
    """Hydrate a :class:`RunContext` from a single-user data directory.

    The directory shape matches today's ``APP_DIR``:

    - ``profile.json``, ``resume.txt``, ``resume.pdf`` (optional)
    - ``searches.yaml``, ``llm.yaml`` (optional)
    - ``.env`` (optional; LLM API keys)
    - ``applypilot.db`` (SQLite — created on first DB access)
    """
    import yaml  # lazy: keep core import cheap

    from applypilot.prompts import load_prompts

    root = Path(data_dir)
    profile_path = root / "profile.json"
    resume_path = root / "resume.txt"
    resume_pdf_path = root / "resume.pdf"
    searches_path = root / "searches.yaml"
    llm_yaml_path = root / "llm.yaml"
    prompts_yaml_path = root / "prompts.yaml"
    db_path = root / "applypilot.db"

    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}
    resume_text = resume_path.read_text(encoding="utf-8") if resume_path.exists() else ""
    resume_pdf = resume_pdf_path.read_bytes() if resume_pdf_path.exists() else None
    search_config = yaml.safe_load(searches_path.read_text(encoding="utf-8")) if searches_path.exists() else {}
    llm_config = yaml.safe_load(llm_yaml_path.read_text(encoding="utf-8")) if llm_yaml_path.exists() else {}
    prompts = load_prompts(prompts_yaml_path)

    user = UserContext(
        user_id=user_id,
        data_dir=root,
        profile=profile or {},
        resume_text=resume_text,
        resume_pdf=resume_pdf,
        search_config=search_config or {},
        llm_config=llm_config or {},
        prompts=prompts,
        secrets=LocalSecretsProvider(root),
        storage=LocalStorageBackend(root),
        db=SqliteDatabase(db_path),
    )
    task = TaskContext(
        run_id=run_id or uuid.uuid4().hex[:12],
        cancellation=LocalCancellationToken(),
    )
    return RunContext(user=user, task=task)
