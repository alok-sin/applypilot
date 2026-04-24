"""Context types threaded through every pipeline stage.

Two layers:

- ``UserContext`` — identity + data + providers. Stable per user; reusable
  across many tasks.
- ``TaskContext`` — per-invocation state. Cancellation token, per-task LLM
  overrides, progress callback, run id. Born and dies with one ``run()``.

``RunContext`` bundles both so stages only take one arg.

Providers are :class:`typing.Protocol` classes — any object with matching
methods satisfies them. No ABC hierarchies, no registration.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from applypilot.llm import LLMConfig


# ---------------------------------------------------------------------------
# Provider protocols — minimal; grow only when a stage needs more.
# ---------------------------------------------------------------------------


@runtime_checkable
class SecretsProvider(Protocol):
    """Read-only access to a user's stored secrets (API keys, etc.)."""

    def get(self, key: str, default: str | None = None) -> str | None: ...


@runtime_checkable
class StorageBackend(Protocol):
    """Write tailored artifacts; resolve paths to return to the DB.

    Local default = filesystem under the user's data root. Spoke swaps in
    S3/R2 without changing callers.
    """

    def tailored_dir(self) -> Path: ...
    def cover_letter_dir(self) -> Path: ...
    def log_dir(self) -> Path: ...


@runtime_checkable
class Database(Protocol):
    """Owns the DB connection for this user / tenant.

    Phase 0 keeps the core's sqlite shape — spoke adds a Postgres
    implementation later without changing what core calls.
    """

    def connection(self) -> sqlite3.Connection: ...
    def close(self) -> None: ...


@runtime_checkable
class CancellationToken(Protocol):
    """Per-task cancellation signal. Replaces the module-global stop_event."""

    def is_set(self) -> bool: ...
    def set(self) -> None: ...
    def clear(self) -> None: ...


@runtime_checkable
class ProgressReporter(Protocol):
    """Optional callback for stage-level progress updates."""

    def stage_start(self, stage: str) -> None: ...
    def stage_finish(self, stage: str, stats: dict[str, Any]) -> None: ...
    def event(self, name: str, payload: dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# Context dataclasses
# ---------------------------------------------------------------------------


@dataclass
class UserContext:
    """Identity + data + providers. Stable per user across tasks."""

    user_id: str
    data_dir: Path
    profile: dict
    resume_text: str
    resume_pdf: bytes | None
    search_config: dict
    llm_config: dict = field(default_factory=dict)
    secrets: SecretsProvider | None = None
    storage: StorageBackend | None = None
    db: Database | None = None


@dataclass
class TaskContext:
    """Per-invocation state. One per ``pipeline.run()`` call."""

    run_id: str
    cancellation: CancellationToken
    llm_overrides: dict[str, "LLMConfig"] = field(default_factory=dict)
    progress: ProgressReporter | None = None


@dataclass
class RunContext:
    """Bundle of :class:`UserContext` + :class:`TaskContext` passed to stages."""

    user: UserContext
    task: TaskContext
