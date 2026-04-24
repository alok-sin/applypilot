"""Public API for applypilot as a library.

Stability contract: everything re-exported here is stable between minor
releases. Anything not re-exported is free to change without notice.
"""

from applypilot.core.context import (
    CancellationToken,
    Database,
    ProgressReporter,
    RunContext,
    SecretsProvider,
    StorageBackend,
    TaskContext,
    UserContext,
)
from applypilot.core.local_providers import (
    EventCancellationToken,
    LocalCancellationToken,
    LocalSecretsProvider,
    LocalStorageBackend,
    SqliteDatabase,
    build_default_run_context,
    build_local_run_context,
)

__all__ = [
    "CancellationToken",
    "Database",
    "EventCancellationToken",
    "LocalCancellationToken",
    "LocalSecretsProvider",
    "LocalStorageBackend",
    "ProgressReporter",
    "RunContext",
    "SecretsProvider",
    "SqliteDatabase",
    "StorageBackend",
    "TaskContext",
    "UserContext",
    "build_default_run_context",
    "build_local_run_context",
]
