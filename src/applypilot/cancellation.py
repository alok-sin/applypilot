"""Cooperative cancellation for long-running pipeline stages.

Module-level `threading.Event` — set when the user requests a graceful stop
(SIGINT). Long-running loops poll `stop_event.is_set()` at safe checkpoints
(per-job, per-URL). In-flight HTTP/LLM calls are not interrupted — the grace
period is "current unit finishes, next unit short-circuits."

Escape hatch: the SIGINT handler counts presses. First press sets the event
and raises KeyboardInterrupt. Second press calls os._exit(130), bypassing
every `except` clause in the stack.
"""

from __future__ import annotations

import os
import signal
import sys
import threading

# Process-global token. The SIGINT handler below still sets this — the CLI
# relies on it. New code (library callers, worker pools) should prefer a
# fresh per-task token via :func:`make_cancellation_token`.
stop_event = threading.Event()


def make_cancellation_token() -> threading.Event:
    """Return a fresh per-task cancellation token.

    A ``threading.Event`` structurally satisfies the core
    ``CancellationToken`` protocol (``is_set``/``set``/``clear``). Two
    concurrent pipeline runs get independent tokens and never observe each
    other's cancellation.
    """
    return threading.Event()

_sigint_count = 0
_sigint_lock = threading.Lock()
_installed = False


def install_sigint_handler() -> None:
    """Install the double-Ctrl+C handler. Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True

    def _handler(signum, frame):  # noqa: ARG001
        global _sigint_count
        with _sigint_lock:
            _sigint_count += 1
            count = _sigint_count

        if count == 1:
            stop_event.set()
            try:
                sys.stderr.write(
                    "\n[applypilot] Interrupt — finishing current unit then "
                    "stopping. Press Ctrl+C again to force quit.\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            raise KeyboardInterrupt()

        try:
            sys.stderr.write("\n[applypilot] Force quit.\n")
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(130)

    signal.signal(signal.SIGINT, _handler)
