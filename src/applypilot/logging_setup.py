"""Shared logging configuration for ApplyPilot CLIs and library callers.

Both the alok CLI and the saas spoke (and any future entry point) need the
same console handler, level filtering, and noisy-library suppression.
This module centralizes that so the behavior stays consistent.
"""

from __future__ import annotations

import logging
from pathlib import Path


class _ColorFormatter(logging.Formatter):
    """Colorize log levels for terminal output only."""

    _RESET = "\033[0m"
    _LEVEL_COLORS = {
        logging.DEBUG: "\033[36m",      # cyan
        logging.INFO: "\033[32m",       # green
        logging.WARNING: "\033[33m",    # yellow
        logging.ERROR: "\033[31m",      # red
        logging.CRITICAL: "\033[1;31m", # bold red
    }

    def format(self, record: logging.LogRecord) -> str:
        original = record.levelname
        color = self._LEVEL_COLORS.get(record.levelno)
        if color:
            record.levelname = f"{color}{original}{self._RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


def parse_log_level(value: str) -> int:
    level = getattr(logging, value.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown log level: {value!r}")
    return level


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """Set consistent logging output for any ApplyPilot entry point."""
    root_level = parse_log_level(level)
    noisy_level = logging.INFO if root_level <= logging.DEBUG else logging.WARNING

    console_handler = logging.StreamHandler()
    console_handler.setLevel(root_level)
    console_handler.setFormatter(
        _ColorFormatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    )
    logging.basicConfig(level=root_level, handlers=[console_handler], force=True)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(root_level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        )
        logging.getLogger().addHandler(file_handler)

    for name in (
        "LiteLLM",
        "LiteLLM Router",
        "LiteLLM Proxy",
        "litellm",
        "httpx",
        "httpcore",
        "openai",
    ):
        noisy = logging.getLogger(name)
        noisy.handlers.clear()
        noisy.setLevel(noisy_level)
        noisy.propagate = True
