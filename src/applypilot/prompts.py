"""LLM prompt loading: package defaults + sparse user overrides.

Defaults ship with the package at ``src/applypilot/config/prompts.yaml``.
CLI users override any prompt by writing the same keys to
``~/.applypilot-plus/prompts.yaml`` -- only the keys present in the user
file replace the defaults; everything else falls through.

Templates use :class:`string.Template` (``${var}`` syntax). Literal JSON
braces ``{ }`` need no escaping; only literal ``$`` must be written ``$$``.
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any

import yaml

from applypilot.config import CONFIG_DIR

PACKAGE_PROMPTS_PATH = CONFIG_DIR / "prompts.yaml"


def load_prompts(user_prompts_path: Path | None = None) -> dict:
    """Load package defaults and deep-merge user overrides on top.

    The package default file is required -- a missing or empty file is a
    deployment bug, not a runtime fallback case, and we fail loud rather
    than silently returning empty prompts.
    """
    if not PACKAGE_PROMPTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing package prompts at {PACKAGE_PROMPTS_PATH}. "
            "This is a packaging bug -- prompts.yaml must ship with the package."
        )
    defaults = yaml.safe_load(PACKAGE_PROMPTS_PATH.read_text(encoding="utf-8")) or {}
    if not defaults:
        raise ValueError(f"Package prompts file at {PACKAGE_PROMPTS_PATH} is empty.")

    if user_prompts_path and Path(user_prompts_path).exists():
        overrides = yaml.safe_load(Path(user_prompts_path).read_text(encoding="utf-8")) or {}
        return _deep_merge(defaults, overrides)
    return defaults


def render_prompt(prompts: dict, key: str, /, **vars: Any) -> str:
    """Look up a dotted key (e.g. ``scoring.score.system``) and substitute ``${vars}``.

    Uses :meth:`string.Template.safe_substitute` so a missing variable renders
    as the literal ``${name}`` rather than raising -- handy during incremental
    refactors where not every call-site supplies every var yet.
    """
    template = _resolve_dotted(prompts, key)
    return Template(template).safe_substitute(**vars)


def _resolve_dotted(d: dict, key: str) -> str:
    parts = key.split(".")
    node: Any = d
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            raise KeyError(f"Prompt key not found: {key!r} (missing segment {p!r})")
        node = node[p]
    if not isinstance(node, str):
        raise TypeError(f"Prompt key {key!r} resolved to non-string ({type(node).__name__})")
    return node


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict where ``override`` keys win, recursing into nested dicts.

    For non-dict leaves (the prompt strings themselves), the override value
    replaces the base value entirely. Lists are not merged element-wise --
    the override list replaces the base list.
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
