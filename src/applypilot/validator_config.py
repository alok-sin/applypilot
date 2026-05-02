"""Validator word-list loading: package defaults + sparse user overrides.

Defaults ship at ``src/applypilot/config/validator.yaml``. A user can drop a
sparse YAML file at ``<data_dir>/validator.yaml`` to override individual
keys; only listed keys replace the defaults, others fall through.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from applypilot.config import CONFIG_DIR

PACKAGE_VALIDATOR_PATH = CONFIG_DIR / "validator.yaml"


def load_validator_config(user_path: Path | None = None) -> dict:
    """Load package defaults and overlay any user override file."""
    if not PACKAGE_VALIDATOR_PATH.exists():
        raise FileNotFoundError(
            f"Missing package validator config at {PACKAGE_VALIDATOR_PATH}. "
            "This is a packaging bug -- validator.yaml must ship with the package."
        )
    defaults = yaml.safe_load(PACKAGE_VALIDATOR_PATH.read_text(encoding="utf-8")) or {}

    if user_path and Path(user_path).exists():
        overrides = yaml.safe_load(Path(user_path).read_text(encoding="utf-8")) or {}
        merged = dict(defaults)
        for k, v in overrides.items():
            merged[k] = v
        return merged
    return defaults
