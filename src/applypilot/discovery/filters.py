"""Shared post-discover filters.

Two layers of cheap (non-LLM) filtering applied to discovered job rows before
enrichment/scoring. Any stage may also call these helpers on demand.

Layer 1 — location filter (existing semantics, promoted from smartrecruiters):
    _location_ok(location, accept, reject) -> bool

Layer 2 — rule gate:
    country_reject(location, blocked_countries) -> bool
    seniority_reject(title, user_years, floor_years) -> bool
    rule_evaluate(job, cfg, profile) -> (ok, reason)

`rule_evaluate` returns a reason tag suitable for storing in the `filter_reason`
column: "seniority_mismatch" | "country_blocked" | None.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from applypilot import config

log = logging.getLogger(__name__)

_REMOTE_TOKENS = ("remote", "anywhere", "work from home", "wfh", "distributed")
_JUNIOR_TITLE_RE = re.compile(
    r"\b(intern|internship|new[\s\-]?grad|entry[\s\-]?level|junior|graduate|trainee|apprentice)\b",
    re.IGNORECASE,
)


def _load_location_filter(search_cfg: dict | None = None) -> tuple[list[str], list[str]]:
    """Load location accept/reject lists from search config."""
    if search_cfg is None:
        search_cfg = config.load_search_config()
    accept = search_cfg.get("location_accept", []) or []
    reject = search_cfg.get("location_reject_non_remote", []) or []
    return accept, reject


def _location_ok(location: str | None, accept: list[str], reject: list[str]) -> bool:
    """Check if a job location passes the user's accept/reject lists.

    Null / empty location passes through (often means HQ/unspecified).
    """
    if not location:
        return True

    loc = location.lower()

    if any(r in loc for r in _REMOTE_TOKENS):
        return True

    for r in reject:
        if r and r.lower() in loc:
            return False

    for a in accept:
        if a and a.lower() in loc:
            return True

    return False


def country_reject(location: str | None, blocked_countries: list[str]) -> bool:
    """Hard-reject if location string clearly names a blocked country.

    Null/empty location does NOT reject — it's commonly missing or HQ-only.
    """
    if not location or not blocked_countries:
        return False
    loc = location.lower()
    for country in blocked_countries:
        if country and country.lower() in loc:
            return True
    return False


def _parse_years(raw: Any) -> int:
    """Parse years-of-experience from profile — tolerates strings like '7 years' or '10+'."""
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        return int(raw)
    m = re.search(r"\d+", str(raw))
    return int(m.group(0)) if m else 0


def seniority_reject(title: str | None, user_years: int, floor_years: int) -> bool:
    """Reject clearly-junior titles when the user's years-of-experience meets the floor."""
    if not title or user_years < floor_years or floor_years <= 0:
        return False
    return bool(_JUNIOR_TITLE_RE.search(title))


def rule_evaluate(job: dict, search_cfg: dict, profile: dict) -> tuple[bool, str | None]:
    """Apply Layer-2 rule gate to a single job row.

    Returns (ok, reason). `reason` is a short tag intended for the
    `filter_reason` DB column.
    """
    blocked = search_cfg.get("defaults", {}).get("blocked_countries", []) or []
    floor = int(search_cfg.get("defaults", {}).get("seniority_floor_years", 0) or 0)
    user_years = _parse_years(profile.get("experience", {}).get("years_of_experience_total"))

    if seniority_reject(job.get("title"), user_years, floor):
        return False, "seniority_mismatch"

    if country_reject(job.get("location"), blocked):
        return False, "country_blocked"

    return True, None
