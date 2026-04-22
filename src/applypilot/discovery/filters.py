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
column: "seniority_mismatch" | "country_mismatch" | "excluded_title" |
"title_not_allowed" | None.
"""

from __future__ import annotations

import logging
import re
import sqlite3
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

    for r in reject:
        if r and r.lower() in loc:
            return False

    if any(r in loc for r in _REMOTE_TOKENS):
        return True

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


def title_excluded(title: str | None, exclude_titles: list[str]) -> bool:
    """Reject if title contains any user-defined exclude phrase (case-insensitive)."""
    if not title or not exclude_titles:
        return False
    t = title.lower()
    return any(phrase and phrase.lower() in t for phrase in exclude_titles)


def title_require_any(title: str | None, required: list[str]) -> bool:
    """True if title contains at least one required phrase (case-insensitive).

    - required absent/empty → True (feature disabled)
    - null/empty title → True (lenient, matches title_excluded / country_reject)
    """
    if not required:
        return True
    if not title:
        return True
    t = title.lower()
    return any(phrase and phrase.lower() in t for phrase in required)


def rule_evaluate(job: dict, search_cfg: dict, profile: dict) -> tuple[bool, str | None]:
    """Apply Layer-2 rule gate to a single job row.

    Returns (ok, reason). `reason` is a short tag intended for the
    `filter_reason` DB column.
    """
    blocked = search_cfg.get("defaults", {}).get("blocked_countries", []) or []
    floor = int(search_cfg.get("defaults", {}).get("seniority_floor_years", 0) or 0)
    excludes = search_cfg.get("exclude_titles", []) or []
    required = search_cfg.get("title_require_any", []) or []
    user_years = _parse_years(profile.get("experience", {}).get("years_of_experience_total"))

    if title_excluded(job.get("title"), excludes):
        return False, "excluded_title"

    if not title_require_any(job.get("title"), required):
        return False, "title_not_allowed"

    if seniority_reject(job.get("title"), user_years, floor):
        return False, "seniority_mismatch"

    ok, reason = geo_gate(job, search_cfg)
    if not ok:
        return False, reason

    return True, None


def geo_gate(job: dict, search_cfg: dict) -> tuple[bool, str | None]:
    """Deterministic geography check.

    Combines `defaults.blocked_countries` + `location_accept`/`location_reject_non_remote`.
    Returns (True, None) if ok, else (False, 'country_mismatch'). Reused by
    every pipeline stage so geo rejection is deterministic, not LLM-dependent.
    """
    blocked = search_cfg.get("defaults", {}).get("blocked_countries", []) or []
    if country_reject(job.get("location"), blocked):
        return False, "country_mismatch"
    accept, reject = _load_location_filter(search_cfg)
    if not _location_ok(job.get("location"), accept, reject):
        return False, "country_mismatch"
    return True, None


def apply_geo_gate(
    jobs: list[dict], search_cfg: dict, conn: sqlite3.Connection
) -> list[dict]:
    """Soft-mark geo-mismatched rows and return the survivors.

    Used by scorer/tailor/cover to skip LLM spend on jobs that fail the
    deterministic geography check.
    """
    survivors: list[dict] = []
    skipped = 0
    for job in jobs:
        ok, reason = geo_gate(job, search_cfg)
        if not ok:
            conn.execute(
                "UPDATE jobs SET filter_reason = ?, prefiltered_at = CURRENT_TIMESTAMP "
                "WHERE url = ? AND filter_reason IS NULL",
                (reason, job["url"]),
            )
            skipped += 1
        else:
            survivors.append(job)
    if skipped:
        conn.commit()
        log.info("geo_gate: skipped %d/%d jobs (country_mismatch)", skipped, len(jobs))
    return survivors


def apply_rule_gate(
    jobs: list[dict], search_cfg: dict, profile: dict, conn: sqlite3.Connection
) -> list[dict]:
    """Run the full rule gate across `jobs`, soft-marking rejects and returning survivors.

    Unlike `_run_rule_gate` in the pipeline, this operates on any caller-supplied
    list — enrichment uses it to catch rows the discovery-time gate never saw
    (e.g. rescrape, standalone `enrich`, or rows that were enriched before the
    gate tightened).
    """
    survivors: list[dict] = []
    by_reason: dict[str, int] = {}
    for job in jobs:
        ok, reason = rule_evaluate(job, search_cfg, profile)
        if not ok:
            conn.execute(
                "UPDATE jobs SET filter_reason = ?, prefiltered_at = CURRENT_TIMESTAMP "
                "WHERE url = ? AND filter_reason IS NULL",
                (reason, job["url"]),
            )
            by_reason[reason or "unknown"] = by_reason.get(reason or "unknown", 0) + 1
        else:
            survivors.append(job)
    if by_reason:
        conn.commit()
        summary = ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items()))
        log.info("rule_gate: skipped %d/%d jobs (%s)", sum(by_reason.values()), len(jobs), summary)
    return survivors
