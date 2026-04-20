"""Cheap per-job LLM classifier — Layer 3 of discover-phase filtering.

Runs at the end of `_run_discover` after the rule gate. One small structured
prompt per surviving row decides PASS/REJECT on `title + location + short
description` using the user's seniority/country/target-role profile.

Uses the `"prefilter"` LLM task — add a `tasks.prefilter` entry in
~/.applypilot/llm.yaml to point this at a cheap/local model (Gemma 4 31B IT,
DeepSeek v3.1, Gemini Flash). If no entry exists, falls back to the default
LLM client.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

from applypilot import config
from applypilot.database import get_connection
from applypilot.llm import get_client

log = logging.getLogger(__name__)

_VALID_REASONS = {"seniority_mismatch", "country_mismatch", "expired"}

SWEEP_SYSTEM = """You are filtering job postings for a candidate.

Reject ONLY if clearly true from the text below:
- seniority_mismatch: role is clearly junior/entry-level/intern vs the candidate's seniority
- country_mismatch:   role is in a country/region the candidate can't work in
- expired:            text explicitly says closed, filled, on hold, or posting is old

If unsure, PASS. Do not reject for minor concerns.

Output exactly two lines, nothing else:
VERDICT: PASS
REASON: ok

or:
VERDICT: REJECT
REASON: <seniority_mismatch|country_mismatch|expired>"""


_LINE_RE = re.compile(r"^\s*(VERDICT|REASON)\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _parse_verdict(text: str) -> tuple[str, str]:
    """Parse LLM output into (verdict, reason). Defaults to PASS on parse failure."""
    verdict = "PASS"
    reason = "ok"
    for m in _LINE_RE.finditer(text):
        key = m.group(1).upper()
        val = m.group(2).strip()
        if key == "VERDICT":
            verdict = "REJECT" if val.upper().startswith("REJECT") else "PASS"
        elif key == "REASON":
            reason = val.lower().split()[0] if val else "ok"
    if verdict == "REJECT" and reason not in _VALID_REASONS:
        # Unknown reason → treat as safety-net pass (avoid mis-classifying).
        return "PASS", "ok"
    return verdict, reason


def _build_user_msg(job: dict, profile: dict, search_cfg: dict) -> str:
    personal = profile.get("personal", {}) or {}
    exp = profile.get("experience", {}) or {}
    years = exp.get("years_of_experience_total") or "?"
    target = exp.get("target_role") or "?"
    country = personal.get("country") or "?"
    remote_ok = "yes" if any(bool(l.get("remote")) for l in search_cfg.get("locations", []) if isinstance(l, dict)) else "unknown"
    desc = (job.get("description") or "")[:800]
    return (
        f"CANDIDATE: {years} yrs experience, target role: {target}, country: {country}, remote ok: {remote_ok}.\n\n"
        f"JOB:\n"
        f"TITLE: {job.get('title') or '?'}\n"
        f"LOCATION: {job.get('location') or '(unspecified)'}\n"
        f"SHORT DESCRIPTION: {desc}"
    )


def run_llm_sweep(limit: int = 0) -> dict:
    """Classify pending discovered rows via a tiny LLM call; mark rejects.

    Processes rows where `filter_reason IS NULL AND detail_scraped_at IS NULL
    AND prefiltered_at IS NULL`. Always stamps `prefiltered_at`; on REJECT
    also writes `filter_reason`.

    Returns: {"checked": n, "rejected": n, "errors": n, "elapsed": s}
    """
    search_cfg = config.load_search_config() or {}
    defaults = search_cfg.get("defaults", {}) or {}
    if not defaults.get("llm_sweep_enabled", True):
        log.info("LLM sweep disabled (defaults.llm_sweep_enabled=false)")
        return {"checked": 0, "rejected": 0, "errors": 0, "elapsed": 0.0, "skipped": True}

    try:
        profile = config.load_profile()
    except FileNotFoundError:
        log.warning("No profile found — LLM sweep needs a profile to compare against; skipping.")
        return {"checked": 0, "rejected": 0, "errors": 0, "elapsed": 0.0, "skipped": True}

    conn = get_connection()
    query = (
        "SELECT url, title, location, description FROM jobs "
        "WHERE filter_reason IS NULL AND detail_scraped_at IS NULL AND prefiltered_at IS NULL"
    )
    params: list = []
    if limit > 0:
        query += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(query, params).fetchall()

    if not rows:
        log.info("LLM sweep: no pending rows.")
        return {"checked": 0, "rejected": 0, "errors": 0, "elapsed": 0.0}

    client = get_client("prefilter")
    log.info("LLM sweep: classifying %d candidate(s) via task=prefilter", len(rows))

    t0 = time.time()
    checked = rejected = errors = 0

    for row in rows:
        job = dict(row) if not isinstance(row, dict) else row
        messages = [
            {"role": "system", "content": SWEEP_SYSTEM, "cache": "ephemeral"},
            {"role": "user", "content": _build_user_msg(job, profile, search_cfg)},
        ]
        try:
            text = client.chat(messages, max_output_tokens=64)
            verdict, reason = _parse_verdict(text)
        except Exception as exc:
            log.warning("LLM sweep error on %s: %s", job.get("url"), exc)
            errors += 1
            # Stamp prefiltered_at so we don't keep retrying; leave filter_reason NULL.
            conn.execute(
                "UPDATE jobs SET prefiltered_at = ? WHERE url = ?",
                (datetime.now(timezone.utc).isoformat(), job["url"]),
            )
            conn.commit()
            continue

        checked += 1
        now = datetime.now(timezone.utc).isoformat()
        if verdict == "REJECT":
            rejected += 1
            conn.execute(
                "UPDATE jobs SET filter_reason = ?, prefiltered_at = ? WHERE url = ?",
                (reason, now, job["url"]),
            )
        else:
            conn.execute(
                "UPDATE jobs SET prefiltered_at = ? WHERE url = ?",
                (now, job["url"]),
            )
        conn.commit()

    elapsed = time.time() - t0
    log.info(
        "LLM sweep done: checked=%d rejected=%d errors=%d (%.1fs)",
        checked, rejected, errors, elapsed,
    )
    return {"checked": checked, "rejected": rejected, "errors": errors, "elapsed": elapsed}
