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

from typing import TYPE_CHECKING

from applypilot.core import build_default_run_context
from applypilot.llm import get_client_for_ctx
from applypilot.prompts import render_prompt

if TYPE_CHECKING:
    from applypilot.core import RunContext

log = logging.getLogger(__name__)

_VALID_REASONS = {"seniority_mismatch", "country_mismatch", "expired"}


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


def format_geo_context(search_cfg: dict, profile: dict) -> str:
    """Build a reusable geo block for LLM prompts.

    Handles single-country, multi-country, and remote-permissive profiles by
    surfacing both the accept list and the reject/blocked list verbatim so the
    model can reason about ambiguous "remote - <country>" strings.
    """
    accept = search_cfg.get("location_accept", []) or []
    reject = search_cfg.get("location_reject_non_remote", []) or []
    blocked = search_cfg.get("defaults", {}).get("blocked_countries", []) or []
    remote_locs = [
        l for l in search_cfg.get("locations", []) or []
        if isinstance(l, dict) and l.get("remote")
    ]
    country = ((profile.get("personal") or {}).get("country")) or "?"

    lines = [f"CANDIDATE COUNTRY: {country}"]
    if accept:
        lines.append(f"ACCEPTS LOCATIONS: {', '.join(accept)}")
    rej_all = list(reject) + [b for b in blocked if b not in reject]
    if rej_all:
        lines.append(f"REJECTS LOCATIONS: {', '.join(rej_all)}")
    if remote_locs:
        lines.append("REMOTE WORK: acceptable, but only if NOT based in a rejected location")
    else:
        lines.append(
            "REMOTE WORK: acceptable only if the role is explicitly remote AND not "
            "restricted to a rejected country"
        )
    return "\n".join(lines)


def _build_user_msg(job: dict, profile: dict, search_cfg: dict) -> str:
    exp = profile.get("experience", {}) or {}
    years = exp.get("years_of_experience_total") or "?"
    target = exp.get("target_role") or "?"
    desc = (job.get("description") or "")[:800]
    geo = format_geo_context(search_cfg, profile)
    return (
        f"{geo}\n\n"
        f"CANDIDATE: {years} yrs experience, target role: {target}.\n\n"
        f"JOB:\n"
        f"TITLE: {job.get('title') or '?'}\n"
        f"LOCATION: {job.get('location') or '(unspecified)'}\n"
        f"SHORT DESCRIPTION: {desc}"
    )


def run_llm_sweep(limit: int = 0, ctx: "RunContext | None" = None) -> dict:
    """Classify pending discovered rows via a tiny LLM call; mark rejects.

    Processes rows where `filter_reason IS NULL AND detail_scraped_at IS NULL
    AND prefiltered_at IS NULL`. Always stamps `prefiltered_at`; on REJECT
    also writes `filter_reason`.

    Returns: {"checked": n, "rejected": n, "errors": n, "elapsed": s}
    """
    if ctx is None:
        ctx = build_default_run_context()
    search_cfg = ctx.user.search_config or {}
    defaults = search_cfg.get("defaults", {}) or {}
    if not defaults.get("llm_sweep_enabled", True):
        log.info("LLM sweep disabled (defaults.llm_sweep_enabled=false)")
        return {"checked": 0, "rejected": 0, "errors": 0, "elapsed": 0.0, "skipped": True}

    profile = ctx.user.profile or {}
    if not profile:
        log.warning("No profile found — LLM sweep needs a profile to compare against; skipping.")
        return {"checked": 0, "rejected": 0, "errors": 0, "elapsed": 0.0, "skipped": True}

    conn = ctx.user.db.connection()
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

    client = get_client_for_ctx(ctx, "prefilter")
    sweep_system = render_prompt(ctx.user.prompts, "prefilter.sweep.system")
    log.info("LLM sweep: classifying %d candidate(s) via task=prefilter", len(rows))

    t0 = time.time()
    checked = rejected = errors = 0

    for row in rows:
        job = dict(row) if not isinstance(row, dict) else row
        messages = [
            {"role": "system", "content": sweep_system, "cache": "ephemeral"},
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
