"""Job fit scoring: LLM-powered evaluation of candidate-job match quality.

Scores jobs on a 1-10 scale by comparing the user's resume against each
job description. All personal data is loaded at runtime from the user's
profile and resume file.
"""

import logging
import re
import time
from datetime import datetime, timezone

from applypilot import config
from applypilot.config import RESUME_PATH
from applypilot.database import get_connection, get_jobs_by_stage, mark_filtered
from applypilot.discovery.filters import apply_rule_gate
from applypilot.llm import get_client

log = logging.getLogger(__name__)

_THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL | re.IGNORECASE)


# ── Scoring Prompt ────────────────────────────────────────────────────────

SCORE_PROMPT = """You are a job fit evaluator. Score how well a candidate fits a role.

SCORING SCALE:
9-10 = Perfect match, 7-8 = Strong, 5-6 = Moderate, 3-4 = Weak, 1-2 = Poor.

Weight technical skills heavily. Consider transferable experience. Be realistic about experience level vs. requirements.

OUTPUT ONLY these 3 lines — nothing else, no preamble, no analysis:
SCORE: [1-10]
KEYWORDS: [comma-separated ATS keywords that match the candidate]
REASONING: [2-3 sentences explaining the score]"""


def _parse_score_response(response: str) -> dict:
    """Parse the LLM's score response into structured data.

    Args:
        response: Raw LLM response text.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    score = 0
    keywords = ""
    reasoning = response

    # Strip <thought>...</thought> blocks (emitted by models like Gemma 4)
    # before parsing — the closing tag may be on the same line as SCORE:
    clean = _THOUGHT_RE.sub("", response).strip()
    log.debug("Cleaned LLM response for parsing:\n%s", clean)

    # Use regex on the full text to find the LAST occurrence of each field.
    # Models like Gemma may echo the prompt, reason at length, then output
    # the final answer — sometimes without a newline before SCORE:.
    score_matches = re.findall(r"SCORE:\s*(\d+)", clean)
    if score_matches:
        score = max(1, min(10, int(score_matches[-1])))

    kw_matches = re.findall(r"KEYWORDS:\s*(.+)", clean)
    if kw_matches:
        keywords = kw_matches[-1].strip()

    reason_matches = re.findall(r"REASONING:\s*(.+)", clean)
    if reason_matches:
        reasoning = reason_matches[-1].strip()

    return {"score": score, "keywords": keywords, "reasoning": reasoning}


def parse_stored_reasoning(job: dict) -> tuple[str, str]:
    """Extract keywords and reasoning from the stored score_reasoning field.

    Handles both tagged format (``KEYWORDS: ...\nREASONING: ...``) and
    legacy format (``keywords\nreasoning``).

    Returns:
        (keywords, reasoning) — either may be empty.
    """
    raw = job.get("score_reasoning") or ""
    # Tagged format
    kw_match = re.search(r"KEYWORDS:\s*(.+)", raw)
    re_match = re.search(r"REASONING:\s*(.+)", raw)
    if kw_match and re_match:
        return kw_match.group(1).strip(), re_match.group(1).strip()
    # Legacy format: first line = keywords, rest = reasoning
    parts = raw.split("\n", 1)
    return (parts[0].strip(), parts[1].strip() if len(parts) > 1 else "")


def build_job_context(job: dict, max_desc_chars: int = 6000) -> str:
    """Build job context for LLM prompts, enriched with scorer output when available."""
    header = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n"
    )

    keywords, reasoning = parse_stored_reasoning(job)
    score = job.get("fit_score")

    if keywords and reasoning and score:
        return (
            f"{header}"
            f"FIT SCORE: {score}/10\n"
            f"MATCHED KEYWORDS: {keywords}\n"
            f"FIT ANALYSIS: {reasoning}\n\n"
            f"DESCRIPTION:\n{(job.get('full_description') or '')[:max_desc_chars]}"
        )

    return f"{header}\nDESCRIPTION:\n{(job.get('full_description') or '')[:max_desc_chars]}"


def score_job(resume_text: str, job: dict) -> dict:
    """Score a single job against the resume.

    Args:
        resume_text: The candidate's full resume text.
        job: Job dict with keys: title, site, location, full_description.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    # Split so the system prompt + resume form stable cached blocks across
    # every score call; the per-job posting is the only varying segment.
    messages = [
        {"role": "system", "content": SCORE_PROMPT, "cache": "ephemeral"},
        {"role": "user", "content": f"RESUME:\n{resume_text}", "cache": "ephemeral"},
        {"role": "user", "content": f"JOB POSTING:\n{job_text}"},
    ]

    try:
        client = get_client("score")
        response = client.chat(messages, max_output_tokens=1024)
        return _parse_score_response(response)
    except Exception as e:
        log.error("LLM error scoring job '%s': %s", job.get("title", "?"), e)
        return {"score": 0, "keywords": "", "reasoning": f"LLM error: {e}"}


def run_scoring(
    limit: int = 0,
    rescore: bool = False,
    rescore_above: int | None = None,
) -> dict:
    """Score unscored jobs that have full descriptions.

    Args:
        limit: Maximum number of jobs to score in this run.
        rescore: If True, re-score all jobs (not just unscored ones).
        rescore_above: If set, re-score only jobs with fit_score >= this value.
            Takes precedence over `rescore`.

    Returns:
        {"scored": int, "errors": int, "elapsed": float, "distribution": list}
    """
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    if rescore_above is not None:
        query = (
            "SELECT * FROM jobs WHERE full_description IS NOT NULL "
            "AND fit_score IS NOT NULL AND fit_score >= ? "
            "AND filter_reason IS NULL "
            "ORDER BY fit_score DESC"
        )
        params: list = [rescore_above]
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        jobs = conn.execute(query, params).fetchall()
        log.info("Rescoring %d job(s) with fit_score >= %d", len(jobs), rescore_above)
    elif rescore:
        query = "SELECT * FROM jobs WHERE full_description IS NOT NULL AND filter_reason IS NULL"
        if limit > 0:
            query += f" LIMIT {limit}"
        jobs = conn.execute(query).fetchall()
    else:
        jobs = get_jobs_by_stage(conn=conn, stage="pending_score", limit=limit)

    if not jobs:
        log.info("No unscored jobs with descriptions found.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0, "distribution": []}

    # Convert sqlite3.Row to dicts if needed
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    search_cfg = config.load_search_config() or {}
    try:
        profile = config.load_profile()
    except FileNotFoundError:
        profile = {}
    jobs = apply_rule_gate(jobs, search_cfg, profile, conn)
    if not jobs:
        log.info("All candidate jobs filtered by rule gate; nothing to score.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0, "distribution": []}

    log.info("Scoring %d jobs sequentially...", len(jobs))
    t0 = time.time()
    completed = 0
    errors = 0
    results: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()

    for job in jobs:
        result = score_job(resume_text, job)
        result["url"] = job["url"]
        completed += 1

        if result["score"] == 0:
            errors += 1

        results.append(result)

        log.info(
            "[%d/%d] score=%d  %s",
            completed, len(jobs), result["score"], job.get("title", "?")[:60],
        )

        conn.execute(
            "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ? WHERE url = ?",
            (result["score"], f"KEYWORDS: {result['keywords']}\nREASONING: {result['reasoning']}", now, result["url"]),
        )
        # Progressive filter: a fit score of 1-2 is an unambiguous mismatch.
        # Hard-filter it so downstream stages don't reconsider if min_score is lowered.
        if 1 <= result["score"] <= 2:
            mark_filtered(result["url"], "low_fit", conn=conn)
        conn.commit()

    elapsed = time.time() - t0
    log.info("Done: %d scored in %.1fs (%.1f jobs/sec)", len(results), elapsed, len(results) / elapsed if elapsed > 0 else 0)

    # Score distribution
    dist = conn.execute("""
        SELECT fit_score, COUNT(*) FROM jobs
        WHERE fit_score IS NOT NULL
        GROUP BY fit_score ORDER BY fit_score DESC
    """).fetchall()
    distribution = [(row[0], row[1]) for row in dist]

    return {
        "scored": len(results),
        "errors": errors,
        "elapsed": elapsed,
        "distribution": distribution,
    }
