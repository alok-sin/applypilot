"""Cover letter generation: LLM-powered, profile-driven, with validation.

Generates concise, engineering-voice cover letters tailored to specific job
postings. All personal data (name, skills, achievements) comes from the user's
profile at runtime. No hardcoded personal information.
"""

import hashlib
import logging
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from applypilot.core import build_default_run_context
from applypilot.discovery.filters import apply_geo_gate
from applypilot.llm import LLMClient, get_client_for_ctx
from applypilot.prompts import render_prompt
from applypilot.scoring.validator import (
    BANNED_WORDS,
    LLM_LEAK_PHRASES,
    sanitize_text,
    validate_cover_letter,
)

if TYPE_CHECKING:
    from applypilot.core import RunContext

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5  # max cross-run retries before giving up


def _load_tailored_resume_text(job: dict, profile: dict, fallback_resume: str) -> str:
    """Load tailored resume text from JSON when available, else fall back safely."""
    json_path = job.get("tailored_resume_json_path")
    if json_path and Path(json_path).exists():
        from applypilot.scoring.tailor import assemble_resume_text

        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        return assemble_resume_text(data, profile)

    tailored_path = job.get("tailored_resume_path")
    if tailored_path and Path(tailored_path).suffix == ".txt" and Path(tailored_path).exists():
        return Path(tailored_path).read_text(encoding="utf-8")

    return fallback_resume


# ── Prompt Builder (profile-driven) ──────────────────────────────────────

def _build_cover_letter_prompt(profile: dict, prompts: dict) -> str:
    """Build the cover letter system prompt from the user's profile.

    Template comes from ``prompts['cover_letter']['generate']['system']``;
    personal data, skills, banned-word list, and sign-off name are
    interpolated from the user's profile.
    """
    personal = profile.get("personal", {})
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Preferred name for the sign-off (falls back to full name)
    sign_off_name = personal.get("preferred_name") or personal.get("full_name", "")

    # Flatten all allowed skills
    all_skills: list[str] = []
    for items in boundary.values():
        if isinstance(items, list):
            all_skills.extend(items)
    skills_str = ", ".join(all_skills) if all_skills else "the tools listed in the resume"

    # Real metrics from resume_facts
    real_metrics = resume_facts.get("real_metrics", [])
    preserved_projects = resume_facts.get("preserved_projects", [])

    # Build achievement examples for the prompt -- leading newline so the hint
    # sits on its own line when present and disappears entirely when empty.
    projects_hint = ""
    if preserved_projects:
        projects_hint = f"\nKnown projects to reference: {', '.join(preserved_projects)}"

    metrics_hint = ""
    if real_metrics:
        metrics_hint = f"\nReal metrics to use: {', '.join(real_metrics)}"

    # Build the full banned list from the validator so the prompt stays in sync
    # with what will actually be rejected — the validator checks all of these.
    all_banned = ", ".join(f'"{w}"' for w in BANNED_WORDS)
    leak_banned = ", ".join(f'"{p}"' for p in LLM_LEAK_PHRASES)

    return render_prompt(
        prompts, "cover_letter.generate.system",
        sign_off_name=sign_off_name,
        projects_hint=projects_hint,
        metrics_hint=metrics_hint,
        all_banned=all_banned,
        leak_banned=leak_banned,
        skills_str=skills_str,
    )


# ── Helpers ──────────────────────────────────────────────────────────────

def _strip_preamble(text: str) -> str:
    """Remove LLM preamble before 'Dear Hiring Manager,' if present.

    Gemini and other models sometimes output "Here is the cover letter:" or
    similar meta-commentary before the actual letter text. Strip everything
    before the first occurrence of "Dear" so the validator's start-check passes.
    """
    dear_idx = text.lower().find("dear")
    if dear_idx > 0:
        return text[dear_idx:]
    return text


# ── Core Generation ──────────────────────────────────────────────────────

def generate_cover_letter(
    resume_text: str, job: dict, profile: dict,
    max_retries: int = 3, validation_mode: str = "normal",
    *, client: "LLMClient", prompts: dict,
) -> str:
    """Generate a cover letter with fresh context on each retry + auto-sanitize.

    Same design as tailor_resume: fresh conversation per attempt, issues noted
    in the prompt, no conversation history stacking.

    Args:
        resume_text:      The candidate's resume text (base or tailored).
        job:              Job dict with title, site, location, full_description.
        profile:          User profile dict.
        max_retries:      Maximum retry attempts.
        validation_mode:  "strict", "normal", or "lenient".

    Returns:
        The cover letter text (best attempt even if validation failed).
    """
    from applypilot.scoring.scorer import build_job_context

    job_text = build_job_context(job)

    avoid_notes: list[str] = []
    letter = ""
    cl_prompt_base = _build_cover_letter_prompt(profile, prompts)

    for attempt in range(max_retries + 1):
        # System prompt + resume stay byte-stable so they hit the prefix cache
        # across every cover-letter call. Avoid-notes and job text vary.
        messages: list = [
            {"role": "system", "content": cl_prompt_base, "cache": "ephemeral"},
            {"role": "user", "content": f"RESUME:\n{resume_text}", "cache": "ephemeral"},
        ]
        if avoid_notes:
            avoid_block = "## AVOID THESE ISSUES:\n" + "\n".join(
                f"- {n}" for n in avoid_notes[-5:]
            )
            messages.append({"role": "user", "content": avoid_block})
        messages.append({"role": "user", "content": (
            f"TARGET JOB:\n{job_text}\n\nWrite the cover letter:"
        )})

        letter = client.chat(messages, max_output_tokens=10000)
        letter = sanitize_text(letter)  # auto-fix em dashes, smart quotes
        letter = _strip_preamble(letter)  # remove any "Here is the letter:" prefix

        validation = validate_cover_letter(letter, mode=validation_mode)
        if validation["passed"]:
            return letter

        avoid_notes.extend(validation["errors"])
        # Warnings never block — only hard errors trigger a retry
        log.debug(
            "Cover letter attempt %d/%d failed: %s",
            attempt + 1, max_retries + 1, validation["errors"],
        )

    return letter  # last attempt even if failed


# ── Batch Entry Point ────────────────────────────────────────────────────

def run_cover_letters(min_score: int = 7, limit: int = 20,
                      validation_mode: str = "normal",
                      ctx: "RunContext | None" = None) -> dict:
    """Generate cover letters for high-scoring jobs that have tailored resumes.

    Args:
        min_score:       Minimum fit_score threshold.
        limit:           Maximum jobs to process.
        validation_mode: "strict", "normal", or "lenient".
        ctx:             Optional :class:`RunContext`. When ``None`` a
            CLI-default context is built from ``APP_DIR``.

    Returns:
        {"generated": int, "errors": int, "elapsed": float}
    """
    if ctx is None:
        ctx = build_default_run_context()

    profile = ctx.user.profile or {}
    resume_text = ctx.user.resume_text
    if not resume_text:
        log.error("Resume text is empty. Run 'applypilot init' first.")
        return {"generated": 0, "errors": 0, "elapsed": 0.0}
    conn = ctx.user.db.connection()
    cover_dir = ctx.user.storage.cover_letter_dir()

    # Fetch jobs that have tailored resumes but no cover letter yet.
    # limit <= 0 means no cap — process every pending job.
    query = (
        "SELECT * FROM jobs "
        "WHERE fit_score >= ? AND tailored_resume_path IS NOT NULL "
        "AND full_description IS NOT NULL "
        "AND (cover_letter_path IS NULL OR cover_letter_path = '') "
        "AND COALESCE(cover_attempts, 0) < ? "
        "ORDER BY fit_score DESC"
    )
    params: list = [min_score, MAX_ATTEMPTS]
    if limit > 0:
        query += " LIMIT ?"
        params.append(limit)
    jobs = conn.execute(query, params).fetchall()

    if not jobs:
        log.info("No jobs needing cover letters (score >= %d).", min_score)
        return {"generated": 0, "errors": 0, "elapsed": 0.0}

    # Convert rows to dicts
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    jobs = apply_geo_gate(jobs, ctx.user.search_config or {}, conn)
    if not jobs:
        log.info("All cover-letter candidates filtered by geo_gate.")
        return {"generated": 0, "errors": 0, "elapsed": 0.0}

    cover_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Generating cover letters for %d jobs (score >= %d)...",
        len(jobs), min_score,
    )
    t0 = time.time()
    completed = 0
    results: list[dict] = []
    error_count = 0
    saved = 0

    client = get_client_for_ctx(ctx, "cover")
    prompts = ctx.user.prompts
    cancel = ctx.task.cancellation

    for job in jobs:
        if cancel.is_set():
            log.info("Cover-letter generation cancelled after %d/%d jobs", completed, len(jobs))
            break
        completed += 1
        try:
            job_resume = _load_tailored_resume_text(job, profile, resume_text)
            letter = generate_cover_letter(
                job_resume, job, profile, validation_mode=validation_mode,
                client=client, prompts=prompts,
            )

            # Build safe filename prefix. The URL-derived hash suffix keeps
            # prefixes unique when two jobs share the same site+title and stays
            # stable across re-runs of the same job.
            safe_title = re.sub(r"[^\w\s-]", "", job["title"])[:50].strip().replace(" ", "_")
            safe_site = re.sub(r"[^\w\s-]", "", job["site"])[:20].strip().replace(" ", "_")
            url_hash = hashlib.blake2b(job["url"].encode("utf-8"), digest_size=4).hexdigest()
            prefix = f"{safe_site}_{safe_title}_{url_hash}"

            cl_path = cover_dir / f"{prefix}_CL.txt"
            cl_path.write_text(letter, encoding="utf-8")

            # Generate PDF (best-effort)
            pdf_path = None
            try:
                from applypilot.scoring.pdf import convert_to_pdf
                pdf_path = str(convert_to_pdf(cl_path))
            except Exception:
                log.debug("PDF generation failed for %s", cl_path, exc_info=True)

            result = {
                "url": job["url"],
                "path": str(cl_path),
                "pdf_path": pdf_path,
                "title": job["title"],
                "site": job["site"],
            }
            results.append(result)
            saved += 1

            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            log.info(
                "%d/%d [OK] | %.1f jobs/min | %s",
                completed, len(jobs), rate * 60, result["title"][:40],
            )
        except Exception as e:
            result = {
                "url": job["url"], "title": job["title"], "site": job["site"],
                "path": None, "pdf_path": None, "error": str(e),
            }
            error_count += 1
            results.append(result)
            log.error("%d/%d [ERROR] %s -- %s", completed, len(jobs), job["title"][:40], e)

        # Persist immediately so work survives if the process is killed
        now = datetime.now(timezone.utc).isoformat()
        if result.get("path"):
            conn.execute(
                "UPDATE jobs SET cover_letter_path=?, cover_letter_at=?, "
                "cover_attempts=COALESCE(cover_attempts,0)+1 WHERE url=?",
                (result["path"], now, result["url"]),
            )
        else:
            conn.execute(
                "UPDATE jobs SET cover_attempts=COALESCE(cover_attempts,0)+1 WHERE url=?",
                (result["url"],),
            )
        conn.commit()

    elapsed = time.time() - t0
    log.info("Cover letters done in %.1fs: %d generated, %d errors", elapsed, saved, error_count)

    return {
        "generated": saved,
        "errors": error_count,
        "elapsed": elapsed,
    }
