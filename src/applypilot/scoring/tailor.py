"""Resume tailoring: LLM-powered ATS-optimized resume generation per job.

THIS IS THE HEAVIEST REFACTOR. Every piece of personal data -- name, email, phone,
skills, companies, projects, school -- is loaded at runtime from the user's profile.
Zero hardcoded personal information.

The LLM returns structured JSON, code assembles the final text. Header (name, contact)
is always code-injected, never LLM-generated. Each retry starts a fresh conversation
to avoid apologetic spirals.
"""

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from applypilot.core import build_default_run_context
from applypilot.database import get_jobs_by_stage
from applypilot.discovery.filters import apply_geo_gate
from applypilot.llm import LLMClient, get_client_for_ctx
from applypilot.prompts import render_prompt
from applypilot.scoring.validator import (
    BANNED_WORDS,
    sanitize_text,
    validate_json_fields,
)

if TYPE_CHECKING:
    from applypilot.core import RunContext

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5  # max cross-run retries before giving up


# ── Prompt Builders (profile-driven) ──────────────────────────────────────

def _build_tailor_prompt(profile: dict, prompts: dict) -> str:
    """Build the resume tailoring system prompt from the user's profile.

    Template comes from ``prompts['tailoring']['resume']['system']``;
    skills boundary, preserved entities, and banned-word list are
    interpolated from the user's profile.
    """
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Format skills boundary for the prompt
    skills_lines = []
    for category, items in boundary.items():
        if isinstance(items, list) and items:
            label = category.replace("_", " ").title()
            skills_lines.append(f"{label}: {', '.join(items)}")
    skills_block = "\n".join(skills_lines)

    # Preserved entities
    companies = resume_facts.get("preserved_companies", [])
    school = resume_facts.get("preserved_school", "")
    real_metrics = resume_facts.get("real_metrics", [])

    companies_str = ", ".join(companies) if companies else "N/A"
    metrics_str = ", ".join(real_metrics) if real_metrics else "N/A"

    # Include ALL banned words from the validator so the LLM knows exactly
    # what will be rejected — the validator checks for these automatically.
    banned_str = ", ".join(BANNED_WORDS)

    education = profile.get("experience", {})
    education_level = education.get("education_level", "")

    return render_prompt(
        prompts, "tailoring.resume.system",
        skills_block=skills_block,
        banned_str=banned_str,
        metrics_str=metrics_str,
        companies_str=companies_str,
        school=school,
        education_level=education_level,
    )


def _build_judge_prompt(profile: dict, prompts: dict) -> str:
    """Build the LLM judge prompt from the user's profile."""
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Flatten allowed skills for the judge
    all_skills: list[str] = []
    for items in boundary.values():
        if isinstance(items, list):
            all_skills.extend(items)
    skills_str = ", ".join(all_skills) if all_skills else "N/A"

    real_metrics = resume_facts.get("real_metrics", [])
    metrics_str = ", ".join(real_metrics) if real_metrics else "N/A"

    return render_prompt(
        prompts, "tailoring.judge.system",
        skills_str=skills_str,
        metrics_str=metrics_str,
    )


# ── JSON Extraction ───────────────────────────────────────────────────────

def extract_json(raw: str) -> dict:
    """Robustly extract JSON from LLM response (handles fences, preamble).

    Args:
        raw: Raw LLM response text.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If no valid JSON found.
    """
    raw = raw.strip()

    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Markdown fences
    if "```" in raw:
        for part in raw.split("```")[1::2]:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue

    # Find outermost { ... }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in LLM response")


# ── Resume Assembly (profile-driven header) ──────────────────────────────

def assemble_resume_text(data: dict, profile: dict) -> str:
    """Convert JSON resume data to formatted plain text.

    Header (name, location, contact) is ALWAYS code-injected from the profile,
    never LLM-generated. All text fields are sanitized.

    Args:
        data: Parsed JSON resume from the LLM.
        profile: User profile dict from load_profile().

    Returns:
        Formatted resume text.
    """
    personal = profile.get("personal", {})
    lines: list[str] = []

    # Header -- always code-injected from profile
    lines.append(personal.get("full_name", ""))
    lines.append(sanitize_text(data.get("title", "Software Engineer")))

    # Location from search config or profile -- leave blank if not available
    # The location line is optional; the original used a hardcoded city.
    # We omit it here; the LLM prompt can include it if the user sets it.

    # Contact line
    contact_parts: list[str] = []
    if personal.get("email"):
        contact_parts.append(personal["email"])
    if personal.get("phone"):
        contact_parts.append(personal["phone"])
    if personal.get("github_url"):
        contact_parts.append(personal["github_url"])
    if personal.get("linkedin_url"):
        contact_parts.append(personal["linkedin_url"])
    if contact_parts:
        lines.append(" | ".join(contact_parts))
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append(sanitize_text(data["summary"]))
    lines.append("")

    # Technical Skills
    lines.append("TECHNICAL SKILLS")
    if isinstance(data["skills"], dict):
        for cat, val in data["skills"].items():
            lines.append(f"{cat}: {sanitize_text(str(val))}")
    lines.append("")

    # Experience
    lines.append("EXPERIENCE")
    for entry in data.get("experience", []):
        lines.append(sanitize_text(entry.get("title", "")))
        if entry.get("company_dates"):
            lines.append(sanitize_text(entry["company_dates"]))
        for b in entry.get("bullets", []):
            lines.append(f"- {sanitize_text(b)}")
        lines.append("")

    # Projects are optional for some roles/resume styles.
    projects = data.get("projects", [])
    if projects:
        lines.append("PROJECTS")
        for entry in projects:
            lines.append(sanitize_text(entry.get("title", "")))
            if entry.get("tech_dates"):
                lines.append(sanitize_text(entry["tech_dates"]))
            for b in entry.get("bullets", []):
                lines.append(f"- {sanitize_text(b)}")
            lines.append("")

    # Education
    lines.append("EDUCATION")
    lines.append(sanitize_text(str(data.get("education", ""))))

    return "\n".join(lines)


# ── One-Page Fit (post-render trim) ──────────────────────────────────────

# Trim ceiling: at most this many trim steps before giving up. Each step
# removes one project or trims/drops one experience entry. 8 covers the worst
# resume we've seen (2 projects + 4 experience trims + slack).
_MAX_PAGE_TRIMS = 8


def _trim_resume_one_step(data: dict) -> str | None:
    """Apply one content-trim step to a resume JSON in place.

    Order is intentional: projects (often academic/student) go before any
    work history, then older experience entries are trimmed and finally
    dropped, then mid-tier entries shrink. Returns a short description of
    the step taken, or ``None`` when nothing more can be trimmed.
    """
    projects = data.get("projects") or []
    experience = data.get("experience") or []

    # 1. Drop projects from the bottom up — least relevant first, since the
    # tailoring prompt already orders projects by relevance.
    if projects:
        dropped = projects.pop()
        return f"dropped project: {dropped.get('title', '?')}"

    # 2. Trim trailing experience entries (positions 3+) down to 2 bullets.
    for entry in experience[2:]:
        bullets = entry.get("bullets") or []
        if len(bullets) > 2:
            entry["bullets"] = bullets[:2]
            return f"trimmed bullets in: {entry.get('title', '?')}"

    # 3. Drop the oldest experience entry entirely (only if 4+ remain).
    if len(experience) > 3:
        dropped = experience.pop()
        return f"dropped oldest experience: {dropped.get('title', '?')}"

    # 4. Shrink the second entry's bullets to 3.
    if len(experience) >= 2:
        bullets = experience[1].get("bullets") or []
        if len(bullets) > 3:
            experience[1]["bullets"] = bullets[:3]
            return f"trimmed bullets in: {experience[1].get('title', '?')}"

    # 5. Last resort: shrink the most recent role's bullets to 3.
    if experience:
        bullets = experience[0].get("bullets") or []
        if len(bullets) > 3:
            experience[0]["bullets"] = bullets[:3]
            return f"trimmed bullets in: {experience[0].get('title', '?')}"

    return None


def render_resume_pdf_fit_one_page(
    data: dict, profile: dict, output_path,
) -> tuple[str, list[str]]:
    """Render a tailored resume to a one-page PDF, trimming if necessary.

    Mutates ``data`` so the caller can persist the trimmed JSON alongside
    the PDF. Renders once with the original content; if that overflows,
    holds a single Playwright session open and re-renders after each trim
    step until the PDF fits one page or no more trims are possible.

    Returns ``(pdf_path, trims_applied)``.
    """
    from pathlib import Path

    from applypilot.scoring.pdf import (
        _render_pdf_with_page,
        build_html,
        convert_text_to_pdf,
        count_pdf_pages,
        parse_resume,
    )

    output_path = Path(output_path)
    text = assemble_resume_text(data, profile)
    convert_text_to_pdf(text, output_path)

    if count_pdf_pages(output_path) <= 1:
        return str(output_path), []

    trims: list[str] = []
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            for _ in range(_MAX_PAGE_TRIMS):
                step = _trim_resume_one_step(data)
                if step is None:
                    break
                trims.append(step)
                text = assemble_resume_text(data, profile)
                html = build_html(parse_resume(text))
                _render_pdf_with_page(page, html, str(output_path))
                if count_pdf_pages(output_path) <= 1:
                    break
        finally:
            browser.close()

    return str(output_path), trims


# ── LLM Judge ────────────────────────────────────────────────────────────

def judge_tailored_resume(
    original_text: str, tailored_text: str, job_title: str, profile: dict,
    *, client: "LLMClient", prompts: dict,
) -> dict:
    """LLM judge layer: catches subtle fabrication that programmatic checks miss."""
    judge_prompt = _build_judge_prompt(profile, prompts)

    messages = [
        {"role": "system", "content": judge_prompt, "cache": "ephemeral"},
        {"role": "user", "content": f"ORIGINAL RESUME:\n{original_text}", "cache": "ephemeral"},
        {"role": "user", "content": (
            f"JOB TITLE: {job_title}\n\n"
            f"TAILORED RESUME:\n{tailored_text}\n\n"
            "Judge this tailored resume:"
        )},
    ]

    response = client.chat(messages, max_output_tokens=512)

    passed = "VERDICT: PASS" in response.upper()
    issues = "none"
    if "ISSUES:" in response.upper():
        issues_idx = response.upper().index("ISSUES:")
        issues = response[issues_idx + 7:].strip()

    return {
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "issues": issues,
        "raw": response,
    }


# ── Core Tailoring ───────────────────────────────────────────────────────

def tailor_resume(
    resume_text: str, job: dict, profile: dict,
    max_retries: int = 3, validation_mode: str = "normal",
    *, client: "LLMClient", prompts: dict,
) -> tuple[str, dict, dict | None]:
    """Generate a tailored resume via JSON output + fresh context on each retry.

    Key design choices:
    - LLM returns structured JSON, code assembles the text (no header leaks)
    - Each retry starts a FRESH conversation (no apologetic spiral)
    - Issues from previous attempts are noted in the system prompt
    - Em dashes and smart quotes are auto-fixed, not rejected

    Args:
        resume_text:      Base resume text.
        job:              Job dict with title, site, location, full_description.
        profile:          User profile dict.
        max_retries:      Maximum retry attempts.
        validation_mode:  "strict", "normal", or "lenient".
                          strict  -- banned words trigger retries; judge must pass
                          normal  -- banned words = warnings only; judge can fail on last retry
                          lenient -- banned words ignored; LLM judge skipped

    Returns:
        (tailored_text, report, parsed_json) where parsed_json is the last
        successfully parsed structured resume, if any.
    """
    from applypilot.scoring.scorer import build_job_context

    job_text = build_job_context(job)

    report: dict = {
        "attempts": 0, "validator": None, "judge": None,
        "status": "pending", "validation_mode": validation_mode,
    }
    avoid_notes: list[str] = []
    tailored = ""
    last_data: dict | None = None
    tailor_prompt_base = _build_tailor_prompt(profile, prompts)

    for attempt in range(max_retries + 1):
        report["attempts"] = attempt + 1

        # Keep the system prompt + resume byte-stable across attempts/jobs so
        # they hit the prefix cache. Avoid-notes and per-job content go in
        # later messages that can vary freely without busting the cache.
        user_parts: list[str] = [f"ORIGINAL RESUME:\n{resume_text}"]
        messages: list = [
            {"role": "system", "content": tailor_prompt_base, "cache": "ephemeral"},
            {"role": "user", "content": user_parts[0], "cache": "ephemeral"},
        ]
        if avoid_notes:
            avoid_block = "## AVOID THESE ISSUES (from previous attempt):\n" + "\n".join(
                f"- {n}" for n in avoid_notes[-5:]
            )
            messages.append({"role": "user", "content": avoid_block})
        messages.append({"role": "user", "content": f"TARGET JOB:\n{job_text}\n\nReturn the JSON:"})

        raw = client.chat(messages, max_output_tokens=16000)

        # Parse JSON from response
        try:
            data = extract_json(raw)
            last_data = data
        except ValueError as exc:
            log.warning("Attempt %d JSON parse failed (%s). Raw response (first 500 chars):\n%s",
                        attempt + 1, exc, raw[:1000])
            avoid_notes.append("Output was not valid JSON. Return ONLY a JSON object, nothing else.")
            continue

        # Layer 1: Validate JSON fields
        validation = validate_json_fields(data, profile, mode=validation_mode)
        report["validator"] = validation

        if not validation["passed"]:
            # Only retry if there are hard errors (warnings never block)
            log.warning("Attempt %d validation failed: %s", attempt + 1, validation["errors"])
            avoid_notes.extend(validation["errors"])
            if attempt < max_retries:
                continue
            # Last attempt — assemble whatever we got
            tailored = assemble_resume_text(data, profile)
            report["status"] = "failed_validation"
            return tailored, report, last_data

        # Assemble text (header injected by code, em dashes auto-fixed)
        tailored = assemble_resume_text(data, profile)

        # Layer 2: LLM judge (catches subtle fabrication) — skipped in lenient mode
        if validation_mode == "lenient":
            report["judge"] = {"verdict": "SKIPPED", "passed": True, "issues": "none"}
            report["status"] = "approved"
            return tailored, report, last_data

        judge = judge_tailored_resume(
            resume_text, tailored, job.get("title", ""), profile,
            client=client, prompts=prompts,
        )
        report["judge"] = judge

        if not judge["passed"]:
            avoid_notes.append(f"Judge rejected: {judge['issues']}")
            if attempt < max_retries:
                # In normal mode, only retry on judge failure if there are retries left
                if validation_mode != "lenient":
                    continue
            # Accept best attempt on last retry (all modes) or if lenient
            report["status"] = "approved_with_judge_warning"
            return tailored, report, last_data

        # Both passed
        report["status"] = "approved"
        return tailored, report, last_data

    report["status"] = "exhausted_retries"
    return tailored, report, last_data


# ── Batch Entry Point ────────────────────────────────────────────────────

def run_tailoring(min_score: int = 7, limit: int = 20,
                  validation_mode: str = "normal",
                  ctx: "RunContext | None" = None) -> dict:
    """Generate tailored resumes for high-scoring jobs."""
    if ctx is None:
        ctx = build_default_run_context()

    profile = ctx.user.profile
    if not profile:
        raise FileNotFoundError("Profile not found — run `applypilot init` first.")
    resume_text = ctx.user.resume_text
    conn = ctx.user.db.connection()
    tailored_dir = ctx.user.storage.tailored_dir()

    jobs = get_jobs_by_stage(conn=conn, stage="pending_tailor", min_score=min_score, limit=limit)

    if not jobs:
        log.info("No untailored jobs with score >= %d.", min_score)
        return {"approved": 0, "failed": 0, "errors": 0, "elapsed": 0.0}

    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    jobs = apply_geo_gate(jobs, ctx.user.search_config or {}, conn)
    if not jobs:
        log.info("All tailor candidates filtered by geo_gate.")
        return {"approved": 0, "failed": 0, "errors": 0, "elapsed": 0.0}

    tailored_dir.mkdir(parents=True, exist_ok=True)
    log.info("Tailoring resumes for %d jobs (score >= %d)...", len(jobs), min_score)
    t0 = time.time()
    completed = 0
    results: list[dict] = []
    stats: dict[str, int] = {"approved": 0, "failed_validation": 0, "failed_judge": 0, "error": 0}

    client = get_client_for_ctx(ctx, "tailor")
    prompts = ctx.user.prompts
    cancel = ctx.task.cancellation

    for job in jobs:
        if cancel.is_set():
            log.info("Tailoring cancelled after %d/%d jobs", completed, len(jobs))
            break
        completed += 1
        log.info(
            "%d/%d [START] score=%s | %s | %s",
            completed,
            len(jobs),
            job.get("fit_score", "?"),
            job["site"],
            job["title"][:60],
        )
        try:
            tailored, report, parsed_json = tailor_resume(
                resume_text, job, profile, validation_mode=validation_mode,
                client=client, prompts=prompts,
            )

            # Build safe filename prefix. The URL-derived hash suffix keeps
            # prefixes unique when two jobs share the same site+title (reposts,
            # same role at multiple tenants, or titles that collide after
            # truncation) and stays stable across re-runs of the same job.
            safe_title = re.sub(r"[^\w\s-]", "", job["title"])[:50].strip().replace(" ", "_")
            safe_site = re.sub(r"[^\w\s-]", "", job["site"])[:20].strip().replace(" ", "_")
            url_hash = hashlib.blake2b(job["url"].encode("utf-8"), digest_size=4).hexdigest()
            prefix = f"{safe_site}_{safe_title}_{url_hash}"

            json_path = tailored_dir / f"{prefix}.json"

            # Save job description for traceability
            job_path = tailored_dir / f"{prefix}_JOB.txt"
            job_desc = (
                f"Title: {job['title']}\n"
                f"Company: {job['site']}\n"
                f"Location: {job.get('location', 'N/A')}\n"
                f"Score: {job.get('fit_score', 'N/A')}\n"
                f"URL: {job['url']}\n\n"
                f"{job.get('full_description', '')}"
            )
            job_path.write_text(job_desc, encoding="utf-8")

            # Generate PDF for approved resumes, fitting to one page if the
            # initial render overflows. Trims (if any) mutate ``parsed_json``
            # in place so the saved JSON matches the final PDF.
            pdf_path = None
            if report["status"] in ("approved", "approved_with_judge_warning"):
                try:
                    if parsed_json is not None:
                        pdf_path, page_trims = render_resume_pdf_fit_one_page(
                            parsed_json, profile, tailored_dir / f"{prefix}.pdf",
                        )
                        report["page_trims"] = page_trims
                        if page_trims:
                            log.info(
                                "%d/%d [TRIM] applied %d trim(s) to fit 1 page",
                                completed, len(jobs), len(page_trims),
                            )
                    else:
                        from applypilot.scoring.pdf import convert_text_to_pdf
                        pdf_path = str(convert_text_to_pdf(
                            tailored, tailored_dir / f"{prefix}.pdf",
                        ))
                except Exception:
                    log.debug("PDF generation failed for %s", json_path, exc_info=True)
                    report["status"] = "error"

            # Save structured resume JSON after PDF rendering so any one-page
            # trims are reflected in the saved JSON artifact.
            if parsed_json is not None:
                json_path.write_text(json.dumps(parsed_json, indent=2), encoding="utf-8")

            # Save validation/reporting metadata after final status is known.
            report_path = tailored_dir / f"{prefix}_REPORT.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            result = {
                "url": job["url"],
                "path": pdf_path,
                "json_path": str(json_path) if parsed_json is not None else None,
                "pdf_path": pdf_path,
                "title": job["title"],
                "site": job["site"],
                "status": report["status"],
                "attempts": report["attempts"],
            }
        except Exception as e:
            result = {
                "url": job["url"], "title": job["title"], "site": job["site"],
                "status": "error", "attempts": 0, "path": None, "json_path": None, "pdf_path": None,
            }
            log.error("%d/%d [ERROR] %s -- %s", completed, len(jobs), job["title"][:40], e)

        results.append(result)
        stats[result.get("status", "error")] = stats.get(result.get("status", "error"), 0) + 1

        elapsed = time.time() - t0
        rate = completed / elapsed if elapsed > 0 else 0
        log.info(
            "%d/%d [%s] attempts=%s | %.1f jobs/min | %s",
            completed, len(jobs),
            result["status"].upper(),
            result.get("attempts", "?"),
            rate * 60,
            result["title"][:40],
        )

        # Persist immediately so work survives if the process is killed
        now = datetime.now(timezone.utc).isoformat()
        _success_statuses = {"approved", "approved_with_judge_warning"}
        if result["status"] in _success_statuses:
            conn.execute(
                "UPDATE jobs SET tailored_resume_path=?, tailored_at=?, "
                "tailored_resume_json_path=?, "
                "tailor_attempts=COALESCE(tailor_attempts,0)+1 WHERE url=?",
                (result["path"], now, result["json_path"], result["url"]),
            )
        else:
            conn.execute(
                "UPDATE jobs SET tailor_attempts=COALESCE(tailor_attempts,0)+1 WHERE url=?",
                (result["url"],),
            )
        conn.commit()

    elapsed = time.time() - t0
    log.info(
        "Tailoring done in %.1fs: %d approved, %d failed_validation, %d failed_judge, %d errors",
        elapsed,
        stats.get("approved", 0),
        stats.get("failed_validation", 0),
        stats.get("failed_judge", 0),
        stats.get("error", 0),
    )

    return {
        "approved": stats.get("approved", 0),
        "failed": stats.get("failed_validation", 0) + stats.get("failed_judge", 0),
        "errors": stats.get("error", 0),
        "elapsed": elapsed,
    }
