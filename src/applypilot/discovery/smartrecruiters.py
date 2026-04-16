"""SmartRecruiters ATS discovery: fetches jobs from SmartRecruiters public API.

SmartRecruiters is used by hundreds of global companies (Bosch, Visa, ServiceNow,
Accor, Freshworks, etc.). Uses the official public API:
https://api.smartrecruiters.com/v1/companies/{slug}/postings
"""

import logging
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import httpx
import yaml

from applypilot import config
from applypilot.config import APP_DIR, CONFIG_DIR
from applypilot.database import get_connection

log = logging.getLogger(__name__)

SMARTRECRUITERS_API_BASE = "https://api.smartrecruiters.com/v1/companies"
PAGE_SIZE = 100


def load_employers() -> dict:
    """Load SmartRecruiters employer registry.

    Tries user config first (~/.applypilot/smartrecruiters.yaml),
    falls back to package config.
    """
    user_path = APP_DIR / "smartrecruiters.yaml"
    if user_path.exists():
        log.info("Loading user SmartRecruiters config from %s", user_path)
        try:
            data = yaml.safe_load(user_path.read_text(encoding="utf-8"))
            if data and "employers" in data:
                return data.get("employers", {})
        except Exception as e:
            log.warning("Failed to load user config: %s", e)

    package_path = CONFIG_DIR / "smartrecruiters.yaml"
    if not package_path.exists():
        log.warning("smartrecruiters.yaml not found at %s", package_path)
        return {}

    try:
        data = yaml.safe_load(package_path.read_text(encoding="utf-8"))
        return data.get("employers", {})
    except Exception as e:
        log.error("Failed to load package config: %s", e)
        return {}


def _load_location_filter(search_cfg: dict | None = None):
    """Load location accept/reject lists from search config."""
    if search_cfg is None:
        search_cfg = config.load_search_config()

    accept = search_cfg.get("location_accept", [])
    reject = search_cfg.get("location_reject_non_remote", [])
    return accept, reject


def _location_ok(location: str | None, accept: list[str], reject: list[str]) -> bool:
    """Check if a job location passes the user's location filter."""
    if not location:
        return True

    loc = location.lower()

    if any(r in loc for r in ("remote", "anywhere", "work from home", "wfh", "distributed")):
        return True

    for r in reject:
        if r.lower() in loc:
            return False

    for a in accept:
        if a.lower() in loc:
            return True

    return False


def _title_matches_query(title: str, query: str) -> bool:
    """Check if job title matches search query (simple keyword matching)."""
    if not query:
        return True

    title_lower = title.lower()
    query_terms = query.lower().split()
    return any(term in title_lower for term in query_terms)


def _strip_html(html_content: str) -> str:
    """Strip HTML tags from content to get plain text."""
    if not html_content:
        return ""

    text = re.sub(r"<[^>]+>", "", html_content)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _format_location(loc_obj: dict | None) -> str:
    """Build a human-readable location string from the API's location object."""
    if not loc_obj or not isinstance(loc_obj, dict):
        return ""
    full = loc_obj.get("fullLocation")
    if full:
        return full.strip()
    parts = [loc_obj.get("city", ""), loc_obj.get("region", ""), loc_obj.get("country", "")]
    return ", ".join(p.strip() for p in parts if p)


def fetch_jobs_api(slug: str, offset: int = 0, limit: int = PAGE_SIZE) -> dict | None:
    """Fetch one page of postings from SmartRecruiters API.

    Returns API response dict with "totalFound" and "content", or None on error.
    """
    url = f"{SMARTRECRUITERS_API_BASE}/{slug}/postings"
    params = {"limit": limit, "offset": offset}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers, params=params)

            if resp.status_code == 404:
                log.debug("Company not found: %s", slug)
                return None
            if resp.status_code == 429:
                log.warning("Rate limited for %s, retrying...", slug)
                time.sleep(2)
                resp = client.get(url, headers=headers, params=params)
                resp.raise_for_status()
            else:
                resp.raise_for_status()

            return resp.json()

    except httpx.HTTPStatusError as e:
        log.warning("HTTP error for %s: %s", slug, e)
        return None
    except Exception as e:
        log.warning("Failed to fetch %s: %s", slug, e)
        return None


def fetch_detail(slug: str, posting_id: str) -> dict | None:
    """Fetch full detail for a single posting (includes applyUrl + description)."""
    url = f"{SMARTRECRUITERS_API_BASE}/{slug}/postings/{posting_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code >= 400:
                return None
            return resp.json()
    except Exception as e:
        log.debug("Detail fetch failed for %s/%s: %s", slug, posting_id, e)
        return None


def _extract_description(detail: dict) -> str:
    """Pull concatenated description text from a posting detail payload."""
    sections = detail.get("jobAd", {}).get("sections", {}) if isinstance(detail, dict) else {}
    parts: list[str] = []
    for key in ("companyDescription", "jobDescription", "qualifications", "additionalInformation"):
        section = sections.get(key) or {}
        text = section.get("text") if isinstance(section, dict) else None
        if text:
            parts.append(_strip_html(text))
    return "\n\n".join(parts).strip()


def parse_api_response(data: dict, company_name: str, query: str = "") -> list[dict]:
    """Parse postings list from a SmartRecruiters /postings response."""
    jobs: list[dict] = []
    for posting in data.get("content", []):
        try:
            title = posting.get("name", "")
            if not title:
                continue
            if query and not _title_matches_query(title, query):
                continue

            location = _format_location(posting.get("location"))
            posting_id = posting.get("id", "")

            jobs.append({
                "title": title,
                "company": company_name,
                "location": location,
                "posting_id": posting_id,
                "url": "",  # populated by detail fetch (applyUrl)
                "released_date": posting.get("releasedDate"),
            })
        except Exception as e:
            log.debug("Error parsing posting: %s", e)
            continue

    return jobs


def search_employer(
    employer_key: str,
    employer: dict,
    search_text: str,
    location_filter: bool = True,
    accept_locs: list[str] | None = None,
    reject_locs: list[str] | None = None,
) -> list[dict]:
    """Search a single SmartRecruiters employer, paginating through all postings."""
    slug = employer.get("slug") or employer_key
    log.info('%s: searching "%s"...', employer["name"], search_text)

    all_jobs: list[dict] = []
    offset = 0
    total = None
    max_pages = 30  # cap at 3000 jobs per employer

    for _ in range(max_pages):
        data = fetch_jobs_api(slug, offset=offset, limit=PAGE_SIZE)
        if not data:
            break

        if total is None:
            total = data.get("totalFound", 0)
            log.info("%s: %d total postings", employer["name"], total)

        page_jobs = parse_api_response(data, employer["name"], search_text)
        all_jobs.extend(page_jobs)

        offset += PAGE_SIZE
        if offset >= (total or 0):
            break

    # Enrich with detail (applyUrl + description)
    enriched: list[dict] = []
    for job in all_jobs:
        detail = fetch_detail(slug, job["posting_id"])
        if detail:
            apply_url = detail.get("applyUrl") or job.get("url") or ""
            job["url"] = apply_url
            job["description"] = _extract_description(detail)
        if not job.get("url"):
            continue
        enriched.append(job)

    # Location filter
    if location_filter and (accept_locs or reject_locs):
        enriched = [
            j for j in enriched
            if _location_ok(j.get("location"), accept_locs or [], reject_locs or [])
        ]

    log.info("%s: %d jobs found", employer["name"], len(enriched))
    return enriched


def search_all(
    search_text: str,
    workers: int = 4,
    location_filter: bool = True,
    _employers_override: dict | None = None,
) -> tuple[int, int]:
    """Search all configured SmartRecruiters employers. Returns (new, existing)."""
    employers = _employers_override if _employers_override else load_employers()
    if not employers:
        log.warning("No SmartRecruiters employers configured")
        return 0, 0

    accept_locs, reject_locs = _load_location_filter()

    log.info(
        'SmartRecruiters search: %d employers, "%s", workers=%d',
        len(employers), search_text, workers,
    )

    all_jobs: list[dict] = []
    errors = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                search_employer,
                key,
                emp,
                search_text,
                location_filter,
                accept_locs,
                reject_locs,
            ): key
            for key, emp in employers.items()
        }

        for future in as_completed(futures):
            key = futures[future]
            try:
                jobs = future.result()
                all_jobs.extend(jobs)
            except Exception as e:
                log.error("Error searching %s: %s", key, e)
                errors += 1

    log.info(
        "SmartRecruiters search complete: %d total jobs from %d employers (%d errors)",
        len(all_jobs), len(employers), errors,
    )

    return _store_jobs(all_jobs)


def _store_jobs(jobs: list[dict]) -> tuple[int, int]:
    """Store discovered jobs in the database. Returns (new, existing)."""
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    new = 0
    existing = 0

    for job in jobs:
        url = job.get("url") or ""
        if not url:
            continue
        description = job.get("description", "") or ""
        try:
            conn.execute(
                "INSERT INTO jobs (url, title, salary, description, location, site, strategy, "
                "discovered_at, full_description, application_url, detail_scraped_at, detail_error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    url,
                    job["title"],
                    None,
                    description[:500] if description else None,
                    job.get("location", ""),
                    job["company"],
                    "smartrecruiters_api",
                    now,
                    description if description else None,
                    url,
                    now if description else None,
                    None,
                ),
            )
            new += 1
        except sqlite3.IntegrityError:
            existing += 1

        if (new + existing) % 50 == 0:
            conn.commit()

    conn.commit()
    return new, existing
