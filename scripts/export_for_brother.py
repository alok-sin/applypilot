#!/usr/bin/env python3
"""
export_for_brother.py — Prepare tailored resumes & cover letters for manual
job-application by packaging them into a score-binned folder hierarchy and a
companion CSV tracker suitable for Google Sheets.

Typical usage
─────────────
  # Export score >= 8 (default) to Desktop
  python3 scripts/export_for_brother.py

  # Custom threshold and output paths
  python3 scripts/export_for_brother.py --min-score 7 \
      --out-dir ~/Desktop/brother_batch_2 \
      --csv ~/Desktop/brother_batch_2.csv

  # Dry-run — see what would be exported without copying anything
  python3 scripts/export_for_brother.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# ── Defaults ─────────────────────────────────────────────────────────────────
# `/` can't appear in filenames on macOS (it's the path separator), so dates
# are stamped as DD-MM-YYYY rather than DD/MM/YYYY.
_DATE_STAMP      = datetime.now().strftime("%d-%m-%Y")
DEFAULT_DB       = os.path.expanduser("~/.applypilot-plus/applypilot.db")
DEFAULT_OUT_DIR  = os.path.expanduser(f"~/Desktop/resumes_{_DATE_STAMP}")
DEFAULT_CSV      = os.path.expanduser(f"~/Desktop/brother_jobs_tracker_{_DATE_STAMP}.csv")
DEFAULT_MIN      = 8

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_text(text: str | None, max_len: int = 200) -> str:
    """Strip markdown noise and truncate for CSV readability."""
    if not text:
        return ""
    text = re.sub(r'\*\*|\\-|\\&|#{1,6}\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def extract_reasoning(raw: str | None) -> str:
    """Pull the human-readable reasoning from the scorer output."""
    if not raw:
        return ""
    out = re.sub(r'^KEYWORDS:.*\n', '', raw, flags=re.MULTILINE).strip()
    out = re.sub(r'^REASONING:\s*', '', out).strip()
    return out


def drive_search_formula(row_num: int, col: str = "H") -> str:
    """Google Sheets formula that hyperlinks to a Drive search for the PDF."""
    return (
        f'=IF(ISBLANK({col}{row_num}), "", '
        f'HYPERLINK("https://drive.google.com/drive/search?q=" & '
        f'REGEXEXTRACT({col}{row_num}, "([^/]+)\\.pdf$"), "📁 Open"))'
    )


# ── Core logic ───────────────────────────────────────────────────────────────

def detect_latest_run_start(db_path: str, gap_minutes: int = 360) -> str | None:
    """
    Walk `discovered_at` from newest → oldest and stop at the first gap larger
    than `gap_minutes`. Returns the earliest ISO timestamp still belonging to
    the most recent run, or None if the table is empty.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT discovered_at FROM jobs "
        "WHERE discovered_at IS NOT NULL "
        "ORDER BY discovered_at DESC"
    )
    timestamps = [r[0] for r in cur.fetchall()]
    conn.close()

    if not timestamps:
        return None

    gap = gap_minutes * 60
    run_start = timestamps[0]
    prev = datetime.fromisoformat(timestamps[0])
    for ts in timestamps[1:]:
        cur_dt = datetime.fromisoformat(ts)
        if (prev - cur_dt).total_seconds() > gap:
            break
        run_start = ts
        prev = cur_dt
    return run_start


def query_jobs(db_path: str, min_score: int, since: str | None = None) -> list[tuple]:
    """Return qualifying rows from the applypilot DB."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    sql = """
        SELECT
            fit_score,
            site,
            title,
            url,
            application_url,
            tailored_resume_path,
            cover_letter_path,
            description,
            location,
            salary,
            score_reasoning
        FROM jobs
        WHERE fit_score >= ?
          AND tailored_resume_path IS NOT NULL
          AND cover_letter_path IS NOT NULL
    """
    params: list = [min_score]
    if since:
        sql += " AND discovered_at >= ?"
        params.append(since)
    sql += " ORDER BY fit_score DESC, title ASC"
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return rows


def build_hierarchy(rows: list[tuple], out_dir: str, *, dry_run: bool = False):
    """
    Copy resume + cover-letter PDFs into a score-binned folder tree:

        <out_dir>/
          Score_10/<job_base>/  resume.pdf  cover_letter.pdf
          Score_9/<job_base>/   …
          Score_8/<job_base>/   …

    Returns a list of dicts ready for CSV writing.
    """
    if not dry_run:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    csv_rows: list[dict] = []
    stats = {"jobs": 0, "files": 0, "skipped": 0}

    for row in rows:
        (score, site, title, listing_url, apply_url,
         resume_path, cl_path,
         desc, location, salary, reasoning) = row

        if not resume_path or not os.path.exists(resume_path):
            stats["skipped"] += 1
            continue

        base_name = os.path.basename(resume_path).replace(".pdf", "")
        resume_dir = os.path.dirname(resume_path)
        cl_dir = resume_dir.replace("tailored_resumes", "cover_letters")

        # Destination folder
        job_folder = os.path.join(out_dir, f"Score_{score}", base_name)

        if not dry_run:
            os.makedirs(job_folder, exist_ok=True)
            shutil.copy2(resume_path, os.path.join(job_folder, os.path.basename(resume_path)))
        stats["files"] += 1

        # Cover-letter PDF
        cl_pdf = os.path.join(cl_dir, base_name + "_CL.pdf")
        cl_copied = False
        if os.path.exists(cl_pdf):
            if not dry_run:
                shutil.copy2(cl_pdf, os.path.join(job_folder, os.path.basename(cl_pdf)))
            stats["files"] += 1
            cl_copied = True
        elif cl_path:
            cl_alt = cl_path.replace("_CL.txt", "_CL.pdf")
            if os.path.exists(cl_alt):
                if not dry_run:
                    shutil.copy2(cl_alt, os.path.join(job_folder, os.path.basename(cl_alt)))
                stats["files"] += 1
                cl_copied = True

        # Job description text (_JOB.txt)
        job_txt = os.path.join(resume_dir, base_name + "_JOB.txt")
        if os.path.exists(job_txt) and not dry_run:
            shutil.copy2(job_txt, os.path.join(job_folder, os.path.basename(job_txt)))
            stats["files"] += 1

        # Resume JSON (the structured tailored resume data)
        resume_json = os.path.join(resume_dir, base_name + ".json")
        if os.path.exists(resume_json) and not dry_run:
            shutil.copy2(resume_json, os.path.join(job_folder, os.path.basename(resume_json)))
            stats["files"] += 1

        stats["jobs"] += 1

        csv_rows.append({
            "score": score,
            "site": site or "",
            "title": title or "",
            "location": location or "",
            "salary": salary or "",
            "listing_url": listing_url or "",
            "apply_url": apply_url or "",
            "resume_file": os.path.basename(resume_path),
            "cl_file": os.path.basename(cl_pdf) if cl_copied else "",
            "description": clean_text(desc),
            "reasoning": extract_reasoning(reasoning),
        })

    return csv_rows, stats


def write_csv(csv_rows: list[dict], csv_path: str, *, dry_run: bool = False):
    """Write the tracker CSV with a Drive-link formula column."""
    if dry_run:
        return

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Score", "Source", "Job Title", "Location", "Salary",
            "Listing URL", "Apply URL", "Resume File", "Cover Letter File",
            "Description Summary", "Why Good Fit",
            "Drive Link", "Applied?", "Notes",
        ])
        for idx, r in enumerate(csv_rows):
            row_num = idx + 2
            writer.writerow([
                r["score"], r["site"], r["title"],
                r["location"], r["salary"],
                r["listing_url"], r["apply_url"],
                r["resume_file"], r["cl_file"],
                r["description"], r["reasoning"],
                drive_search_formula(row_num),
                "",   # Applied?
                "",   # Notes
            ])


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export tailored resumes & cover letters for manual job applications.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--db", default=DEFAULT_DB, metavar="PATH",
        help=f"Path to applypilot.db  [default: {DEFAULT_DB}]",
    )
    p.add_argument(
        "--min-score", type=int, default=DEFAULT_MIN, metavar="N",
        help=f"Minimum fit_score to include  [default: {DEFAULT_MIN}]",
    )
    p.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR, metavar="DIR",
        help=f"Root directory for binned folders  [default: {DEFAULT_OUT_DIR}]",
    )
    p.add_argument(
        "--csv", default=DEFAULT_CSV, metavar="PATH",
        help=f"Output CSV path  [default: {DEFAULT_CSV}]",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be exported without copying any files.",
    )
    p.add_argument(
        "--latest", action="store_true",
        help="Auto-detect and export only the most recent discovery run "
             "(uses discovered_at gaps to find the run boundary).",
    )
    p.add_argument(
        "--latest-gap", type=int, default=360, metavar="MINUTES",
        help="Gap threshold for --latest run detection  [default: 360 = 6h]",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.db):
        print(f"ERROR: Database not found at {args.db}", file=sys.stderr)
        return 1

    # 1. Query
    since = None
    if args.latest:
        since = detect_latest_run_start(args.db, args.latest_gap)
        if since is None:
            print("ERROR: No discovered_at timestamps in DB; cannot detect latest run.", file=sys.stderr)
            return 1
        print(f"Latest run detected: jobs discovered since {since} (gap threshold {args.latest_gap} min)")

    rows = query_jobs(args.db, args.min_score, since=since)
    print(f"Found {len(rows)} jobs with score >= {args.min_score}")
    if not rows:
        print("Nothing to export.")
        return 0

    # 2. Build hierarchy
    csv_rows, stats = build_hierarchy(rows, args.out_dir, dry_run=args.dry_run)
    verb = "Would copy" if args.dry_run else "Copied"
    print(f"{verb} {stats['files']} files across {stats['jobs']} jobs → {args.out_dir}")
    if stats["skipped"]:
        print(f"  Skipped {stats['skipped']} (missing resume PDF)")

    # 3. Write CSV
    write_csv(csv_rows, args.csv, dry_run=args.dry_run)
    verb2 = "Would generate" if args.dry_run else "Generated"
    print(f"{verb2} CSV with {len(csv_rows)} rows → {args.csv}")

    # 4. Summary
    print()
    if not args.dry_run:
        for entry in sorted(os.listdir(args.out_dir), reverse=True):
            entry_path = os.path.join(args.out_dir, entry)
            if os.path.isdir(entry_path):
                count = len(os.listdir(entry_path))
                print(f"  {entry}/ → {count} job folders")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
