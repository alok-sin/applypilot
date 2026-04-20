#!/usr/bin/env python3
"""Read-only dedup impact analysis.

Quantifies how much LLM spend could be saved by content-hash deduplication
of scored jobs, with and without boilerplate stripping. The DB is opened in
read-only mode; this script makes no writes.

Usage:
    python scripts/dedup_dry_run.py [db_path]

Defaults to ~/.applypilot-plus/applypilot.db.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path


DEFAULT_DB = Path.home() / ".applypilot-plus" / "applypilot.db"

# Rough char-to-token ratio (English resume+job text): ~4 chars/token.
CHARS_PER_TOKEN = 4

# Fixed prefix ships on every score call: SCORE_PROMPT (~500 chars) + resume
# (~6000 chars typical). Use 7000 chars = 1750 tokens as a conservative est.
ASSUMED_PREFIX_TOKENS = 1750

# Per-provider input pricing ($ / 1M tokens). Output is small (~150 tokens
# at ~2x input rate) so its effect on this analysis is in the noise.
PROVIDER_PRICING: dict[str, float] = {
    "Gemini 2.5 Flash":      0.075,
    "Gemini 2.5 Flash Lite": 0.04,
    "gpt-5-mini":            0.25,
    "Claude Haiku 4.5":      1.00,
    "Claude Sonnet 4.5":     3.00,
}

# Boilerplate section headers that commonly start large verbatim blocks in
# job descriptions. Stripping from header to next blank line gives a more
# stable description fingerprint.
_BOILERPLATE_HEADERS = [
    r"equal[\s\-]?(opportunity|employment)[^\n]*",
    r"affirmative action[^\n]*",
    r"about (us|our company|the company)[^\n]*",
    r"our (values|mission|culture)[^\n]*",
    r"benefits[:\s]",
    r"what we offer[^\n]*",
    r"perks[^\n]*",
    r"compensation( and benefits)?[:\s]",
    r"eeo statement[^\n]*",
]
_BOILERPLATE_RE = re.compile(
    r"(?im)^(?:" + "|".join(_BOILERPLATE_HEADERS) + r").*?(?:\n\s*\n|\Z)",
    re.DOTALL,
)
_WHITESPACE_RE = re.compile(r"\s+")


def _norm(s: str | None) -> str:
    if not s:
        return ""
    return _WHITESPACE_RE.sub(" ", s.strip().lower())


def _normalize_description(desc: str | None, strip_boilerplate: bool) -> str:
    if not desc:
        return ""
    text = desc
    if strip_boilerplate:
        text = _BOILERPLATE_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _hash_job(title: str | None, site: str | None, location: str | None,
              desc_norm: str, include_location: bool = True) -> str:
    parts = [_norm(title), _norm(site)]
    if include_location:
        parts.append(_norm(location))
    parts.append(desc_norm[:1000])
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _load_scored(db_path: Path) -> list[tuple]:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT title, site, location, full_description "
            "FROM jobs WHERE fit_score IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    return rows


def _compute_groups(
    rows: list[tuple],
    strip_boilerplate: bool,
    include_location: bool = True,
) -> tuple[Counter, list[tuple[str, str, int]]]:
    hashes: Counter = Counter()
    hash_to_sample: dict[str, tuple[str, str]] = {}
    for title, site, location, desc in rows:
        norm = _normalize_description(desc, strip_boilerplate)
        h = _hash_job(title, site, location, norm, include_location=include_location)
        hashes[h] += 1
        if h not in hash_to_sample:
            hash_to_sample[h] = (title or "?", site or "?")
    top = [(hash_to_sample[h][0], hash_to_sample[h][1], n) for h, n in hashes.most_common(20)]
    return hashes, top


def _avg_desc_len(rows: list[tuple]) -> int:
    lens = [len(r[3]) for r in rows if r[3]]
    return sum(lens) // len(lens) if lens else 0


def _dollar_projection(wasted_tokens: int) -> dict[str, float]:
    per_M = wasted_tokens / 1_000_000
    return {k: round(per_M * v, 4) for k, v in PROVIDER_PRICING.items()}


def _render_report(db_path: Path, rows: list[tuple]) -> str:
    total = len(rows)
    if total == 0:
        return f"# Dedup Dry-Run\n\nNo scored jobs found in {db_path}.\n"

    avg_desc = _avg_desc_len(rows)
    avg_job_tokens = (avg_desc + 200) // CHARS_PER_TOKEN  # desc + title/site/location
    tokens_per_call = ASSUMED_PREFIX_TOKENS + avg_job_tokens

    raw_hashes, raw_top = _compute_groups(rows, strip_boilerplate=False)
    norm_hashes, norm_top = _compute_groups(rows, strip_boilerplate=True)
    # Upper-bound: ignore location (useful when the user takes remote roles
    # and sees the same job reposted across multiple office cities).
    nolocs_hashes, nolocs_top = _compute_groups(
        rows, strip_boilerplate=True, include_location=False,
    )

    def stats(hashes: Counter) -> dict:
        unique = len(hashes)
        dup_groups = sum(1 for n in hashes.values() if n > 1)
        jobs_in_dups = sum(n for n in hashes.values() if n > 1)
        wasted_calls = sum(n - 1 for n in hashes.values() if n > 1)
        wasted_tokens = wasted_calls * tokens_per_call
        return {
            "unique": unique,
            "dup_groups": dup_groups,
            "jobs_in_dups": jobs_in_dups,
            "wasted_calls": wasted_calls,
            "wasted_tokens": wasted_tokens,
            "dollars": _dollar_projection(wasted_tokens),
        }

    raw = stats(raw_hashes)
    norm = stats(norm_hashes)
    nolocs = stats(nolocs_hashes)

    lines: list[str] = []
    lines.append("# Dedup Dry-Run Report")
    lines.append("")
    lines.append(f"Source: `{db_path}`  (opened read-only)")
    lines.append("")
    lines.append(f"- Scored jobs analyzed: **{total:,}**")
    lines.append(f"- Avg full_description length: **{avg_desc:,} chars** "
                 f"(~{avg_desc // CHARS_PER_TOKEN:,} tokens)")
    lines.append(f"- Assumed tokens per score call: "
                 f"~{tokens_per_call:,} ({ASSUMED_PREFIX_TOKENS} prefix + {avg_job_tokens} job)")
    lines.append("")

    lines.append("## Raw dedup (identical title+site+location+description prefix)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Unique content hashes | {raw['unique']:,} |")
    lines.append(f"| Duplicate groups (size>1) | {raw['dup_groups']:,} |")
    lines.append(f"| Jobs in duplicate groups | {raw['jobs_in_dups']:,} "
                 f"({100 * raw['jobs_in_dups'] / total:.1f}%) |")
    lines.append(f"| **Wasted LLM calls** | **{raw['wasted_calls']:,}** "
                 f"({100 * raw['wasted_calls'] / total:.1f}%) |")
    lines.append(f"| Wasted input tokens | ~{raw['wasted_tokens']:,} |")
    lines.append("")
    lines.append("### Raw: $ savings if deduped (per run)")
    lines.append("")
    lines.append("| Provider | Savings |")
    lines.append("|---|---|")
    for prov, amt in raw["dollars"].items():
        lines.append(f"| {prov} | ${amt:.4f} |")
    lines.append("")

    lines.append("## Normalized dedup (after stripping EEO/benefits/About Us boilerplate)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Unique content hashes | {norm['unique']:,} "
                 f"(-{raw['unique'] - norm['unique']} vs raw) |")
    lines.append(f"| **Wasted LLM calls** | **{norm['wasted_calls']:,}** "
                 f"(+{norm['wasted_calls'] - raw['wasted_calls']} extra wins) |")
    lines.append(f"| Wasted input tokens | ~{norm['wasted_tokens']:,} |")
    lines.append("")
    lines.append("### Normalized: $ savings if deduped (per run)")
    lines.append("")
    lines.append("| Provider | Savings |")
    lines.append("|---|---|")
    for prov, amt in norm["dollars"].items():
        lines.append(f"| {prov} | ${amt:.4f} |")
    lines.append("")

    lines.append("## Upper bound: ignore location (remote-friendly seekers)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Unique content hashes | {nolocs['unique']:,} |")
    lines.append(f"| **Wasted LLM calls** | **{nolocs['wasted_calls']:,}** "
                 f"({100 * nolocs['wasted_calls'] / total:.1f}%) |")
    lines.append(f"| Wasted input tokens | ~{nolocs['wasted_tokens']:,} |")
    lines.append("")
    lines.append("### Upper-bound: $ savings if deduped (per run)")
    lines.append("")
    lines.append("| Provider | Savings |")
    lines.append("|---|---|")
    for prov, amt in nolocs["dollars"].items():
        lines.append(f"| {prov} | ${amt:.4f} |")
    lines.append("")

    lines.append("## Top 20 duplicate groups (raw hashing)")
    lines.append("")
    lines.append("| Title | Site | Count |")
    lines.append("|---|---|---|")
    for title, site, count in raw_top:
        if count <= 1:
            break
        t = (title or "?")[:60]
        s = (site or "?")[:20]
        lines.append(f"| {t} | {s} | {count} |")
    lines.append("")

    lines.append("## Top 20 duplicate groups (normalized)")
    lines.append("")
    lines.append("| Title | Site | Count |")
    lines.append("|---|---|---|")
    for title, site, count in norm_top:
        if count <= 1:
            break
        t = (title or "?")[:60]
        s = (site or "?")[:20]
        lines.append(f"| {t} | {s} | {count} |")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        return 1
    # Capture mtime before opening so we can sanity-check non-modification.
    pre_mtime = os.path.getmtime(db_path)
    rows = _load_scored(db_path)
    print(_render_report(db_path, rows))
    post_mtime = os.path.getmtime(db_path)
    if pre_mtime != post_mtime:
        print(f"\nWARNING: DB mtime changed ({pre_mtime} → {post_mtime})", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
