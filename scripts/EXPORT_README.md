# export_for_brother.py

Packages tailored resumes and cover letters into a **score-binned folder hierarchy** for Google Drive upload, plus a **CSV tracker** for Google Sheets.

---

## Quick Start

```bash
# From the applypilot-alok repo root:
python3 scripts/export_for_brother.py

# Outputs → ~/Desktop/resumes_DD-MM-YYYY/  and  ~/Desktop/brother_jobs_tracker_DD-MM-YYYY.csv
# (today's date is stamped automatically so runs don't overwrite each other)
```

## What It Creates

### 📁 Folder Hierarchy (for Google Drive)

```
~/Desktop/resumes/
├── Score_10/
│   └── linkedin_B2B_Sales_Executive_d28b7650/
│       ├── linkedin_B2B_Sales_Executive_d28b7650.pdf          ← Resume
│       ├── linkedin_B2B_Sales_Executive_d28b7650_CL.pdf       ← Cover Letter
│       ├── linkedin_B2B_Sales_Executive_d28b7650_JOB.txt      ← Job Description
│       └── linkedin_B2B_Sales_Executive_d28b7650.json         ← Resume JSON
├── Score_9/
│   ├── linkedin_BDM_e14e0695/
│   │   ├── ...pdf, _CL.pdf, _JOB.txt, .json
│   └── ... (more job folders)
└── Score_8/
    └── ...
```

Each job folder contains up to **4 files**:

| File              | What it is                                                    |
| ----------------- | ------------------------------------------------------------- |
| `<name>.pdf`      | Tailored resume PDF — ready to submit                         |
| `<name>_CL.pdf`   | Tailored cover letter PDF                                     |
| `<name>_JOB.txt`  | Full job description text — useful for interview prep          |
| `<name>.json`     | Structured resume JSON — the raw data used to generate the PDF |

**Upload this entire `resumes/` folder to Google Drive.**

### 📊 CSV Tracker (for Google Sheets)

| Column                  | Purpose                                                  |
| ----------------------- | -------------------------------------------------------- |
| **Score**               | Fit score (10 = perfect match)                           |
| **Source**              | Where the job was found (linkedin, Dice, etc.)           |
| **Job Title**           | Role name                                                |
| **Location**            | City / region                                            |
| **Salary**              | If listed                                                |
| **Listing URL**         | Link to the original job posting — always populated      |
| **Apply URL**           | Direct application link (if available, otherwise empty)  |
| **Resume File**         | PDF filename — matches the file in Drive                 |
| **Cover Letter File**   | CL PDF filename                                         |
| **Description Summary** | First ~200 chars of the job description                  |
| **Why Good Fit**        | AI reasoning for why this is a strong match              |
| **Drive Link**          | Auto-link formula — opens the matching folder in Drive   |
| **Applied?**            | ✅ Empty — fill in as you apply                           |
| **Notes**               | Empty — for your own notes                               |

**Usage:** Upload the CSV to Google Sheets. The "Drive Link" column has a `HYPERLINK` + `REGEXEXTRACT` formula that auto-searches your Google Drive for the matching resume PDF.

---

## Options

```
usage: export_for_brother.py [-h] [--db PATH] [--min-score N]
                              [--out-dir DIR] [--csv PATH] [--dry-run]

  --db PATH             Path to applypilot.db   [default: ~/.applypilot-plus/applypilot.db]
  --min-score N         Minimum fit_score        [default: 8]
  --out-dir DIR         Root for binned folders   [default: ~/Desktop/resumes_DD-MM-YYYY]
  --csv PATH            Output CSV path           [default: ~/Desktop/brother_jobs_tracker_DD-MM-YYYY.csv]
  --dry-run             Preview without copying files
  --latest              Only export jobs from the most recent discovery run
  --latest-gap MINUTES  Gap threshold used by --latest  [default: 360 = 6h]
```

## Examples

```bash
# Include score 7 jobs too
python3 scripts/export_for_brother.py --min-score 7

# Export only the run you just finished
python3 scripts/export_for_brother.py --latest

# Custom output location
python3 scripts/export_for_brother.py --out-dir ~/Drive/brother_batch_3 --csv ~/Drive/tracker_3.csv

# See what would happen without doing anything
python3 scripts/export_for_brother.py --dry-run
```

### How `--latest` works

It walks `discovered_at` from newest to oldest and stops at the first gap larger than `--latest-gap` (default 6h). Everything on the newest side of that gap counts as "the latest run." The wide default is intentional — runs are typically separated by many hours, and erring on the side of including a long run is safer than clipping its tail.

## Workflow

1. Run the ApplyPilot pipeline to discover, score, and tailor resumes
2. Run this script to package the output
3. Upload `resumes/` folder to Google Drive
4. Upload the CSV to Google Sheets
5. Share both with your brother
6. Brother uses the sheet to track applications — clicking "📁 Open" in the Drive Link column jumps to the matching resume folder

> **Note:** The script wipes and recreates `--out-dir` each run to avoid stale files mixing with new ones.
