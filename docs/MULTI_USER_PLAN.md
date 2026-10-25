# Multi-user ApplyPilot — Architecture & Implementation Plan

**Status as of this document:** Phase 0 (core library boundary) shipped. **Phase 1 v1 (admin-operated multi-user, per-profile DBs) shipped** in the sibling repo `applypilot-saas/` — see [`../../applypilot-saas/README.md`](../../applypilot-saas/README.md). The shared `jobs` / `user_jobs` schema split is a Phase 1.5 optimization (still pending). Phases 2 (backend) and 3 (local agent) sketched but deferred.

This is a working document. It exists to make future-me pick up where present-me left off without re-deriving the architecture from `git log`. Everything here is opinionated: it tells you what the code does today, what it should do next, and what specifically to write.

---

## 0. Mental model: hub-spoke

Two repos, one library boundary.

```
┌──────── applypilot-saas (sibling repo at ../applypilot-saas/, no remote) ───────┐
│                                                                                  │
│   Admin CLI (Phase 1 v1, SHIPPED)  FastAPI backend (Phase 2, sketched)           │
│   Schema split (Phase 1.5, TODO)   Local agent (Phase 3, sketched)               │
│   Postgres + alembic (Phase 2)     Browser extension (Phase 3, sketched)         │
│                                                                                  │
│   ┌────────────────────────────────────────────────────────────┐                 │
│   │ Worker / admin command                                     │                 │
│   │   1. Hydrate UserContext from profile dir / DB row         │                 │
│   │   2. Build TaskContext (run_id, cancellation, overrides)   │                 │
│   │   3. Call applypilot.core.pipeline.run(ctx)                │                 │
│   └────────────────────────────────────────────────────────────┘                 │
│                              │                                                   │
└──────────────────────────────┼───────────────────────────────────────────────────┘
                               │ depends on (path-dep during dev, semver later)
                               ▼
┌─────────────── applypilot (THIS REPO, the core) ────────────────────────────────┐
│                                                                                  │
│   pipeline.run(ctx) ──► discover ─► enrich ─► score ─► tailor ─► cover ─► pdf   │
│                         apply (browser-driven)                                   │
│                                                                                  │
│   applypilot.core/                                                               │
│     context.py     UserContext, TaskContext, RunContext                          │
│                    SecretsProvider, StorageBackend, Database,                    │
│                    CancellationToken, ProgressReporter (Protocols)               │
│     local_providers.py  LocalSecrets, LocalStorage, SqliteDatabase,              │
│                         LocalCancellationToken, EventCancellationToken,          │
│                         build_default_run_context, build_local_run_context       │
│                                                                                  │
│   Single-user CLI (init, run, apply, dashboard, doctor) builds a default         │
│   RunContext from APP_DIR and hands it to pipeline.run.                          │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Why two repos, not a flag in one:** the moment we add `if saas_mode:` branches in core, the open-source story rots — every config knob doubles, every dependency creeps. The spoke imports core as a versioned dependency. Core stays a clean, single-user library forever.

**Posture inside the spoke:** pool everything. One Postgres, one worker pool, one crawler with global politeness budget. Per-user isolation is row-level + secret/artifact boundaries, not separate infra.

---

## 1. What Phase 0 actually shipped (this repo, today)

The core is now a library. Every pipeline stage takes `ctx: RunContext`. Module-globals for paths, profile, search config, LLM keys, DB connection, and cancellation are all gone from the call path — they survive only as module-level fallbacks for the standalone CLI to keep working.

### 1.1 The contract

[src/applypilot/core/context.py](../src/applypilot/core/context.py) defines:

```python
@runtime_checkable
class SecretsProvider(Protocol):
    def get(self, key: str, default: str | None = None) -> str | None: ...

@runtime_checkable
class StorageBackend(Protocol):
    def tailored_dir(self) -> Path: ...
    def cover_letter_dir(self) -> Path: ...
    def log_dir(self) -> Path: ...

@runtime_checkable
class Database(Protocol):
    def connection(self) -> sqlite3.Connection: ...
    def close(self) -> None: ...

@runtime_checkable
class CancellationToken(Protocol):
    def is_set(self) -> bool: ...
    def set(self) -> None: ...
    def clear(self) -> None: ...

@runtime_checkable
class ProgressReporter(Protocol):
    def stage_start(self, stage: str) -> None: ...
    def stage_finish(self, stage: str, stats: dict) -> None: ...
    def event(self, name: str, payload: dict) -> None: ...

@dataclass
class UserContext:
    user_id: str
    data_dir: Path
    profile: dict
    resume_text: str
    resume_pdf: bytes | None
    search_config: dict
    llm_config: dict = field(default_factory=dict)
    secrets: SecretsProvider | None = None
    storage: StorageBackend | None = None
    db: Database | None = None

@dataclass
class TaskContext:
    run_id: str
    cancellation: CancellationToken
    llm_overrides: dict[str, "LLMConfig"] = field(default_factory=dict)
    progress: ProgressReporter | None = None

@dataclass
class RunContext:
    user: UserContext
    task: TaskContext
```

**Two layers, deliberately.** `UserContext` is identity + data + providers — stable per user, reusable across many tasks (a worker holds it warm). `TaskContext` is per-invocation: a new run_id, a fresh cancellation token, optional LLM overrides for *this* task. Cancelling one task doesn't mutate who the user is.

[src/applypilot/core/local_providers.py](../src/applypilot/core/local_providers.py) ships the local-disk implementations:

- `LocalSecretsProvider(data_dir)` — reads `data_dir/.env`, falls through to `os.environ`.
- `LocalStorageBackend(data_dir)` — filesystem dirs under the user's data root.
- `SqliteDatabase(db_path)` — per-thread connection cache, WAL mode. Single-table schema (no `jobs` / `user_jobs` split — that's a spoke concern).
- `LocalCancellationToken()` — fresh `threading.Event` per task.
- `EventCancellationToken(event)` — adapter that wraps an existing Event (used by the CLI to bridge the SIGINT-driven module-global into per-task ctxs).
- `build_default_run_context()` — the CLI's bootstrap: `APP_DIR` + module-global `stop_event`.
- `build_local_run_context(data_dir)` — hydrates a ctx from any directory that has the standard layout (`profile.json`, `resume.txt`, `resume.pdf?`, `searches.yaml`, `llm.yaml?`, `.env?`, `applypilot.db`).

### 1.2 What changed at the call sites

Every stage entry point now takes `ctx: "RunContext | None" = None` and falls back to `build_default_run_context()` when None. The full set of files that was rewritten in Phase 0:

- [pipeline.py](../src/applypilot/pipeline.py) — orchestrator threads ctx to all runners; `_build_stage_kwargs` centralizes per-stage arg shaping; `init_db_for_ctx` / `get_stats_for_ctx` replace module-global versions.
- [cli.py](../src/applypilot/cli.py) — every command (`run`, `apply`, `add-url`, `status`, `dashboard`) calls `_bootstrap()` to build a ctx and threads it through.
- [llm.py](../src/applypilot/llm.py) — `resolve_config_for_ctx(ctx, task)` resolves in priority order: `ctx.task.llm_overrides[task]` → `ctx.user.llm_config` (parsed `llm.yaml`) + user secrets/env → env-only fallback. `get_client_for_ctx(ctx, task)` builds a fresh client per ctx (no shared cache — two users must never share a client built with someone else's credentials).
- [database.py](../src/applypilot/database.py) — `init_db_for_ctx(ctx)` and `get_stats_for_ctx(ctx)` operate on `ctx.user.db.connection()`.
- [cancellation.py](../src/applypilot/cancellation.py) — module-global `stop_event` survives for the SIGINT handler, but `make_cancellation_token()` returns fresh per-task events.
- [scoring/{scorer,tailor,cover_letter,pdf}.py](../src/applypilot/scoring/) — read profile, resume, search config, LLM client all from ctx; check `ctx.task.cancellation` between jobs in the inner loop.
- [enrichment/detail.py](../src/applypilot/enrichment/detail.py) — same pattern; LLM client passed into per-row scrapers.
- [discovery/{greenhouse,smartrecruiters,workday,smartextract,jobspy,llm_sweep}.py](../src/applypilot/discovery/) — every backend takes ctx; storage writes go through `ctx.user.db.connection()`.
- [apply/{launcher,prompt}.py](../src/applypilot/apply/) — `acquire_job`, `run_job`, `worker_loop`, `mark_*`, `reset_*`, `remove_expired`, `reset_in_progress` all take ctx.
- [view.py](../src/applypilot/view.py) — dashboard generator takes ctx.

### 1.3 What this *guarantees* today

These are tested invariants. If they break, Phase 1 breaks.

**Two concurrent ctxs do not cross-contaminate.** [tests/test_pipeline_ctx.py::test_two_pipeline_ctxs_cancel_independently](../tests/test_pipeline_ctx.py) builds two `RunContext`s in different temp dirs, asserts cancelling one leaves the other's token clear. [tests/test_database_ctx.py](../tests/test_database_ctx.py) covers DB isolation; [tests/test_llm_ctx.py](../tests/test_llm_ctx.py) covers credential isolation across the resolver.

**The CLI is byte-identical.** Same files in `~/.applypilot-plus/`, same commands, same outputs. The only internal change is that `_bootstrap()` builds a ctx and `pipeline.run_pipeline(ctx=...)` is the call.

**LLM resolution honors precedence.** Task-specific override in `TaskContext.llm_overrides` beats `llm.yaml` beats `.env` beats `os.environ`. [tests/test_llm_resolution.py](../tests/test_llm_resolution.py) and [tests/test_llm_ctx.py](../tests/test_llm_ctx.py) cover every fallback step.

### 1.4 What Phase 0 did *not* do

- No schema split. The single-user core writes one `jobs` table with both shared facts (title, description) and per-user facts (fit_score, tailored_resume_path) in the same row. That's wrong for multi-user but right for single-user — keep core simple.
- No tenant tables. No `users`, `user_jobs`, `user_secrets`, `usage_events` — those are spoke-only.
- No Postgres, no alembic, no FastAPI, no S3, no KMS. None of those should ever appear in this repo.
- No admin CLI in core. The `applypilot` binary is still the single-user CLI. The admin CLI lives in the spoke (see §1.6).

### 1.5 The bare-CLI escape hatch (still works, mostly superseded by §1.6)

You can run for N users from the existing single-user CLI by overriding `APPLYPILOT_DIR`:

```bash
APPLYPILOT_DIR=~/profiles/alice applypilot run
APPLYPILOT_DIR=~/profiles/bob   applypilot run
APPLYPILOT_DIR=~/profiles/alice applypilot apply
```

Each invocation gets its own ctx, its own DB, its own artifacts. No cross-user dedup (Greenhouse gets crawled N times) and no shared crawler politeness budget. Useful as a manual fallback or for one-off testing of a single profile dir. For routine multi-user runs, use the spoke (§1.6) — it batches, validates, and exports in one command.

### 1.6 Phase 1 v1 — what the spoke ships *today*

Phase 1 v1 is shipped. It lives in the sibling repo `../applypilot-saas/` (no remote configured intentionally — profiles + `.env` keys stay on the admin's box). The spoke is a pip-installable CLI built on core's `RunContext` boundary.

**What it does:**

- Each user is a directory under `profiles/<name>/` matching core's `APP_DIR` shape (`profile.json`, `resume.txt`, `searches.yaml`, `llm.yaml?`, `.env`, `applypilot.db`, `tailored_resumes/`, `cover_letters/`).
- The admin CLI iterates profiles sequentially, calling `applypilot.core.build_local_run_context(profile_dir)` per user and then `applypilot.pipeline.run_pipeline(ctx=ctx, ...)`. Per-user isolation is total: independent secrets, independent DB files, independent LLM clients, independent cancellation tokens. (Verified by `tests/test_profile_isolation.py` in the spoke.)
- Commands shipped: `admin onboard <name> [--from-zip]`, `admin run --profiles a,b,c [stages…] [--dry-run]`, `admin run --all`, `admin apply --profiles a,b,c [--limit N] [--workers N] [--continuous] [--url URL]`, `admin export --profile alice [--min-score N] [--latest]`, `admin status`.
- `admin export` is a profile-aware descendant of `scripts/export_for_brother.py` — the score-binned folder + `jobs.csv` shape is preserved verbatim.
- `admin run --dry-run` resolves the LLM model per stage per profile via core's `resolve_config_for_ctx`, prints pending-work counts, and performs zero LLM calls.
- Failure isolation: if alice's run crashes, bob's still runs. Errors land in the structured JSONL log under `logs/<run_id>.jsonl`.

**What it intentionally doesn't do (deferred to Phase 1.5 — see §2):**

- No shared `jobs` table across users. Each profile has its own `applypilot.db`. If alice and bob both watch Greenhouse, Greenhouse gets crawled twice. Per-user cost is correct; total infrastructure cost is N×.
- No `usage_events` table; `admin spend` is not implemented.
- No global crawler politeness budget across tenants.

**Round-trip:** the spoke's `profiles/<name>/` is byte-equivalent to a core `APP_DIR`. A user can author a dir locally with `applypilot init`, zip it, send it; the admin imports it via `admin onboard <name> --from-zip`; later the admin returns the dir (with results) and the user can re-open it locally with `APPLYPILOT_DIR=~/alice applypilot dashboard`. Same shape, no translation.

---

## 2. Phase 1.5: shared `jobs` / per-tenant `user_jobs` schema (next thing to build)

This is the optimization that makes the spoke scale past ~5 users without N× infrastructure cost. v1 (§1.6) proves the multi-user *architecture* works; v1.5 makes it *efficient*.

The deliverable: one shared SQLite (or Postgres-ready schema) with a clean shared/per-tenant split, so two profiles dedup discovery against one DB. No backend, no Postgres, no remote, no users typing API keys into a web form. The only network traffic is to ATS/job-board scrapers and LLM providers, exactly as today.

### 2.1 Repo bootstrap — DONE in v1

Spoke repo created at `../applypilot-saas/` with `git init` and no remote configured. `pyproject.toml` path-deps on `../applypilot-alok` (this repo) via `[tool.uv.sources]`. Editable install: `pip install -e ../applypilot-saas` from core's venv. Phase 1.5 reuses this — no further bootstrap needed.

### 2.2 On-disk layout (admin's box)

**v1 (shipped) layout** — each profile is a complete `APP_DIR`, including its own DB:

```
~/.applypilot-saas/
├── profiles/
│   ├── alice/                      # exactly a core APP_DIR — no adapter
│   │   ├── profile.json
│   │   ├── resume.txt
│   │   ├── resume.pdf              # optional; auto-generated from .txt
│   │   ├── searches.yaml
│   │   ├── llm.yaml                # optional
│   │   ├── .env                    # alice's API keys
│   │   ├── applypilot.db           # PER-USER in v1
│   │   ├── tailored_resumes/
│   │   └── cover_letters/
│   └── bob/...
└── logs/
    └── <run_id>.jsonl
```

**v1.5 (this section) layout** — single shared DB, profiles drop their own:

```
~/.applypilot-saas/
├── shared.db                       # ONE sqlite DB; jobs + user_jobs split
├── profiles/
│   ├── alice/                      # exactly a core APP_DIR minus the DB
│   │   ├── profile.json
│   │   ├── resume.txt
│   │   ├── searches.yaml
│   │   ├── llm.yaml
│   │   ├── .env
│   │   ├── tailored_resumes/
│   │   └── cover_letters/
│   └── bob/...
└── logs/<run_id>.jsonl
```

**Critical property (preserved across v1 → v1.5):** `profiles/alice/` is *exactly* a core `APP_DIR`. A user can author the dir locally with `applypilot init`, zip it, send it; the admin imports it under `profiles/alice/` and runs. Symmetric. v1.5 just promotes `applypilot.db` from per-user to shared — the rest is unchanged.

### 2.3 The schema split — what each column means and where it goes

Today's `jobs` table mixes concerns. Split it.

**`jobs` (shared across all users; PK = `url`):**

| Column | Source | Why shared |
|---|---|---|
| `url` | discovery | dedup key |
| `title`, `salary`, `description`, `location`, `site`, `strategy` | discovery | same for everyone |
| `discovered_at` | discovery | provenance |
| `full_description`, `application_url` | enrichment | same for everyone |
| `detail_scraped_at`, `detail_error` | enrichment | provenance |
| `filter_reason` | rule gate (universal reasons only) | same for everyone — see below |
| `prefiltered_at` | rule gate (universal) | provenance |

**`user_jobs` (per-tenant; PK = `(user_id, url)`; FK to `jobs.url`):**

| Column | Source | Why per-user |
|---|---|---|
| `user_id` | spoke | tenant identity |
| `user_filter_reason`, `user_prefiltered_at` | rule gate (user-rule reasons) | depends on alice's `searches.yaml` |
| `fit_score`, `score_reasoning`, `scored_at` | scoring | uses alice's resume |
| `tailored_resume_path`, `tailored_resume_json_path`, `tailored_at`, `tailor_attempts` | tailoring | per-user output |
| `cover_letter_path`, `cover_letter_at`, `cover_attempts` | cover letter | per-user output |
| `applied_at`, `apply_status`, `apply_error`, `apply_attempts`, `agent_id`, `last_attempted_at`, `apply_duration_ms`, `apply_task_id`, `verification_confidence` | apply | per-user state |

### 2.4 Filter-reason routing — universal vs. user-rule

The audit walked [discovery/filters.py](../src/applypilot/discovery/filters.py), [enrichment/detail.py](../src/applypilot/enrichment/detail.py), [discovery/llm_sweep.py](../src/applypilot/discovery/llm_sweep.py), and [database.py](../src/applypilot/database.py). Today's reason codes split as follows:

**Universal reasons → `jobs.filter_reason`** (true for everyone):
- `invalid_url` — emitted by [enrichment/detail.py:121](../src/applypilot/enrichment/detail.py#L121) when the URL isn't reachable. Same outcome for every user.

**User-rule reasons → `user_jobs.user_filter_reason`** (depends on this user's `searches.yaml` / `profile.json`):
- `excluded_title` — `searches.yaml: exclude_titles`
- `title_not_allowed` — `searches.yaml: title_require_any`
- `seniority_mismatch` — `profile.years_of_experience_total` vs `defaults.seniority_floor_years`
- `country_mismatch` — `defaults.blocked_countries` + `location_accept`/`location_reject_non_remote`
- LLM sweep verdicts — driven by per-user prefilter LLM routing

**Future universal reasons** (not yet implemented but the table allocates space): `not_a_job_posting`, `404`, `manual_ats_host`, `expired_posting`. Add them to `jobs.filter_reason` when the corresponding detection lands.

Codify this map in `applypilot-saas/src/applypilot_saas/filters/reason_map.py` as a literal `dict[str, Literal["jobs", "user_jobs"]]`. Core doesn't need to know — core writes to whichever table it's pointed at via the `Database` provider.

### 2.5 How a multi-user run actually executes

Pseudocode for `applypilot-saas admin run --profiles alice,bob discover enrich score tailor cover`:

```python
shared_db = TenantSqliteDatabase("~/.applypilot-saas/shared.db")  # spoke-side

# 1. Discovery + enrichment — share results across users via the `jobs` table.
#    Use any user's discovery config to drive crawls? No — union the configs.
#    The crawler runs ONCE per (employer, query) tuple, regardless of which
#    users requested it.
discovery_ctx = build_admin_discovery_ctx(
    profiles=["alice", "bob"], db=shared_db,
)
core.pipeline.run(discovery_ctx, stages=["discover", "enrich"])

# 2. Per-user stages — score, tailor, cover are user-specific.
for profile in ["alice", "bob"]:
    user_ctx = build_user_ctx(profile, db=shared_db)
    core.pipeline.run(user_ctx, stages=["score", "tailor", "cover"])
```

Two subtleties this hides:

**Subtlety A: discovery union.** Alice wants ML/Backend roles in NYC; Bob wants Design roles globally. Naively running two discoveries doubles crawl traffic. The spoke union-merges queries before invoking core's `_run_discover`, then the post-discover rule gate runs *per user* — Alice's user_filter_reason is set on the `user_jobs` row even though the `jobs` row was created by a query union that included Bob's terms.

**Subtlety B: the universal rule gate still runs once.** `not_a_job_posting`, `404`, etc. write `jobs.filter_reason` once. The user-rule rule gate runs N times, once per user, writing `user_jobs.user_filter_reason`.

**Subtlety C: per-user LLM keys never pool.** When core's `get_client_for_ctx(ctx, "score")` runs inside Alice's per-user loop, `ctx.user.secrets` is Alice's `LocalSecretsProvider` reading her `.env`. Bob's `.env` is never opened during Alice's scoring. This is already true today — the contract is in [llm.py](../src/applypilot/llm.py)'s `resolve_config_for_ctx` and `get_client_for_ctx`.

### 2.6 The admin CLI surface — what shipped vs. what's left

Status flags: ✅ shipped in v1; ⏳ pending v1.5 (schema split); 🔮 deferred to Phase 2 (backend).

```
✅ applypilot-saas admin onboard <name>
   Re-execs `applypilot init` with APPLYPILOT_DIR pointing at profiles/<name>/.
   Single-sources the wizard in core — no duplicated prompts.

✅ applypilot-saas admin onboard <name> --from-zip ~/Downloads/alice.zip
   Imports a zipped APP_DIR-shaped folder. Tolerates flat or single-rooted zips.
   Validates required files (profile.json, resume.txt, searches.yaml, .env).
   ⏳ Refusing-on-missing-LLM-key check not yet wired; today the import succeeds
   if .env exists at all.

✅ applypilot-saas admin run --profiles alice,bob [stages...]
✅ applypilot-saas admin run --all
   Iterates profiles sequentially. v1: per-profile DBs (no shared dedup).
   ⏳ v1.5: shares jobs/enrichment via shared.db; per-user score/tailor/cover.
   Log lines prefixed [profile=alice]. SIGINT aborts the whole batch in v1
   (the per-user "skip and continue" flow is ⏳ wanted for v1.5 with --halt-on-cancel).

✅ applypilot-saas admin run --profiles alice --dry-run
   Resolves the LLM model per stage per user via resolve_config_for_ctx.
   Prints pending-work counts per profile. Zero LLM calls.

✅ applypilot-saas admin apply --profiles alice,bob [--limit N] [--workers N]
                                [--min-score N] [--model haiku|sonnet]
                                [--continuous] [--headless/--no-headless]
                                [--dry-run] [--url URL]
   Iterates profiles sequentially and invokes core's apply.launcher.main(ctx)
   per profile. Apply is intentionally a sibling subcommand, not a `run`
   stage — its flags (continuous, target URL, model selection) don't map
   onto pipeline-stage semantics. Per-profile isolation is the same as
   `run`: each profile binds its own ctx, its own DB, its own Chrome
   worker dirs (per §6 sharp-edge #8).

✅ applypilot-saas admin export --profile alice [--min-score 8] [--latest]
   Generalized descendant of scripts/export_for_brother.py. Reads the
   profile's per-user DB in v1; ⏳ v1.5 reads shared.db with the join.
   Produces score-binned folders + jobs.csv tracker.
   ⏳ --zip flag for one-shot handoff bundling not yet implemented.

✅ applypilot-saas admin status
   One-line summary per profile: total/scored/≥8/tailored/applied. Profile-scoped.
   ⏳ Spend column blocked on usage_events (v1.5).

⏳ applypilot-saas admin spend [--profile alice] [--since 2026-04-01]
   LLM cost breakdown from usage_events. Blocked on core emitting llm_call
   ProgressReporter events; small core change in §2.8 step 9.
```

`status` and `spend` *must* be profile-scoped by default. A bug here leaks one user's data to the admin's terminal. `--all` is opt-in and aggregated, never per-row. (v1's `status` is already profile-scoped.)

### 2.7 Spoke file layout

**v1 (shipped):**

```
applypilot-saas/
├── pyproject.toml                  # path-deps applypilot via [tool.uv.sources]
├── README.md
├── src/applypilot_saas/
│   ├── __init__.py
│   ├── paths.py                    # APPLYPILOT_SAAS_DIR resolution
│   ├── profile.py                  # validate(name) + build_ctx(name)
│   └── admin/
│       ├── cli.py                  # argparse app: onboard, run, export, status
│       ├── onboard.py              # wizard re-exec + zip import
│       ├── run.py                  # multi-profile loop + dry-run resolver
│       ├── export.py               # export_for_brother.py descendant
│       └── status.py               # one-line-per-profile summary
└── tests/
    ├── conftest.py                 # saas_root + two_profiles fixtures
    ├── test_admin_cli.py           # argparse wiring
    └── test_profile_isolation.py   # the smoke test the plan calls out
```

**v1.5 additions (TODO):**

```
applypilot-saas/src/applypilot_saas/
├── providers/
│   ├── tenant_sqlite.py            # Database protocol; jobs + user_jobs
│   └── tenant_storage.py           # optional; v1 reuses core's LocalStorageBackend
├── db/
│   ├── schema.py                   # CREATE TABLE jobs/user_jobs; indexes
│   └── migrations/                 # alembic-ready (even on sqlite)
├── filters/
│   └── reason_map.py               # _REASON_TO_TABLE literal map (audit done in §2.4)
├── ctx/
│   └── builder.py                  # build_user_ctx(profile, shared_db) → RunContext
└── usage/
    └── tracker.py                  # ProgressReporter → usage_events table
```

### 2.8 Implementation roadmap — what's done and what's left

Steps 1, 5, 6, 7, 8, 10 are done in v1 (per-profile DBs). Steps 2–4 and 9 are the v1.5 work that earns the shared DB and usage tracking. Do v1.5 in order; don't skip ahead.

✅ **Step 1 (DONE in this doc, §2.4): Audit filter reasons.** Reasons were labelled and codified above. Step 1 of v1.5 is just to mirror that map into `applypilot-saas/src/applypilot_saas/filters/reason_map.py` as a literal `dict[str, Literal["jobs", "user_jobs"]]`.

⏳ **Step 2 (v1.5): Write the schema.** `applypilot-saas/src/applypilot_saas/db/schema.py` — `CREATE TABLE jobs (...)` (universal columns only; PK url) and `CREATE TABLE user_jobs (...)` (PK `(user_id, url)`; FK `url`). Indexes on `user_id`, `(user_id, fit_score)`, `(user_id, applied_at)`. Also `usage_events (user_id, run_id, stage, model, prompt_tokens, completion_tokens, est_cost, created_at)` — populated in step 9.

⏳ **Step 3 (v1.5): Build `TenantSqliteDatabase`.** Implements core's `Database` protocol. **Decision (carried from the original plan):** extend `Database` protocol with a `user_jobs_connection()` method that returns the same underlying conn but with `user_id` bound. Core's per-user writers use `Database.user_jobs_connection()`; shared writers use `Database.connection()`. The local-single-user `SqliteDatabase` implements `user_jobs_connection()` as an alias for `connection()` and writes to the unified `jobs` table — single-user CLI unchanged.

⏳ **Step 4 (v1.5): Refactor core's per-user writes.** This is the only v1.5 change to *core*. In [scoring/scorer.py](../src/applypilot/scoring/scorer.py), [scoring/tailor.py](../src/applypilot/scoring/tailor.py), [scoring/cover_letter.py](../src/applypilot/scoring/cover_letter.py), [apply/launcher.py](../src/applypilot/apply/launcher.py): every `UPDATE jobs SET fit_score = ?` (or `tailored_resume_path`, `cover_letter_path`, `applied_at`, etc.) becomes a call to `Database.update_user_job(url, **fields)` on the protocol. Local impl writes to the unified `jobs` table; tenant impl writes to `user_jobs`. **Net core diff: ~50 lines** if done with discipline.

✅ **Step 5 (DONE in v1, will need a v1.5 swap): `build_user_ctx(profile_name, db)`.** v1 uses core's `build_local_run_context(profile_dir)` directly — each profile gets its own DB. v1.5 adds `applypilot_saas.ctx.builder.build_user_ctx(profile, shared_db)` that reads the profile dir but binds the spoke's `TenantSqliteDatabase`. Same `RunContext` shape; different `db` provider.

✅ **Step 6 (DONE in v1): `applypilot-saas admin run`.** Single-profile + multi-profile loop both work. v1.5 swaps the per-profile `build_local_run_context` call for `build_user_ctx(profile, shared_db)` and adds an early "discovery + enrichment runs once across all profiles" branch.

✅ **Step 7 (DONE in v1, will gain `--zip` in v1.5): `applypilot-saas admin export`.** v1 reads each profile's per-user DB. v1.5 swaps the source query for the `jobs JOIN user_jobs WHERE user_id=...` form.

✅ **Step 8 (DONE in v1): dry-run resolver.** `admin run --dry-run` already calls `resolve_config_for_ctx(ctx, task)` for `default`, `score`, `tailor`, `cover` per profile. v1.5 can add token-cost estimates once `usage_events` exists for historical data.

⏳ **Step 9 (v1.5): `usage_events`.** Two parts:
   - **Core change:** add a hook in [llm.py](../src/applypilot/llm.py) at the end of `LLMClient.chat()` that calls `ctx.task.progress.event("llm_call", {...})` if `progress` is set. Payload: `model`, `prompt_tokens`, `completion_tokens`, `task` (the stage), `est_cost`. No-op if `progress is None` — single-user CLI unchanged.
   - **Spoke change:** implement `ProgressReporter` in `applypilot-saas/src/applypilot_saas/usage/tracker.py`. On `event("llm_call", payload)`, INSERT into `usage_events`. Wire it into `RunContext.task.progress` inside `admin run`.

✅ **Step 10 (PARTIALLY DONE in v1): `admin status`.** v1 emits per-profile total/scored/≥8/tailored/applied counts. ⏳ v1.5 adds `spend_this_week` column (blocked on step 9) and the standalone `admin spend` command.

### 2.9 Verification

**v1 (passed on initial ship):**
- ✅ `tests/test_profile_isolation.py::test_two_profiles_have_independent_secrets_storage_db` — two profiles get independent secrets, DB files, cancellation tokens. Writes to alice's DB do not appear in bob's.
- ✅ `admin status` on an empty root returns 0 with a "no profiles" banner. With two profiles, prints exactly two rows, profile-scoped.
- ✅ `admin run --dry-run` resolves the LLM model per stage per profile and performs zero LLM calls.
- ✅ Argparse wiring: `admin run --profiles a,b stages...`, `admin export --profile alice`, `admin status` all parse and dispatch.

**v1.5 (gates the schema split):**
- Onboard alice + bob with different resumes / searches / llm.yaml. Run `admin run --profiles alice,bob discover enrich score tailor cover` against `shared.db`.
  - `jobs` has each url exactly once: `SELECT COUNT(*) FROM jobs GROUP BY url HAVING COUNT(*) > 1` returns empty.
  - `user_jobs` has up to two rows per overlapping url with *different* `fit_score`s, *different* `tailored_resume_path`s.
- A url rejected for alice on geo (`country_mismatch`) is not rejected for bob:
  - `SELECT user_filter_reason FROM user_jobs WHERE user_id='alice' AND url=?` returns `'country_mismatch'`.
  - Same query for bob returns NULL.
- Alice's run only spends on Alice's API key: `SELECT DISTINCT api_key_fingerprint FROM usage_events WHERE user_id='alice'` returns exactly one fingerprint matching `profiles/alice/.env`'s `*_API_KEY`.
- `admin export --profile alice` and `--profile bob` produce bundles with zero shared filenames in `tailored_resumes/`.
- Round-trip: zip alice's profile dir (without `applypilot.db`), copy to a clean machine, `APPLYPILOT_DIR=~/alice_profile applypilot run` reproduces the same scores (LLM nondeterminism allowed — assert score is within ±1).

---

## 3. Phase 2: backend (sketch only)

Triggered by Phase 1.5 hitting a wall: admin's box runs out of disk; users want self-serve; you want to charge.

- Swap `TenantSqliteDatabase` for `PostgresDatabase`. Same `Database` protocol; only the SQL dialect changes (and a few `INSERT OR IGNORE` → `ON CONFLICT DO NOTHING`).
- alembic for migrations (already shaped that way in §2.8 step 2).
- FastAPI routes: `POST /auth/{signup,login}`, `PUT /{profile,resume,searches,secrets}`, `POST /runs`, `GET /runs/{id}`, `GET /jobs`, `POST /jobs/{url}/apply-result`, `POST /jobs/capture`.
- Worker pool: Postgres `SELECT … FOR UPDATE SKIP LOCKED` to start. Each task hydrates a `UserContext` from Postgres + KMS + S3, calls `applypilot.core.pipeline.run(ctx, stages=[...])`. Cancellation: `DELETE /runs/{id}` writes a token row that workers' `CancellationToken` impls poll.
- Secrets: KMS-encrypted rows in `user_secrets`. Write-only API. In-memory decrypt cache per request.
- S3/R2 for artifacts. `s3://bucket/{user_id}/tailored/{hash}.pdf`.

The single change to core that Phase 2 forces: `Database.update_user_job` (added in Phase 1.5 step 4) needs to handle Postgres-specific bits — but if step 4 was done right with parameterized SQL, this is zero core-side work.

## 4. Phase 3: local agent + browser extension (sketch only)

Triggered by users wanting auto-apply (Tier 3) and auth-gated discovery (LinkedIn, Workday SSO) without uploading session cookies.

- Local daemon polls backend for tasks the server can't run:
  - **Auto-apply:** server marks `user_jobs.apply_status='apply_queued'`. Agent runs [apply/launcher.py](../src/applypilot/apply/launcher.py) against local Chrome. Posts result.
  - **Auth-gated discover/enrich:** server emits `discover_task` the agent claims; scrapes LinkedIn/Workday-SSO using user's local browser session.
- Browser extension (lighter alternative): captures the current tab's job posting (URL + DOM title/description), `POST /jobs/capture`.

Server-side workers continue to handle public ATS feeds (Greenhouse, SmartRecruiters) in parallel.

---

## 5. Non-goals (these are temptations to refuse)

- **Multi-user logic in this repo.** If it's tenant-aware, it lives in the spoke. The moment a `user_id` parameter shows up in a core function, something has gone wrong.
- **Server-side browser automation.** No headless-Chrome-as-a-service, no cookie upload. Local agent owns anything that needs a logged-in browser.
- **Per-tenant infrastructure.** SaaS posture: pool everything; isolate at row + secret + artifact boundaries.
- **Reuniting core and spoke once both exist.** No shared `applypilot.common` package, no merge. Two repos, one library boundary, one direction of dependency.
- **A `saas_mode` flag in core.** If it would only make sense in a multi-user world, it belongs in the spoke. Period.

---

## 6. Sharp edges that will bite if you forget

1. **`Database.connection()` returns a per-thread sqlite conn.** When the spoke's worker pool is multi-threaded, each worker thread gets its own conn from `SqliteDatabase._local`. Don't try to share a conn across workers; let the cache do its job.
2. **`SqliteDatabase` enables WAL.** That's fine for one process with N threads; not fine if two `applypilot-saas admin run` processes ever target the same `shared.db`. Add file-locking at the spoke level (or just refuse to start a second process — `flock` on `shared.db.lock`).
3. **Stage-level cancellation must propagate inside per-job loops.** Every per-job loop in `scoring/`, `tailor.py`, `cover_letter.py` already does `if ctx.task.cancellation.is_set(): break`. Don't regress this — it's the only thing that lets a hung admin run stop cleanly.
4. **`get_client_for_ctx` does NOT cache.** Two users *must not* share an LLM client. If you ever cache, key on `(ctx.user.user_id, task)`, not on `task` alone.
5. **The single-user CLI's `_bootstrap()` calls `build_default_run_context()` which reads the module-global `stop_event`.** That's the SIGINT bridge. The spoke's workers should NOT reuse that — they build fresh `LocalCancellationToken`s per task and flip them in response to API cancellation requests.
6. **`searches.yaml` `discover_backends` is enforced in core.** When the spoke unions queries across users, it has to also union (intersect, actually) backends. If alice has `[greenhouse]` and bob has `[workday]`, the union run needs both — but the *per-user* rule gate then sees rows that came from a backend they didn't ask for. Solution: don't filter by `discover_backends` at the user level during a shared run; treat it as advisory metadata only when the spoke is driving.
7. **Validator paths in [scoring/validator.py](../src/applypilot/scoring/validator.py)** — preserved companies, schools, metrics — come from each user's `profile.json`. They're already threaded via ctx. Don't accidentally cache validators across users.
8. **`apply/launcher.py` writes Chrome worker dirs under the user's APP_DIR.** When the spoke runs auto-apply for alice, those dirs are under `profiles/alice/.chrome-workers/`. Don't pool them; one user's session never touches another's profile dir.

---

## 7. Open questions

**Decided in v1:**

- ~~**Profile-dir adapter shape.**~~ ✅ Resolved by *not* adapting. v1's profiles use core's exact `APP_DIR` shape (`tailored_resumes/` and `cover_letters/` at the profile root, not under `artifacts/`). One less translation layer; the spoke's `profile.build_ctx` is a one-line delegation to `build_local_run_context`.
- ~~**Per-stage workers count.**~~ ✅ Resolved: thread it through `pipeline.run_pipeline(workers=...)` per profile; no new ctx field. `admin run --workers N` applies per profile (sequential profiles → no contention).
- ~~**Failure isolation across users.**~~ ✅ Default: yes, log and continue. v1 catches `Exception` in the per-profile loop and proceeds; `--halt-on-error` is an open feature add when someone needs it.

**Still open for v1.5:**

1. **Discovery union semantics.** When alice's `searches.yaml` says "ML, Remote, US" and bob's says "Design, Remote, EU", should the shared discover run a *union* (broader crawl, more rule-gate work) or *intersection* (fewer overlapping rows, more per-user crawls)? Default to union — disk is cheap, LLM-sweep cost is bounded by the rule gate.
2. **`usage_events` emission point.** Today core's LLM client doesn't emit progress events. Add a hook in `LLMClient.chat()` that calls `ctx.task.progress.event("llm_call", payload)`? Or wrap every `client.chat()` callsite? Cleanest is the wrapper inside `chat()` — one place, easy to test, no-op when `progress is None`.
3. **Concurrency across profiles.** v1 runs profiles sequentially. v1.5 with shared.db could parallelize cleanly *only* for stages that touch `user_jobs` (score/tailor/cover); discovery + enrichment must stay single-writer to avoid url-dedup races. Worth doing only if measured serial latency is the bottleneck.

---

## 8. What "done" looks like for each phase

**Phase 1 v1 — DONE.**

- ✅ Two profiles run end-to-end through the spoke's `admin run`, each with their own DB / secrets / LLM client / cancellation token.
- ✅ `admin export --profile <name>` produces score-binned folders + jobs.csv per profile.
- ✅ `admin status` and `admin run --dry-run` work without LLM calls.
- ✅ `applypilot-saas` is installable in a clean venv with `pip install -e ../applypilot-alok && pip install -e ../applypilot-saas` and nothing else.
- ✅ This file documents the divergence between what the original plan called for and what v1 actually shipped.

**Phase 1.5 (schema split) — done when:**

- Two real users (you + one other person) get score-binned result zips from a single `admin run` invocation against `shared.db`.
- The `shared.db` has one row per `jobs.url` and N rows per `(user_jobs.user_id, url)`.
- Per-user spend numbers are accurate to within 5% of provider dashboards.
- The round-trip test (zip a profile *without its DB*, run on a fresh machine, get the same scores ±1) passes.
- This file gets updated to mark Phase 1.5 complete with the same "what changed during implementation" honesty applied here.

That last bullet is a feature, not housekeeping. The architecture *will* change during implementation. Update this file in the same PR; don't let documentation drift become the reason a future contributor doesn't trust it.
