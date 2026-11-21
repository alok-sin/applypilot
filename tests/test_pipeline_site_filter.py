from __future__ import annotations

from pathlib import Path

import applypilot.pipeline as pipeline
from applypilot.core.local_providers import build_local_run_context


def test_run_discover_with_site_filter_only_runs_matching_smart_extract(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        pipeline,
        "console",
        type("DummyConsole", (), {"print": staticmethod(lambda *args, **kwargs: None)})(),
    )

    # Build the stage's ctx ourselves over a tmp dir: the test must not read
    # the developer's real APP_DIR, and the ctx's sqlite connection must not
    # be leaked for the GC finalizer to complain about later.
    ctx = build_local_run_context(tmp_path)
    monkeypatch.setattr(pipeline, "build_default_run_context", lambda **_: ctx)

    def _fake_load_sites():
        return [
            {"name": "Lensa", "url": "https://lensa.example", "type": "search"},
            {"name": "Dice", "url": "https://dice.example", "type": "search"},
        ]

    def _fake_run_smart_extract(*, sites, workers, ctx=None):
        captured["sites"] = sites
        captured["workers"] = workers

    monkeypatch.setattr("applypilot.discovery.smartextract.load_sites", _fake_load_sites)
    monkeypatch.setattr("applypilot.discovery.smartextract.run_smart_extract", _fake_run_smart_extract)

    result = pipeline._run_discover(workers=2, site_filter=["Lensa"])

    assert captured["sites"] == [{"name": "Lensa", "url": "https://lensa.example", "type": "search"}]
    assert captured["workers"] == 2
    assert result["jobspy"] == "skipped (site-filter)"
    assert result["workday"] == "skipped (site-filter)"
    assert result["smartextract"] == "ok"
    assert result["greenhouse"] == "skipped (site-filter)"

    ctx.user.db.close()
