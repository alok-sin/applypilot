from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from typer.testing import CliRunner

import applypilot.cli as cli
from applypilot.database import close_connection, init_db


runner = CliRunner()


def _fake_ctx(conn):
    class _FakeDB:
        def __init__(self, c):
            self._c = c

        def connection(self):
            return self._c

    return SimpleNamespace(
        user=SimpleNamespace(db=_FakeDB(conn), profile={}, search_config={}, resume_text=""),
        task=SimpleNamespace(),
    )


def _make_test_dir() -> Path:
    root = Path.cwd() / ".test_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / str(uuid4())
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_add_url_inserts_job_with_defaults(monkeypatch) -> None:
    tmp_root = _make_test_dir()
    db_path = tmp_root / "applypilot.db"

    try:
        conn = init_db(db_path)
        monkeypatch.setattr(cli, "_bootstrap", lambda: _fake_ctx(conn))

        target = "https://example.com/jobs/123"
        result = runner.invoke(cli.app, ["add-url", target, "--title", "Example Job"])

        assert result.exit_code == 0
        row = conn.execute(
            "SELECT url, title, site, strategy, application_url FROM jobs WHERE url = ?",
            (target,),
        ).fetchone()
        assert row is not None
        assert row["url"] == target
        assert row["title"] == "Example Job"
        assert row["site"] == "Manual"
        assert row["strategy"] == "manual_url"
        assert row["application_url"] == target
    finally:
        close_connection(db_path)
        shutil.rmtree(tmp_root, ignore_errors=True)
