from __future__ import annotations

from applypilot.scoring import scorer


class _CaptureClient:
    def __init__(self) -> None:
        self.captured: list[list[dict]] = []

    def chat(self, messages, max_output_tokens=1024):  # noqa: ARG002
        self.captured.append(list(messages))
        return "SCORE: 5\nKEYWORDS: a, b\nREASONING: test"


def test_score_job_emits_three_messages_with_cache_markers(monkeypatch) -> None:
    cap = _CaptureClient()

    job = {
        "title": "Engineer",
        "site": "Example",
        "location": "Remote",
        "full_description": "Build stuff.",
    }
    scorer.score_job("RESUME BODY", job, client=cap)

    assert len(cap.captured) == 1
    msgs = cap.captured[0]
    assert len(msgs) == 3

    # System prompt is cached
    assert msgs[0]["role"] == "system"
    assert msgs[0].get("cache") == "ephemeral"

    # Resume is cached
    assert msgs[1]["role"] == "user"
    assert "RESUME:" in msgs[1]["content"]
    assert msgs[1].get("cache") == "ephemeral"

    # Per-job block is NOT cached
    assert msgs[2]["role"] == "user"
    assert "JOB POSTING:" in msgs[2]["content"]
    assert "cache" not in msgs[2]


def test_score_job_keeps_system_and_resume_byte_identical_across_jobs(monkeypatch) -> None:
    """Regression guard: if per-job content leaks into the cached blocks,
    the cache-read path never fires and we pay full price every call."""
    cap = _CaptureClient()

    job_a = {"title": "A", "site": "X", "location": "L1", "full_description": "aaa"}
    job_b = {"title": "B", "site": "Y", "location": "L2", "full_description": "bbb"}

    scorer.score_job("RESUME", job_a, client=cap)
    scorer.score_job("RESUME", job_b, client=cap)

    assert cap.captured[0][0]["content"] == cap.captured[1][0]["content"]
    assert cap.captured[0][1]["content"] == cap.captured[1][1]["content"]
    # Per-job differs
    assert cap.captured[0][2]["content"] != cap.captured[1][2]["content"]
