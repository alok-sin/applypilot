"""Shared test fixtures."""

from __future__ import annotations

from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _restore_litellm_suppress_debug_info() -> Iterator[None]:
    """Save/restore ``litellm.suppress_debug_info`` around every test.

    Every ``LLMClient.__init__`` sets this litellm process global to True.
    Without restoration the mutation leaks between tests (and into later
    test files), so snapshot it before each test and put the original value
    back afterwards. litellm is imported lazily here to keep collection of
    runs that never touch the LLM stack free of the litellm import cost.
    """
    import litellm

    original = litellm.suppress_debug_info
    yield
    litellm.suppress_debug_info = original
