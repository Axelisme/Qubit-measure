"""Fixtures for the real Fluxdep RPC route seam."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness


@pytest.fixture
def route_harness(tmp_path: Path) -> Iterator[RouteHarness]:
    """Create an independent owner-turn adapter with native project-root paths."""
    harness = RouteHarness.create(str(tmp_path))
    yield harness
    harness.close()
