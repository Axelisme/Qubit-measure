"""Fixtures for the real Fluxdep RPC route seam."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from zcu_tools.gui.app.fluxdep.controller import Controller

from tests.gui.app.fluxdep._components import skip_calibration
from tests.gui.app.fluxdep.remote._route_harness import RouteHarness, copy_sources


@pytest.fixture
def route_harness(tmp_path: Path) -> Iterator[RouteHarness]:
    """Create an independent owner-turn adapter with native project-root paths."""
    harness = RouteHarness.create(str(tmp_path), export_planner=skip_calibration)
    yield harness
    harness.close()


@pytest.fixture
def one_route(
    route_harness: RouteHarness, onetone_controller: Controller
) -> RouteHarness:
    return copy_sources(route_harness, onetone_controller)


@pytest.fixture
def two_route(
    route_harness: RouteHarness, twotone_controller: Controller
) -> RouteHarness:
    return copy_sources(route_harness, twotone_controller)


@pytest.fixture
def joint_route(
    route_harness: RouteHarness, cross_controller: Controller
) -> RouteHarness:
    return copy_sources(route_harness, cross_controller)
