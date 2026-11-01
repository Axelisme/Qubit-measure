"""Apply the simulation marker to every test owned by this directory."""

from pathlib import Path

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark this subtree before pytest applies command-line marker selection."""
    simulation_tests = Path(__file__).parent
    for item in items:
        if item.path.is_relative_to(simulation_tests):
            item.add_marker(pytest.mark.sim)
