"""Apply the simulation marker to every test owned by this directory."""

import pytest


def pytest_itemcollected(item: pytest.Item) -> None:
    """Mark each item through its path-scoped hook before marker selection."""
    item.add_marker(pytest.mark.sim)
