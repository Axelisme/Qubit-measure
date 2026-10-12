"""Typed access to the synthetic nested numeric form input."""

from collections.abc import Mapping

from pydantic import JsonValue


def sample_input(data: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Require the captured sample subtree without changing values."""
    child = data["sample"]
    assert isinstance(child, dict)
    return child
