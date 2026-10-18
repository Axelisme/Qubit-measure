"""Typed access to the synthetic nested numeric form input."""

from collections.abc import Mapping

from pydantic import JsonValue


def draft_data(envelope: Mapping[str, JsonValue] | None) -> dict[str, JsonValue]:
    """Read raw input from a captured versioned FormInput envelope."""
    assert envelope is not None
    data = envelope["data"]
    assert isinstance(data, dict)
    return data


def sample_input(data: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Require the captured sample subtree without changing values."""
    child = data["sample"]
    assert isinstance(child, dict)
    return child
