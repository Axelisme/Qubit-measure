"""Typed access to the relocated T1 lifetime subtree of public form input."""

from collections.abc import Mapping

from pydantic import JsonValue


def t1_input(data: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Require the captured Coherence T1 knob subtree without changing values."""
    child = data["t1"]
    assert isinstance(child, dict)
    return child
