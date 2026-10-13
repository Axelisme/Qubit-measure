"""Observe ordinary Form leaves after a recipe, not a prior Run capture."""

from collections.abc import Mapping
from typing import TypedDict


class RecordedSweep(TypedDict):
    """Resolved range endpoints and point count in caller units."""

    start: int | float
    stop: int | float
    expts: int


def cfg_value(
    publication: Mapping[str, object], field: str
) -> str | bool | int | float | None:
    """Read a current evaluated leaf or selected actual module key."""
    values = publication["values"]
    if not isinstance(values, list):
        raise ValueError("Malformed Form values")
    for leaf in values:
        if not isinstance(leaf, dict):
            raise ValueError("Malformed Form value")
        if leaf["path"] == field.split("."):
            value = leaf["value"]
            if value is None or isinstance(value, (str, bool, int, float)):
                return value
            raise ValueError("Expected scalar recording Form value")
    raw: object = publication["input"]
    for part in field.split("."):
        if not isinstance(raw, dict):
            raise ValueError("Expected Form input model")
        raw = raw.get(part)
    if isinstance(raw, dict) and raw.get("kind") == "ref":
        key = raw.get("name")
        if isinstance(key, str):
            return key
    if raw is None:
        return None
    raise ValueError(f"Not a scalar/key field: {field}")


def cfg_sweep(publication: Mapping[str, object], field: str) -> RecordedSweep:
    """Read resolved start/stop/n leaves; expose the author's count as expts."""
    start = cfg_value(publication, f"{field}.start")
    stop = cfg_value(publication, f"{field}.stop")
    n = cfg_value(publication, f"{field}.n")
    if not isinstance(start, (int, float)) or not isinstance(stop, (int, float)):
        raise ValueError("Expected numeric recording sweep endpoints")
    if not isinstance(n, int):
        raise ValueError("Expected integer recording sweep count")
    return {"start": start, "stop": stop, "expts": n}
