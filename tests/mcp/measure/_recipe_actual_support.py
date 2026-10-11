"""Observe native values in an unchanged recording publication after a recipe.

These observations are not fixed Run captures and carry no source metadata.
"""

from collections.abc import Mapping
from typing import TypedDict


class RecordedSweep(TypedDict):
    """Native resolved sweep coordinates and integer count in a recording cfg.

    start/stop are finite coordinate values in the recipe's native unit; expts is
    the GUI's resolved integer point count. No provenance or conversion is added.
    """

    start: int | float
    stop: int | float
    expts: int


def cfg_value(
    publication: Mapping[str, object], field: str
) -> str | bool | int | float | None:
    """Read a resolved scalar or selected reference in the current recording cfg.

    publication is the existing collaborator's public publication, unchanged
    since the tested recipe finished. field is a dotted native scalar/reference
    path. Do not use after live cfg changes to assert an earlier Run snapshot.
    No RPC or provenance is synthesized. Unknown paths raise KeyError; malformed
    fixture data or a nonscalar field raises ValueError.
    """
    node = _node(publication, field)
    if node["kind"] == "reference":
        value = node.get("ref")
    elif node["kind"] == "scalar":
        value = _mapping(node["input"])["resolved"]
    else:
        raise ValueError(f"Not a scalar/reference field: {field}")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("Expected scalar recording cfg value")


def cfg_sweep(publication: Mapping[str, object], field: str) -> RecordedSweep:
    """Read resolved start/stop/expts in the current recording sweep publication.

    publication and field follow cfg_value's observation-time contract. field
    must be a dotted native sweep path. Return a typed coordinate/count mapping,
    not an actual capture or source projection. Missing paths raise KeyError;
    malformed numeric inputs or a nonsweep field raise ValueError. No RPC runs.
    """
    node = _node(publication, field)
    if node["kind"] != "sweep":
        raise ValueError(f"Not a sweep field: {field}")
    inputs = _mapping(node["inputs"])
    start = _mapping(inputs["start"])["resolved"]
    stop = _mapping(inputs["stop"])["resolved"]
    expts = _mapping(inputs["expts"])["resolved"]
    if not isinstance(start, (int, float)) or not isinstance(stop, (int, float)):
        raise ValueError("Expected numeric recording sweep endpoints")
    if not isinstance(expts, int):
        raise ValueError("Expected integer recording sweep count")
    return RecordedSweep(start=start, stop=stop, expts=expts)


def _node(publication: Mapping[str, object], field: str) -> Mapping[str, object]:
    node = _mapping(publication["tree"])
    for part in field.split("."):
        node = _mapping(_mapping(node["children"])[part])
    return node


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Malformed recording cfg node")
    return {key: member for key, member in value.items() if isinstance(key, str)}
