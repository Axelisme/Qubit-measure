"""Shared calibration-source decisions over GUI cfg publications."""

from math import isfinite
from typing import Any, TypeGuard

from zcu_tools.mcp.measure.recipe_context import MissingParameter
from zcu_tools.mcp.measure.session import GuiRpcError


def finite_number(value: object) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(value)
    )


def cfg_node(publication: dict[str, Any], *path: str) -> dict[str, Any]:
    node = publication["tree"]
    for part in path:
        node = node["children"][part]
    return node


def usable_frequency(node: dict[str, Any]) -> bool:
    value = node.get("input", {})
    return bool(
        node.get("valid")
        and not value.get("error")
        and not value.get("validation_error")
        and finite_number(value.get("resolved"))
    )


def readout_frequency(
    publication: dict[str, Any], md: dict[str, Any], modules: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[tuple[str, ...], str], list[MissingParameter]]:
    readout = cfg_node(publication, "modules", "readout")
    if "ro_freq" in readout["children"]:
        tails = (("ro_freq",),)
    elif "pulse_cfg" in readout["children"] and "ro_cfg" in readout["children"]:
        tails = (("pulse_cfg", "freq"), ("ro_cfg", "ro_freq"))
    else:
        raise GuiRpcError("Unsupported readout shape", reason="invalid_cfg")
    edits = []
    origins = {}
    missing = []
    for tail in tails:
        path = ("modules", "readout", *tail)
        if readout.get("ref") in modules and usable_frequency(
            cfg_node(publication, *path)
        ):
            origins[path] = f"library:{readout['ref']}"
        elif finite_number(md.get("r_f")):
            edits.append({"path": list(path), "value": {"__expr": "r_f"}})
            origins[path] = "r_f"
        elif not missing:
            missing.append(
                MissingParameter(
                    "readout_ref", "Provide a valid readout_ref or calibrated r_f"
                )
            )
    return edits, origins, missing
