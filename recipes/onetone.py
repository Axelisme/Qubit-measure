"""Single-run onetone recipes using GUI-owned calibration and defaults."""

import re
from math import isfinite
from typing import Any, TypeGuard

from zcu_tools.mcp.measure.recipe_context import MissingParameter, RecipeContext
from zcu_tools.mcp.measure.session import GuiRpcError


def _finite(value: object) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(value)
    )


def _validate(arguments: dict[str, Any]) -> None:
    for name in ("reuse_tab_id", "readout_ref"):
        value = arguments.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a non-empty string or null")
    for name in ("center_mhz", "span_mhz", "gain"):
        value = arguments.get(name)
        if value is not None and not _finite(value):
            raise ValueError(f"{name} must be a finite real number or null")
    if arguments.get("span_mhz") is not None and arguments["span_mhz"] <= 0:
        raise ValueError("span_mhz must be positive")
    for name in ("points", "reps", "rounds"):
        value = arguments.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError(f"{name} must be an integer or null")


def _node(publication: dict[str, Any], *path: str) -> dict[str, Any]:
    node = publication["tree"]
    for part in path:
        node = node["children"][part]
    return node


def _input_value(value: dict[str, Any]) -> Any:
    return {"__expr": value["raw"]} if value["mode"] == "expression" else value["raw"]


def _frequency(
    publication: dict[str, Any], arguments: dict[str, Any], md: dict[str, Any]
) -> tuple[dict[str, Any], list[MissingParameter]]:
    missing = []
    center = arguments.get("center_mhz")
    if center is None:
        if not _finite(md.get("r_f")):
            missing.append(MissingParameter("center_mhz", "No finite r_f calibration"))
        center = "r_f"
    span = arguments.get("span_mhz")
    inputs = _node(publication, "sweep", "freq")["inputs"]
    if span is None:
        width = md.get("rf_w")
        edges = [inputs[key] for key in ("start", "stop")]
        if (
            not _finite(width)
            or width <= 0
            or any(
                edge["mode"] != "expression" or not re.search(r"\brf_w\b", edge["raw"])
                for edge in edges
            )
        ):
            missing.append(
                MissingParameter("span_mhz", "No GUI linewidth-derived range")
            )
            return {}, missing
        # Remove only the known center identifier; the GUI remains the width owner.
        start, stop = [re.sub(r"\br_f\b", "0", edge["raw"]) for edge in edges]
        span = f"({stop}) - ({start})"
    return {
        "start": {"__expr": f"({center}) - ({span}) / 2"},
        "stop": {"__expr": f"({center}) + ({span}) / 2"},
        "expts": arguments["points"]
        if arguments.get("points") is not None
        else _input_value(inputs["expts"]),
    }, missing


def onetone_spectrum_over_flux(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Run one frequency/physical-flux survey with Primary analysis."""
    _validate(arguments)
    sources = ctx.rpc("context.snapshot", {})
    publication = ctx.prepare_tab("onetone/flux_dep", arguments.get("reuse_tab_id"))
    _, missing = _frequency(
        publication,
        {**arguments, "points": arguments.get("freq_points")},
        sources["md"],
    )
    device = arguments.get("flux_device")
    if device is None:
        values = ctx.rpc("value.list", {})["values"]
        if any(value["key"] == "device.flux.name" for value in values):
            device = ctx.rpc("value.read", {"key": "device.flux.name"})["value"]
    if not isinstance(device, str) or not device.strip():
        missing.append(
            MissingParameter("flux_device", "No registered flux device source")
        )
    else:
        snapshot = ctx.rpc("device.snapshot", {"name": device})["snapshot"]
        unit = snapshot["unit"]
        if not isinstance(unit, str) or not unit.strip() or unit == "none":
            if arguments.get("flux_device") is not None:
                raise GuiRpcError(
                    "Flux device has no physical unit", reason="invalid_device"
                )
            missing.append(
                MissingParameter("flux_device", "Flux device has no physical unit")
            )
    md = sources["md"]
    if arguments.get("flux_range") is None and (
        not _finite(md.get("flx_half"))
        or not _finite(md.get("flx_int"))
        or md["flx_half"] == md["flx_int"]
    ):
        missing.append(
            MissingParameter("flux_range", "No distinct calibrated flux endpoints")
        )
    if missing:
        ctx.needs_parameters(missing)
        return
    raise NotImplementedError("Onetone flux Run preparation is not implemented")


def onetone_spectrum(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Prepare a calibrated spectrum, run once and save raw and analysis."""
    _validate(arguments)
    sources = ctx.rpc("context.snapshot", {})
    publication = ctx.prepare_tab("onetone/freq", arguments.get("reuse_tab_id"))
    if arguments.get("readout_ref") is not None:
        key = arguments["readout_ref"]
        publication = ctx.edit_cfg(
            publication, [{"path": ["modules", "readout"], "value": {"__ref": key}}]
        )
        reference = _node(publication, "modules", "readout")
        if reference.get("error") or reference.get("ref") != key:
            raise GuiRpcError("Invalid readout reference", reason="invalid_cfg")
    frequency, missing = _frequency(publication, arguments, sources["md"])
    if missing:
        ctx.needs_parameters(missing)
        return
    edits = [{"path": ["sweep", "freq"], "value": frequency}]
    paths = {
        "gain": ("modules", "readout", "pulse_cfg", "gain"),
        "reps": ("reps",),
        "rounds": ("rounds",),
    }
    for name, path in paths.items():
        if arguments.get(name) is not None:
            edits.append({"path": list(path), "value": arguments[name]})
    publication = ctx.edit_cfg(publication, edits)
    if publication["status"] != "Valid":
        raise GuiRpcError("Onetone cfg is not Valid", reason="invalid_cfg")
    inputs = _node(publication, "sweep", "freq")["inputs"]
    values = {key: inputs[key]["resolved"] for key in ("start", "stop", "expts")}
    if (
        not all(_finite(values[key]) for key in ("start", "stop"))
        or values["stop"] <= values["start"]
    ):
        raise GuiRpcError("Invalid resolved frequency range", reason="invalid_cfg")
    fields = {
        "sweep.freq": {"value": values, "input": inputs},
        "center_mhz": {
            "value": (values["start"] + values["stop"]) / 2,
            "source": "explicit" if arguments.get("center_mhz") is not None else "r_f",
        },
        "span_mhz": {
            "value": values["stop"] - values["start"],
            "source": "explicit"
            if arguments.get("span_mhz") is not None
            else "gui_linewidth",
        },
        "modules.readout": {
            "value": _node(publication, "modules", "readout").get("ref"),
            "source": "explicit"
            if arguments.get("readout_ref") is not None
            else "gui_default",
        },
    }
    for name, path in paths.items():
        input_state = _node(publication, *path)["input"]
        fields[".".join(path)] = {
            "value": input_state["resolved"],
            "input": input_state,
            "source": "explicit" if arguments.get(name) is not None else "gui_default",
        }
    ctx.run_once(publication, fields)
