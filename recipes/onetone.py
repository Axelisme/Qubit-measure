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
    for name in ("reuse_tab_id", "readout_ref", "flux_device"):
        value = arguments.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a non-empty string or null")
    for name in ("center_mhz", "span_mhz", "gain"):
        value = arguments.get(name)
        if value is not None and not _finite(value):
            raise ValueError(f"{name} must be a finite real number or null")
    if arguments.get("span_mhz") is not None and arguments["span_mhz"] <= 0:
        raise ValueError("span_mhz must be positive")
    for name in ("points", "freq_points", "flux_points", "reps", "rounds"):
        value = arguments.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError(f"{name} must be an integer or null")
    value = arguments.get("flux_range")
    if value is not None and (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(_finite(endpoint) for endpoint in value)
    ):
        raise ValueError("flux_range must contain two finite real endpoints")


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


def _flux_device(
    ctx: RecipeContext, arguments: dict[str, Any], missing: list[MissingParameter]
) -> tuple[str | None, str | None]:
    unit: str | None = None
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
    return device if isinstance(device, str) else None, unit


def onetone_spectrum_over_flux(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Run one frequency/physical-flux survey with Primary analysis."""
    _validate(arguments)
    sources = ctx.rpc("context.snapshot", {})
    publication = ctx.prepare_tab("onetone/flux_dep", arguments.get("reuse_tab_id"))
    publication = _select_readout(ctx, publication, arguments)
    frequency, missing = _frequency(
        publication,
        {**arguments, "points": arguments.get("freq_points")},
        sources["md"],
    )
    device, unit = _flux_device(ctx, arguments, missing)
    md = sources["md"]
    if arguments.get("flux_range") is None and (
        not _finite(md.get("flx_half"))
        or not _finite(md.get("flx_int"))
        or md["flx_half"] == md["flx_int"]
    ):
        missing.append(
            MissingParameter("flux_range", "No distinct calibrated flux endpoints")
        )
    flux: dict[str, Any] = {}
    if not any(item.parameter == "flux_range" for item in missing):
        inputs = _node(publication, "sweep", "flux")["inputs"]
        explicit_range = arguments.get("flux_range")
        if explicit_range is None:
            edges = [inputs[key] for key in ("start", "stop")]
            if any(
                edge["mode"] != "expression"
                or set(re.findall(r"\bflx_(?:half|int)\b", edge["raw"]))
                != {"flx_half", "flx_int"}
                for edge in edges
            ):
                missing.append(
                    MissingParameter("flux_range", "No GUI calibration-derived range")
                )
            flux = {key: _input_value(inputs[key]) for key in ("start", "stop")}
        else:
            flux = dict(zip(("start", "stop"), explicit_range, strict=True))
        flux["expts"] = (
            arguments["flux_points"]
            if arguments.get("flux_points") is not None
            else _input_value(inputs["expts"])
        )
    if missing:
        ctx.needs_parameters(missing)
        return
    publication = ctx.edit_cfg(
        publication,
        [
            {"path": ["sweep", "freq"], "value": frequency},
            {"path": ["sweep", "flux"], "value": flux},
            {"path": ["dev", "flux_dev"], "value": device},
            *_scalar_edits(arguments),
        ],
    )
    fields = _actual_fields(publication, arguments)
    inputs = _node(publication, "sweep", "flux")["inputs"]
    values = {key: inputs[key]["resolved"] for key in ("start", "stop", "expts")}
    if (
        not all(_finite(values[key]) for key in ("start", "stop"))
        or values["start"] == values["stop"]
    ):
        raise GuiRpcError("Invalid resolved flux range", reason="invalid_cfg")
    fields["sweep.flux"] = {
        "value": values,
        "input": inputs,
        "source": "explicit"
        if arguments.get("flux_range") is not None
        else "gui_calibration",
    }
    fields["dev.flux_dev"] = {
        "value": _node(publication, "dev", "flux_dev")["input"]["resolved"],
        "unit": unit,
        "source": "explicit"
        if arguments.get("flux_device") is not None
        else "device.flux.name",
    }
    ctx.run_once(publication, fields)


def _select_readout(
    ctx: RecipeContext, publication: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    if arguments.get("readout_ref") is not None:
        key = arguments["readout_ref"]
        publication = ctx.edit_cfg(
            publication, [{"path": ["modules", "readout"], "value": {"__ref": key}}]
        )
        reference = _node(publication, "modules", "readout")
        if reference.get("error") or reference.get("ref") != key:
            raise GuiRpcError("Invalid readout reference", reason="invalid_cfg")
    return publication


_SCALAR_PATHS = {
    "gain": ("modules", "readout", "pulse_cfg", "gain"),
    "reps": ("reps",),
    "rounds": ("rounds",),
}


def _scalar_edits(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"path": list(path), "value": arguments[name]}
        for name, path in _SCALAR_PATHS.items()
        if arguments.get(name) is not None
    ]


def _actual_fields(
    publication: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, Any]:
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
    for name, path in _SCALAR_PATHS.items():
        input_state = _node(publication, *path)["input"]
        fields[".".join(path)] = {
            "value": input_state["resolved"],
            "input": input_state,
            "source": "explicit" if arguments.get(name) is not None else "gui_default",
        }
    return fields


def onetone_spectrum_over_power(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Save one frequency/gain survey and return its Run preview without analysis."""
    _validate(arguments)
    sources = ctx.rpc("context.snapshot", {})
    publication = ctx.prepare_tab("onetone/power_dep", arguments.get("reuse_tab_id"))
    publication = _select_readout(ctx, publication, arguments)
    frequency, missing = _frequency(
        publication,
        {**arguments, "points": arguments.get("freq_points")},
        sources["md"],
    )
    if missing:
        ctx.needs_parameters(missing)
        return
    inputs = _node(publication, "sweep", "gain")["inputs"]
    explicit_range = arguments.get("gain_range")
    gain: dict[str, Any] = (
        {key: _input_value(inputs[key]) for key in ("start", "stop")}
        if explicit_range is None
        else dict(zip(("start", "stop"), explicit_range, strict=True))
    )
    gain["expts"] = (
        arguments["gain_points"]
        if arguments.get("gain_points") is not None
        else _input_value(inputs["expts"])
    )
    publication = ctx.edit_cfg(
        publication,
        [
            {"path": ["sweep", "freq"], "value": frequency},
            {"path": ["sweep", "gain"], "value": gain},
            *_scalar_edits(arguments),
        ],
    )
    fields = _actual_fields(publication, arguments)
    inputs = _node(publication, "sweep", "gain")["inputs"]
    fields["sweep.gain"] = {
        "value": {key: inputs[key]["resolved"] for key in ("start", "stop", "expts")},
        "input": inputs,
        "source": "explicit" if explicit_range is not None else "gui_default",
    }
    ctx.run_once(publication, fields, analysis_mode="none")


def onetone_spectrum(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Prepare a calibrated spectrum, run once and save raw and analysis."""
    _validate(arguments)
    sources = ctx.rpc("context.snapshot", {})
    publication = ctx.prepare_tab("onetone/freq", arguments.get("reuse_tab_id"))
    publication = _select_readout(ctx, publication, arguments)
    frequency, missing = _frequency(publication, arguments, sources["md"])
    if missing:
        ctx.needs_parameters(missing)
        return
    publication = ctx.edit_cfg(
        publication,
        [{"path": ["sweep", "freq"], "value": frequency}] + _scalar_edits(arguments),
    )
    ctx.run_once(publication, _actual_fields(publication, arguments))
