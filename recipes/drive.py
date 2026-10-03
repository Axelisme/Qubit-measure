"""Single-run drive recipes using GUI-owned configuration sources."""

import re
from typing import Any

from zcu_tools.mcp.measure.recipe_context import MissingParameter, RecipeContext
from zcu_tools.mcp.measure.session import GuiRpcError

from .cfg_sources import cfg_node as _node
from .cfg_sources import finite_number as _finite
from .cfg_sources import readout_frequency as _readout_frequency
from .cfg_sources import usable_frequency as _usable_frequency


def _validate(arguments: dict[str, Any]) -> None:
    for name in ("reuse_tab_id", "readout_ref", "drive_ref", "use_reset"):
        value = arguments.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a non-empty string or null")
    for name in (
        "center_mhz",
        "span_mhz",
        "gain",
        "pulse_length_us",
        "frequency_mhz",
        "max_length_us",
    ):
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


def _select_modules(
    ctx: RecipeContext, publication: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    references = {"reset": arguments.get("use_reset")}
    for parameter, slot in (("readout_ref", "readout"), ("drive_ref", "qub_pulse")):
        if arguments.get(parameter) is not None:
            references[slot] = arguments[parameter]
    publication = ctx.edit_cfg(
        publication,
        [
            {
                "path": ["modules", slot],
                "value": {"__ref": key},
            }
            for slot, key in references.items()
        ],
    )
    for slot, key in references.items():
        if key is not None:
            node = _node(publication, "modules", slot)
            if node.get("error") or node.get("ref") != key:
                raise GuiRpcError(f"Invalid {slot} reference", reason="invalid_cfg")
    return publication


def _frequency_sweep(
    publication: dict[str, Any], arguments: dict[str, Any], md: dict[str, Any]
) -> tuple[dict[str, Any], list[MissingParameter]]:
    missing = []
    center = arguments.get("center_mhz")
    if center is None:
        if not _finite(md.get("q_f")):
            missing.append(MissingParameter("center_mhz", "No finite q_f calibration"))
        center = "q_f"
    span = arguments.get("span_mhz")
    inputs = _node(publication, "sweep", "freq")["inputs"]
    if span is None:
        width = md.get("qf_w")
        edges = [inputs[key] for key in ("start", "stop")]
        if (
            not _finite(width)
            or width <= 0
            or any(
                edge["mode"] != "expression" or not re.search(r"\bqf_w\b", edge["raw"])
                for edge in edges
            )
        ):
            missing.append(
                MissingParameter("span_mhz", "No GUI linewidth-derived range")
            )
            return {}, missing
        start, stop = [re.sub(r"\bq_f\b", "0", edge["raw"]) for edge in edges]
        span = f"({stop}) - ({start})"
    sweep: dict[str, Any] = {
        "start": {"__expr": f"({center}) - ({span}) / 2"},
        "stop": {"__expr": f"({center}) + ({span}) / 2"},
    }
    if arguments.get("points") is not None:
        sweep["expts"] = arguments["points"]
    return sweep, missing


def _drive_frequency(
    publication: dict[str, Any], arguments: dict[str, Any], sources: dict[str, Any]
) -> tuple[list[dict[str, Any]], str, list[MissingParameter]]:
    path = ["modules", "qub_pulse", "freq"]
    if arguments.get("frequency_mhz") is not None:
        return (
            [{"path": path, "value": arguments["frequency_mhz"]}],
            "frequency_mhz",
            [],
        )
    drive = _node(publication, "modules", "qub_pulse")
    if drive.get("ref") in sources["ml"]["modules"] and _usable_frequency(
        _node(publication, *path)
    ):
        return [], f"library:{drive['ref']}", []
    if _finite(sources["md"].get("q_f")):
        return [{"path": path, "value": {"__expr": "q_f"}}], "q_f", []
    return (
        [],
        "missing",
        [
            MissingParameter(
                "frequency_mhz",
                "Provide frequency_mhz, a valid drive_ref frequency, or calibrated q_f",
            )
        ],
    )


def amplitude_rabi(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Run one gain sweep without requiring a prior pi calibration."""
    _validate(arguments)
    gain_range = arguments.get("gain_range")
    if gain_range is not None and (
        not isinstance(gain_range, list)
        or len(gain_range) != 2
        or not all(_finite(value) for value in gain_range)
    ):
        raise ValueError("gain_range must contain exactly two finite real endpoints")
    sources = ctx.rpc("context.snapshot", {})
    publication = ctx.prepare_tab(
        "twotone/rabi/amp_rabi", arguments.get("reuse_tab_id")
    )
    publication = _select_modules(ctx, publication, arguments)
    edits, origins, missing = _readout_frequency(
        publication, sources["md"], sources["ml"]["modules"]
    )
    drive_edits, drive_origin, drive_missing = _drive_frequency(
        publication, arguments, sources
    )
    edits.extend(drive_edits)
    origins[("modules", "qub_pulse", "freq")] = drive_origin
    if missing or drive_missing:
        ctx.needs_parameters([*drive_missing, *missing])
        return
    sweep = {}
    if gain_range is not None:
        sweep.update(start=gain_range[0], stop=gain_range[1])
    if arguments.get("points") is not None:
        sweep["expts"] = arguments["points"]
    if sweep:
        edits.append({"path": ["sweep", "gain"], "value": sweep})
    for parameter, path in (
        ("pulse_length_us", ("modules", "qub_pulse", "waveform", "length")),
        ("reps", ("reps",)),
        ("rounds", ("rounds",)),
    ):
        if arguments.get(parameter) is not None:
            edits.append({"path": list(path), "value": arguments[parameter]})
        origins[path] = (
            parameter if arguments.get(parameter) is not None else "gui_default"
        )
    publication = ctx.edit_cfg(publication, edits)
    if publication["status"] != "Valid":
        raise GuiRpcError("Amplitude Rabi cfg is not Valid", reason="invalid_cfg")
    fields = {
        ".".join(path): {
            "value": _node(publication, *path)["input"]["resolved"],
            "input": _node(publication, *path)["input"],
            "source": origin,
        }
        for path, origin in origins.items()
    }
    inputs = _node(publication, "sweep", "gain")["inputs"]
    fields["sweep.gain"] = {
        "value": {key: inputs[key]["resolved"] for key in ("start", "stop", "expts")},
        "input": inputs,
        "source": {
            "start": "gain_range" if gain_range is not None else "gui_default",
            "stop": "gain_range" if gain_range is not None else "gui_default",
            "expts": "points" if arguments.get("points") is not None else "gui_default",
        },
    }
    for slot, parameter in (
        ("reset", "use_reset"),
        ("readout", "readout_ref"),
        ("qub_pulse", "drive_ref"),
    ):
        fields[f"modules.{slot}"] = {
            "value": _node(publication, "modules", slot).get("ref"),
            "source": "explicit"
            if arguments.get(parameter) is not None
            else "disabled"
            if slot == "reset"
            else "gui_default",
        }
    ctx.run_once(publication, fields)


def time_rabi(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Run one length sweep without requiring a prior pi calibration."""
    _validate(arguments)
    sources = ctx.rpc("context.snapshot", {})
    publication = ctx.prepare_tab(
        "twotone/rabi/len_rabi", arguments.get("reuse_tab_id")
    )
    publication = _select_modules(ctx, publication, arguments)
    edits, origins, missing = _readout_frequency(
        publication, sources["md"], sources["ml"]["modules"]
    )
    drive_edits, drive_origin, drive_missing = _drive_frequency(
        publication, arguments, sources
    )
    edits.extend(drive_edits)
    origins[("modules", "qub_pulse", "freq")] = drive_origin
    if missing or drive_missing:
        ctx.needs_parameters([*drive_missing, *missing])
        return
    sweep = {}
    if arguments.get("max_length_us") is not None:
        sweep["stop"] = arguments["max_length_us"]
    if arguments.get("points") is not None:
        sweep["expts"] = arguments["points"]
    if sweep:
        edits.append({"path": ["sweep", "length"], "value": sweep})
    for parameter, path in (
        ("gain", ("modules", "qub_pulse", "gain")),
        ("reps", ("reps",)),
        ("rounds", ("rounds",)),
    ):
        if arguments.get(parameter) is not None:
            edits.append({"path": list(path), "value": arguments[parameter]})
        origins[path] = (
            parameter if arguments.get(parameter) is not None else "gui_default"
        )
    publication = ctx.edit_cfg(publication, edits)
    if publication["status"] != "Valid":
        raise GuiRpcError("Time Rabi cfg is not Valid", reason="invalid_cfg")
    fields = {
        ".".join(path): {
            "value": _node(publication, *path)["input"]["resolved"],
            "input": _node(publication, *path)["input"],
            "source": origin,
        }
        for path, origin in origins.items()
    }
    inputs = _node(publication, "sweep", "length")["inputs"]
    fields["sweep.length"] = {
        "value": {key: inputs[key]["resolved"] for key in ("start", "stop", "expts")},
        "input": inputs,
        "source": {
            "start": "gui_default",
            "stop": "max_length_us"
            if arguments.get("max_length_us") is not None
            else "gui_default",
            "expts": "points" if arguments.get("points") is not None else "gui_default",
        },
    }
    for slot, parameter in (
        ("reset", "use_reset"),
        ("readout", "readout_ref"),
        ("qub_pulse", "drive_ref"),
    ):
        fields[f"modules.{slot}"] = {
            "value": _node(publication, "modules", slot).get("ref"),
            "source": "explicit"
            if arguments.get(parameter) is not None
            else "disabled"
            if slot == "reset"
            else "gui_default",
        }
    ctx.run_once(publication, fields)


def twotone_spectrum(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Run one qubit spectrum, save raw, and complete Primary without accepting."""
    _validate(arguments)
    sources = ctx.rpc("context.snapshot", {})
    publication = ctx.prepare_tab("twotone/freq", arguments.get("reuse_tab_id"))
    publication = _select_modules(ctx, publication, arguments)
    sweep, missing = _frequency_sweep(publication, arguments, sources["md"])
    edits, origins, readout_missing = _readout_frequency(
        publication, sources["md"], sources["ml"]["modules"]
    )
    if missing or readout_missing:
        ctx.needs_parameters([*missing, *readout_missing])
        return
    edits.append({"path": ["sweep", "freq"], "value": sweep})
    scalar_paths = {
        "gain": ("modules", "qub_pulse", "gain"),
        "pulse_length_us": ("modules", "qub_pulse", "waveform", "length"),
        "reps": ("reps",),
        "rounds": ("rounds",),
    }
    for parameter, path in scalar_paths.items():
        if arguments.get(parameter) is not None:
            edits.append({"path": list(path), "value": arguments[parameter]})
        origins[path] = (
            parameter if arguments.get(parameter) is not None else "gui_default"
        )
    publication = ctx.edit_cfg(publication, edits)
    if publication["status"] != "Valid":
        raise GuiRpcError("Two-tone cfg is not Valid", reason="invalid_cfg")
    inputs = _node(publication, "sweep", "freq")["inputs"]
    values = {key: inputs[key]["resolved"] for key in ("start", "stop", "expts")}
    if not all(_finite(values[key]) for key in ("start", "stop")):
        raise GuiRpcError("Invalid resolved frequency endpoints", reason="invalid_cfg")
    span = values["stop"] - values["start"]
    if not _finite(span) or span <= 0:
        raise GuiRpcError("Invalid resolved frequency span", reason="invalid_cfg")
    fields = {
        ".".join(path): {
            "value": _node(publication, *path)["input"]["resolved"],
            "input": _node(publication, *path)["input"],
            "source": origin,
        }
        for path, origin in origins.items()
    }
    fields["sweep.freq"] = {"value": values, "input": inputs}
    fields["center_mhz"] = {
        "value": values["start"] + span / 2,
        "source": "explicit" if arguments.get("center_mhz") is not None else "q_f",
    }
    fields["span_mhz"] = {
        "value": span,
        "source": "explicit"
        if arguments.get("span_mhz") is not None
        else "gui_linewidth",
    }
    for slot, parameter in (
        ("reset", "use_reset"),
        ("readout", "readout_ref"),
        ("qub_pulse", "drive_ref"),
    ):
        fields[f"modules.{slot}"] = {
            "value": _node(publication, "modules", slot).get("ref"),
            "source": "explicit"
            if arguments.get(parameter) is not None
            else "disabled"
            if slot == "reset"
            else "gui_default",
        }
    ctx.run_once(publication, fields)
