"""Single-run drive recipes using GUI-owned configuration sources."""

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
    for name in ("reuse_tab_id", "readout_ref", "drive_ref", "use_reset"):
        value = arguments.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a non-empty string or null")
    for name in ("center_mhz", "span_mhz", "gain", "pulse_length_us"):
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


def _usable_frequency(node: dict[str, Any]) -> bool:
    value = node.get("input", {})
    return bool(
        node.get("valid")
        and not value.get("error")
        and not value.get("validation_error")
        and _finite(value.get("resolved"))
    )


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
                "value": {"__ref": key} if key is not None else None,
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


def _readout_frequency(
    publication: dict[str, Any], md: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[tuple[str, ...], str], list[MissingParameter]]:
    readout = _node(publication, "modules", "readout")
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
        if readout.get("ref") is not None and _usable_frequency(
            _node(publication, *path)
        ):
            origins[path] = f"library:{readout['ref']}"
        elif _finite(md.get("r_f")):
            edits.append({"path": list(path), "value": {"__expr": "r_f"}})
            origins[path] = "r_f"
        elif not missing:
            missing.append(
                MissingParameter(
                    "readout_ref", "Provide a valid readout_ref or calibrated r_f"
                )
            )
    return edits, origins, missing


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


def twotone_spectrum(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Run one qubit spectrum, save raw, and complete Primary without accepting."""
    _validate(arguments)
    sources = ctx.rpc("context.snapshot", {})
    publication = ctx.prepare_tab("twotone/freq", arguments.get("reuse_tab_id"))
    publication = _select_modules(ctx, publication, arguments)
    sweep, missing = _frequency_sweep(publication, arguments, sources["md"])
    edits, origins, readout_missing = _readout_frequency(publication, sources["md"])
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
