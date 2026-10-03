"""Coherence recipes using calibrated library pulses."""

from typing import Any

from zcu_tools.mcp.measure.recipe_context import MissingParameter, RecipeContext
from zcu_tools.mcp.measure.session import GuiRpcError

from .cfg_sources import cfg_node as _node
from .cfg_sources import finite_number as _finite
from .cfg_sources import readout_frequency as _readout_frequency


def t2ramsey(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Run one calibrated Ramsey delay sweep."""
    _run(ctx, arguments, "t2ramsey", (("pi2_pulse", "pi2_ref"),))


def t2echo(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Run one calibrated Echo total-delay sweep."""
    _run(ctx, arguments, "t2echo", (("pi_pulse", "pi_ref"), ("pi2_pulse", "pi2_ref")))


def _validate(arguments: dict[str, Any]) -> None:
    for name in ("reuse_tab_id", "readout_ref", "pi_ref", "pi2_ref", "use_reset"):
        value = arguments.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a non-empty string or null")
    delay = arguments.get("max_delay_us")
    if delay is not None and (not _finite(delay) or delay <= 0):
        raise ValueError("max_delay_us must be a positive finite real number or null")
    detune = arguments.get("detune_ratio")
    if detune is not None and not _finite(detune):
        raise ValueError("detune_ratio must be a finite real number or null")
    for name in ("points", "reps", "rounds"):
        value = arguments.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError(f"{name} must be an integer or null")
    if arguments.get("points") is not None and arguments["points"] < 2:
        raise ValueError("points must be at least two")


def _select_modules(
    ctx: RecipeContext,
    publication: dict[str, Any],
    arguments: dict[str, Any],
    pulse_slots: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    references = {"reset": arguments.get("use_reset")}
    for slot, parameter in (*pulse_slots, ("readout", "readout_ref")):
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
        node = _node(publication, "modules", slot)
        if key is not None and (node.get("error") or node.get("ref") != key):
            raise GuiRpcError(f"Invalid {slot} reference", reason="invalid_cfg")
    return publication


def t1(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Run one calibrated T1 delay sweep, save raw data and Primary analysis."""
    _run(ctx, arguments, "t1", (("pi_pulse", "pi_ref"),))


def _run(
    ctx: RecipeContext,
    arguments: dict[str, Any],
    experiment: str,
    pulse_slots: tuple[tuple[str, str], ...],
) -> None:
    _validate(arguments)
    sources = ctx.rpc("context.snapshot", {})
    publication = ctx.prepare_tab(
        f"twotone/{experiment}", arguments.get("reuse_tab_id")
    )
    publication = _select_modules(ctx, publication, arguments, pulse_slots)
    edits, origins, missing = _readout_frequency(
        publication, sources["md"], sources["ml"]["modules"]
    )
    for slot, parameter in pulse_slots:
        pulse = _node(publication, "modules", slot)
        if pulse.get("ref") not in sources["ml"]["modules"]:
            missing.append(
                MissingParameter(parameter, f"Provide a calibrated library {slot}")
            )
    if missing:
        ctx.needs_parameters(missing)
        return
    sweep = {}
    for parameter, key in (("max_delay_us", "stop"), ("points", "expts")):
        if arguments.get(parameter) is not None:
            sweep[key] = arguments[parameter]
    if sweep:
        edits.append({"path": ["sweep", "length"], "value": sweep})
    scalars = (
        ("reps", "rounds") if experiment == "t1" else ("reps", "rounds", "detune_ratio")
    )
    for parameter in scalars:
        if arguments.get(parameter) is not None:
            edits.append({"path": [parameter], "value": arguments[parameter]})
        origins[(parameter,)] = (
            parameter if arguments.get(parameter) is not None else "gui_default"
        )
    publication = ctx.edit_cfg(publication, edits)
    if publication["status"] != "Valid":
        raise GuiRpcError(f"{experiment} cfg is not Valid", reason="invalid_cfg")
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
            key: parameter
            if parameter is not None and arguments.get(parameter) is not None
            else "gui_default"
            for key, parameter in (
                ("start", None),
                ("stop", "max_delay_us"),
                ("expts", "points"),
            )
        },
    }
    for slot, parameter in (
        ("reset", "use_reset"),
        ("readout", "readout_ref"),
        *pulse_slots,
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
