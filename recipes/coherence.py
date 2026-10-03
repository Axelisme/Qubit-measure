"""Coherence recipes using calibrated library pulses."""

from typing import Any

from zcu_tools.mcp.measure.recipe_context import MissingParameter, RecipeContext
from zcu_tools.mcp.measure.session import GuiRpcError

from .cfg_sources import cfg_node as _node
from .cfg_sources import readout_frequency as _readout_frequency


def _select_modules(
    ctx: RecipeContext, publication: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    references = {"reset": arguments.get("use_reset")}
    for slot, parameter in (("pi_pulse", "pi_ref"), ("readout", "readout_ref")):
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
    sources = ctx.rpc("context.snapshot", {})
    publication = ctx.prepare_tab("twotone/t1", arguments.get("reuse_tab_id"))
    publication = _select_modules(ctx, publication, arguments)
    edits, origins, missing = _readout_frequency(
        publication, sources["md"], sources["ml"]["modules"]
    )
    pulse = _node(publication, "modules", "pi_pulse")
    if pulse.get("ref") not in sources["ml"]["modules"]:
        missing.append(
            MissingParameter("pi_ref", "Provide a calibrated library pi pulse")
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
    for parameter in ("reps", "rounds"):
        if arguments.get(parameter) is not None:
            edits.append({"path": [parameter], "value": arguments[parameter]})
        origins[(parameter,)] = (
            parameter if arguments.get(parameter) is not None else "gui_default"
        )
    publication = ctx.edit_cfg(publication, edits)
    if publication["status"] != "Valid":
        raise GuiRpcError("T1 cfg is not Valid", reason="invalid_cfg")
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
        ("pi_pulse", "pi_ref"),
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
