"""Single-shot GE calibration recipe."""

from typing import Any

from zcu_tools.mcp.measure.recipe_context import MissingParameter, RecipeContext
from zcu_tools.mcp.measure.session import GuiRpcError

from .cfg_sources import cfg_node, readout_frequency


def singleshot_ge(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    sources = ctx.rpc("context.snapshot", {})
    library = sources["ml"]["modules"]
    publication = ctx.prepare_tab("singleshot/ge", arguments.get("reuse_tab_id"))
    references = {
        "reset": arguments.get("use_reset"),
        "init_pulse": arguments.get("init_pulse_ref"),
    }
    for slot, parameter in (("probe_pulse", "pi_ref"), ("readout", "readout_ref")):
        if arguments.get(parameter) is not None:
            references[slot] = arguments[parameter]
    for slot, key in references.items():
        if key is not None and key not in library:
            raise GuiRpcError(f"Invalid {slot} library reference", reason="invalid_cfg")
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
        node = cfg_node(publication, "modules", slot)
        if key is not None and (node.get("error") or node.get("ref") != key):
            raise GuiRpcError(f"Invalid {slot} reference", reason="invalid_cfg")
    edits, origins, missing = readout_frequency(publication, sources["md"], library)
    pulse = cfg_node(publication, "modules", "probe_pulse")
    if pulse.get("ref") not in library:
        missing.append(
            MissingParameter("pi_ref", "Provide a calibrated library probe_pulse")
        )
    if missing:
        ctx.needs_parameters(missing)
        return
    if arguments.get("shots") is not None:
        edits.append({"path": ["shots"], "value": arguments["shots"]})
    origins[("shots",)] = (
        "shots" if arguments.get("shots") is not None else "gui_default"
    )
    publication = ctx.edit_cfg(publication, edits)
    if publication["status"] != "Valid":
        raise GuiRpcError("GE cfg is not Valid", reason="invalid_cfg")
    fields = {
        ".".join(path): {
            "value": cfg_node(publication, *path)["input"]["resolved"],
            "input": cfg_node(publication, *path)["input"],
            "source": origin,
        }
        for path, origin in origins.items()
    }
    for slot, parameter in (
        ("reset", "use_reset"),
        ("init_pulse", "init_pulse_ref"),
        ("readout", "readout_ref"),
        ("probe_pulse", "pi_ref"),
    ):
        fields[f"modules.{slot}"] = {
            "value": cfg_node(publication, "modules", slot).get("ref"),
            "source": "explicit"
            if arguments.get(parameter) is not None
            else "disabled"
            if slot in ("reset", "init_pulse")
            else "gui_default",
        }
    ctx.run_once(publication, fields, analysis_mode="primary_post")
