"""Single-Run lookback recipe with finite inputs; GUI owns cfg defaults."""

from math import isfinite
from typing import Any

from zcu_tools.mcp.measure.recipe_context import MissingParameter, RecipeContext
from zcu_tools.mcp.measure.session import GuiRpcError

_FREQUENCIES = (
    ("modules", "readout", "pulse_cfg", "freq"),
    ("modules", "readout", "ro_cfg", "ro_freq"),
)
_OPTIONAL_FIELDS = {
    "readout_length_us": ("modules", "readout", "ro_cfg", "ro_length"),
    "trigger_offset_us": ("modules", "readout", "ro_cfg", "trig_offset"),
    "rounds": ("rounds",),
}


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (float, int))
        and isfinite(value)
    )


def _node(publication: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
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


def _validate(arguments: dict[str, Any]) -> None:
    for name in ("reuse_tab_id", "readout_ref", "use_reset", "init_pulse_ref"):
        value = arguments.get(name)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{name} must be a non-empty string or null")
    for name in ("frequency_mhz", "readout_length_us", "trigger_offset_us"):
        value = arguments.get(name)
        if value is not None and not _finite(value):
            raise ValueError(f"{name} must be a finite real number or null")
    rounds = arguments.get("rounds")
    if rounds is not None and (isinstance(rounds, bool) or not isinstance(rounds, int)):
        raise ValueError("rounds must be an integer or null")


def _select_modules(
    ctx: RecipeContext, publication: dict[str, Any], arguments: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str | None]]:
    references = {
        "reset": arguments.get("use_reset"),
        "init_pulse": arguments.get("init_pulse_ref"),
    }
    if arguments.get("readout_ref") is not None:
        references["readout"] = arguments["readout_ref"]
    publication = ctx.edit_cfg(
        publication,
        [
            {
                "path": ["modules", name],
                "value": {"__ref": key},
            }
            for name, key in references.items()
        ],
    )
    for name, key in references.items():
        if key is not None:
            reference = _node(publication, ("modules", name))
            if reference.get("error") or reference.get("ref") != key:
                raise GuiRpcError(
                    f"Invalid {name} reference: {key}", reason="invalid_cfg"
                )

    return publication, references


def _frequency_edits(
    publication: dict[str, Any], arguments: dict[str, Any], sources: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[tuple[str, ...], str], bool]:
    edits: list[dict[str, Any]] = []
    origins: dict[tuple[str, ...], str] = {}
    missing_frequency = False
    for path in _FREQUENCIES:
        explicit = arguments.get("frequency_mhz")
        if explicit is not None:
            value = float(explicit)
            source = "frequency_mhz"
        elif arguments.get("readout_ref") is not None and _usable_frequency(
            _node(publication, path)
        ):
            origins[path] = f"library:{arguments['readout_ref']}"
            continue
        elif _finite(sources["md"].get("r_f")):
            value = {"__expr": "r_f"}
            source = "r_f"
        else:
            missing_frequency = True
            continue
        edits.append({"path": list(path), "value": value})
        origins[path] = source
    return edits, origins, missing_frequency


def lookback(ctx: RecipeContext, arguments: dict[str, Any]) -> None:
    """Prepare one lookback tab, save raw data, then analyze without accepting."""
    _validate(arguments)
    sources = ctx.rpc("context.snapshot", {})
    publication = ctx.prepare_tab("lookback", arguments.get("reuse_tab_id"))
    publication, references = _select_modules(ctx, publication, arguments)
    edits, origins, missing_frequency = _frequency_edits(
        publication, arguments, sources
    )
    if missing_frequency:
        ctx.needs_parameters(
            [
                MissingParameter(
                    "frequency_mhz",
                    "No valid library or calibrated readout frequency is available",
                )
            ]
        )
        return
    for parameter, path in _OPTIONAL_FIELDS.items():
        if arguments.get(parameter) is not None:
            value = arguments[parameter]
            if parameter != "rounds":
                value = float(value)
            edits.append({"path": list(path), "value": value})
            origins[path] = parameter
        else:
            origins[path] = "gui_default"
    if edits:
        publication = ctx.edit_cfg(publication, edits)
    if publication["status"] != "Valid":
        raise GuiRpcError("Lookback cfg is not Valid", reason="invalid_cfg")
    fields = {
        ".".join(path): {
            "value": _node(publication, path)["input"]["resolved"],
            "input": _node(publication, path)["input"],
            "source": source,
        }
        for path, source in origins.items()
    }
    for name in ("readout", "reset", "init_pulse"):
        fields[f"modules.{name}"] = {
            "value": _node(publication, ("modules", name)).get("ref"),
            "source": "explicit"
            if references.get(name) is not None
            else "gui_default"
            if name == "readout"
            else "disabled",
        }
    ctx.run_once(publication, fields)
