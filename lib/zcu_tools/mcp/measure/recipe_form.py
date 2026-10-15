"""Detached author field views of the GUI's single-layer description wire.

This module projects the public app wire. It does not generate defaults,
resolve library entries, evaluate input, or reconstruct a CfgSchema tree.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import TypeAdapter

from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure.remote.form_description import FormFieldObservation
from zcu_tools.mcp.measure.session import GuiRpcError


@dataclass(frozen=True)
class RecipeFormError:
    """One Form validation failure from the current atomic observation.

    path is the absolute model path, with integer indexes for container items;
    an empty tuple addresses the whole Form. message is the owner's diagnostic.
    No error is synthesized from absent preview values.
    """

    path: tuple[str | int, ...]
    message: str


@dataclass(frozen=True)
class RecipeFormField:
    """One detached Form field, including unfinished input and owner errors.

    input is authored JSON, retaining {expr: text} and complex tags. Missing and
    explicit null both appear as None; input_present distinguishes them.
    value is the evaluated ordinary leaf or subtree, encoded as YAML/JSON with
    complex tags. None may mean null or unavailable; values never prove validity.
    mode is key for a library key, inline for an inline module/Spec, or None for
    scalar, ordinary section, range, disabled or unfilled fields.
    key is the actual library key in key mode, never an alias, otherwise None.
    locked is the owner's selection_fixed or fixed flag: this level has a
    declaration-fixed branch or complete value. It does not freeze inner
    choices, hide children or imply readonly descendants.
    fields lists the selected Option fields, or direct section properties, in
    declaration order. Option-selected tags are omitted; ref and other options without
    fields have no children. Tag leaves are not read_field paths. Inline leaves
    remain accessible as <field>.module.<leaf>.
    errors contains directed errors at or below this field, and root errors.
    All containers are detached from the current reply.
    """

    input: YamlValue
    input_present: bool
    value: YamlValue
    mode: Literal["key", "inline"] | None
    key: str | None
    locked: bool
    fields: tuple[str, ...]
    errors: tuple[RecipeFormError, ...]


@dataclass(frozen=True)
class RecipeModuleValues:
    """Detached field-local module value from the tab's fixed Reset capture.

    value contains materialized module fields in their declared units. It is not
    the full native cfg after to_cfg and excludes extras lowering. mode is key
    or inline. source is library:<actual-key> for key mode, or the dotted Form
    module field for inline. Other invalid draft fields do not block this view;
    invalid module fields raise instead of returning unavailable/empty values.
    All containers are detached from input, library and retained observations.
    """

    value: dict[str, YamlValue]
    mode: Literal["key", "inline"]
    source: str


def read_module_values(reply: Mapping[str, object]) -> RecipeModuleValues:
    """Decode the owner's value/mode/source reply; malformed wire raises.

    reply is a tab.read_module response, not a Form preview. No values are
    materialized or repaired here; incompatible replies raise incompatible_wire.
    """
    mode = reply.get("mode")
    source = reply.get("source")
    if mode not in ("key", "inline") or not isinstance(source, str) or not source:
        raise GuiRpcError("Invalid module origin", reason="incompatible_wire")
    values = wire_object(reply.get("value"))
    return RecipeModuleValues(
        {name: wire_value(value) for name, value in values.items()},
        "key" if mode == "key" else "inline",
        source,
    )


def module_scalar(value: Mapping[str, YamlValue], path: tuple[str, ...]) -> YamlValue:
    """Project one scalar module value for summary capture, not Form editing.

    path is a nonempty tuple of module-relative field names, without the Slot's
    module envelope. Unknown paths, blank segments and non-scalar targets raise
    ValueError. The returned scalar uses the module's units; no policy runs.
    """
    if not path or any(not part or part != part.strip() for part in path):
        raise ValueError("value_path must contain nonempty module field names")
    selected: YamlValue = dict(value)
    for part in path:
        if not isinstance(selected, dict) or part not in selected:
            raise ValueError(f"Unknown module value path: {'.'.join(path)}")
        selected = selected[part]
    if isinstance(selected, (dict, list)):
        raise ValueError("Module summary projection must select a scalar")
    return selected


def wire_object(value: object) -> dict[str, object]:
    """Validate an object received from the GUI without accepting bad keys."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise GuiRpcError("Expected a GUI object", reason="incompatible_wire")
    return dict(value)


def wire_value(value: object) -> YamlValue:
    """Detach ordinary JSON/YAML values without resolving Form tags."""
    try:
        return TypeAdapter(YamlValue).validate_python(value)
    except ValueError as error:
        raise GuiRpcError("Invalid GUI value", reason="incompatible_wire") from error


def field_observation(reply: Mapping[str, object]) -> FormFieldObservation:
    """Decode a tab.describe_cfg reply; malformed wire raises incompatible_wire.

    reply must follow the app's typed single-layer wire, including local input
    and value. This validates transport shape, never core annotation semantics.
    """
    try:
        return TypeAdapter(FormFieldObservation).validate_python(reply, strict=True)
    except ValueError as error:
        raise GuiRpcError(
            "Invalid Form description", reason="incompatible_wire"
        ) from error


def slot_description(
    observation: FormFieldObservation,
) -> Literal["selectable", "locked"] | None:
    """Classify S2 module metadata, without option matching or union inference.

    observation is a decoded current field. Declared ref/inline option values
    identify module selectors; the owner's selection_fixed or fixed flag
    decides locking. Missing library options alone never imply a lock.
    Nonmodule fields return None.
    """
    target = observation["description"]["target"]
    if target is not None:
        for option in target["options"] or []:
            value = option["value"]
            if isinstance(value, dict) and value.get("kind") in ("ref", "inline"):
                return (
                    "locked"
                    if target["selection_fixed"] or target["fixed"]
                    else "selectable"
                )
    return None


def library_option_value(
    observation: FormFieldObservation, name: str, field: str
) -> YamlValue:
    """Return the offered ref value for literal key name at diagnostic field.

    observation is a decoded current description. No input is synthesized or
    source resolved. Unoffered keys raise invalid_cfg with available keys.
    """
    target = observation["description"]["target"]
    keys: list[str] = []
    for option in target["options"] or [] if target is not None else []:
        value = option["value"]
        if value == {"kind": "ref", "name": name}:
            return wire_value(value)
        if isinstance(value, dict) and value.get("kind") == "ref":
            key = value.get("name")
            if isinstance(key, str):
                keys.append(key)
    raise GuiRpcError(
        f"Invalid {field} reference {name!r}. Available keys: {keys}",
        reason="invalid_cfg",
    )


def directed_errors(
    publication: Mapping[str, object], path: tuple[str, ...]
) -> tuple[RecipeFormError, ...]:
    """Decode root and descendant errors from the Form observation."""
    raw = publication.get("errors")
    if not isinstance(raw, list):
        raise GuiRpcError("Invalid Form errors", reason="incompatible_wire")
    errors: list[RecipeFormError] = []
    for item in raw:
        error = wire_object(item)
        segments = error.get("path")
        message = error.get("message")
        if (
            not isinstance(segments, list)
            or any(
                not isinstance(part, (str, int)) or isinstance(part, bool)
                for part in segments
            )
            or not isinstance(message, str)
        ):
            raise GuiRpcError("Invalid Form error", reason="incompatible_wire")
        error_path = tuple(segments)
        if not error_path or error_path[: len(path)] == path:
            errors.append(RecipeFormError(error_path, message))
    return tuple(errors)


def read_form_field(
    observation: FormFieldObservation, path: tuple[str, ...]
) -> RecipeFormField:
    """Project one decoded atomic field read into the author's detached view.

    observation comes from tab.describe_cfg, and path is its requested canonical
    model-field path. Fixed hidden structural tags are not author field paths.
    No option matching, schema traversal or source reads occur here.
    """
    description = observation["description"]
    target = description["target"]
    if target is None or (target["hidden"] and target["fixed"]):
        raise ValueError(f"Unknown cfg field: {'.'.join(path)}")
    raw = observation["input"]
    mode: Literal["key", "inline"] | None = None
    key = None
    if isinstance(raw, dict):
        if raw.get("kind") == "ref":
            mode = "key"
            name = raw.get("name")
            if isinstance(name, str):
                key = name
        elif raw.get("kind") == "inline":
            mode = "inline"
    return RecipeFormField(
        input=wire_value(raw),
        input_present=observation["input_present"],
        value=wire_value(observation["value"]),
        mode=mode,
        key=key,
        locked=target["selection_fixed"] or target["fixed"],
        fields=tuple(
            child["name"]
            for child in description["children"]
            if not (child["hidden"] and child["fixed"])
        ),
        errors=directed_errors(dict(observation), path),
    )
