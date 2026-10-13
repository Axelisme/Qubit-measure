"""Detached author field views of the GUI's schema/input/values/errors wire.

This module projects the public Form schema. It does not generate defaults,
resolve library entries, evaluate input, or reconstruct a CfgSchema tree.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import TypeAdapter

from zcu_tools.format_version import YamlValue
from zcu_tools.mcp.measure.session import GuiRpcError


@dataclass(frozen=True)
class RecipeFormError:
    """One Form validation failure from the last successful observation.

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
    locked reports an inline-only Slot or a single-literal constrained field;
    it does not imply that descendants of a key are readonly cfg nodes.
    fields lists the active model's immediate field names, in schema order. Leaf
    fields have an empty tuple. Paths use these names, including module/name/kind.
    errors contains directed errors at or below this field, and root errors.
    All containers are detached from the retained observation.
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


def schema_branches(schema: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    """Flatten public inline anyOf/oneOf alternatives, retaining their metadata."""
    alternatives = schema.get("oneOf", schema.get("anyOf"))
    if alternatives is None:
        return (dict(schema),)
    if not isinstance(alternatives, list):
        raise GuiRpcError(
            "Invalid Form schema alternatives", reason="incompatible_wire"
        )
    return tuple(
        branch for item in alternatives for branch in schema_branches(wire_object(item))
    )


def active_schema(schema: Mapping[str, object], raw: object) -> dict[str, object]:
    """Select the declared model branch using intrinsic public discriminator tags."""
    branches = schema_branches(schema)
    models = [
        branch
        for branch in branches
        if "properties" in branch and set(wire_object(branch["properties"])) != {"expr"}
    ]
    if isinstance(raw, dict):
        for branch in models:
            properties = wire_object(branch["properties"])
            for name, item in properties.items():
                tags = [
                    variant["const"]
                    for variant in schema_branches(wire_object(item))
                    if "const" in variant
                ]
                if name in raw and raw[name] in tags:
                    return branch
    if len(models) == 1:
        return models[0]
    return dict(schema)


def field_schema(
    publication: Mapping[str, object], path: tuple[str, ...]
) -> tuple[dict[str, object], object, bool]:
    """Locate a public model-field path; inactive branches have no child paths."""
    schema = wire_object(publication.get("schema"))
    raw: object = wire_object(publication.get("input"))
    present = True
    for part in path:
        properties = wire_object(active_schema(schema, raw).get("properties", {}))
        if part not in properties:
            raise ValueError(f"Unknown cfg field: {'.'.join(path)}")
        schema = wire_object(properties[part])
        present = isinstance(raw, dict) and part in raw
        raw = raw.get(part) if isinstance(raw, dict) else None
    return schema, raw, present


def slot_schema(schema: Mapping[str, object]) -> Literal["selectable", "locked"] | None:
    """Describe module selection from public ModuleRef/ModuleInline field schemas."""
    properties = [
        wire_object(branch["properties"])
        for branch in schema_branches(schema)
        if "properties" in branch
    ]
    if any("kind" in fields and "name" in fields for fields in properties):
        return "selectable"
    if any("kind" in fields and "module" in fields for fields in properties):
        return "locked"
    return None


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


def field_value(publication: Mapping[str, object], path: tuple[str, ...]) -> YamlValue:
    """Project evaluated leaves into one ordinary subtree without lowering cfg."""
    leaves = publication.get("values")
    if not isinstance(leaves, list):
        raise GuiRpcError("Invalid Form values", reason="incompatible_wire")
    subtree: dict[str, YamlValue] = {}
    for raw in leaves:
        leaf = wire_object(raw)
        segments = leaf.get("path")
        if not isinstance(segments, list) or any(
            not isinstance(part, str) for part in segments
        ):
            raise GuiRpcError("Invalid Form value path", reason="incompatible_wire")
        if tuple(segments) == path:
            return wire_value(leaf.get("value"))
        if tuple(segments[: len(path)]) != path:
            continue
        parent = subtree
        for name in segments[len(path) : -1]:
            child = parent.setdefault(name, {})
            if not isinstance(child, dict):
                raise GuiRpcError(
                    "Conflicting Form value paths", reason="incompatible_wire"
                )
            parent = child
        parent[segments[-1]] = wire_value(leaf.get("value"))
    return subtree or None


def read_form_field(
    publication: Mapping[str, object], path: tuple[str, ...]
) -> RecipeFormField:
    """Return one author-facing field using only the last GUI observation."""
    schema, raw, present = field_schema(publication, path)
    slot = slot_schema(schema)
    mode: Literal["key", "inline"] | None = None
    key = None
    if slot and isinstance(raw, dict):
        if raw.get("kind") == "ref":
            mode = "key"
            if isinstance(raw.get("name"), str):
                key = raw["name"]
        elif raw.get("kind") == "inline":
            mode = "inline"
    elif schema.get("x-form-key-or-inline"):
        if isinstance(raw, str):
            mode, key = "key", raw
        elif isinstance(raw, dict):
            mode = "inline"
    properties = wire_object(active_schema(schema, raw).get("properties", {}))
    literal = any("const" in branch for branch in schema_branches(schema))
    return RecipeFormField(
        input=wire_value(raw),
        input_present=present,
        value=field_value(publication, path),
        mode=mode,
        key=key,
        locked=slot == "locked" or literal,
        fields=tuple(properties),
        errors=directed_errors(publication, path),
    )
