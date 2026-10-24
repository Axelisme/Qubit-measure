"""Detached author field views of the GUI's single-layer description wire.

This module projects the public app wire. It does not generate defaults,
resolve library entries, evaluate input, or reconstruct a CfgSchema tree.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, TypeAdapter

from zcu_tools.cfg_model import Expr
from zcu_tools.core.complex_codec import encode_complex
from zcu_tools.core.element import ElementIdentity
from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure.remote.form_description import (
    RecipeFieldObservation,
    RecipeSelectorWire,
    RecipeSourceWire,
)
from zcu_tools.mcp.measure.session import GuiRpcError

type RecipeFieldPath = tuple[str | int, ...]
"""Logical field names and original list indexes; numeric strings stay names."""


def dotted_path(path: object) -> RecipeFieldPath:
    """Decode a nonempty dotted recipe path into logical owner segments.

    Decimal segments address original list positions. Empty/blank segments and
    negative indexes and non-string carriers raise ValueError.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("field must be a nonempty dotted string")
    parts = path.split(".")
    if any(not part or part != part.strip() or part.startswith("-") for part in parts):
        raise ValueError("field requires nonblank names or nonnegative indexes")
    return tuple(int(part) if part.isdecimal() else part for part in parts)


def field_path(path: RecipeFieldPath) -> RecipeFieldPath:
    """Validate a nonempty typed path carrier, without resolving owner fields.

    Strings are field names and nonnegative integers are original list indexes.
    Booleans, blank names and dotted-string carriers raise ValueError. Names
    containing dots are literal names; no splitting or name/index lookup occurs.
    """
    if isinstance(path, str) or not path:
        raise ValueError("field must be a nonempty typed path")
    if any(
        (isinstance(part, str) and (not part or part != part.strip()))
        or (type(part) is int and part < 0)
        or (not isinstance(part, str) and type(part) is not int)
        for part in path
    ):
        raise ValueError("field requires nonblank names or nonnegative int indexes")
    return tuple(path)


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
    locked is the owner's selection_fixed, fixed or readonly flag: this level
    has a declaration-fixed branch, complete value or readonly subtree.
    A fixed branch alone does not freeze its inner choices or hide children.
    fields lists logical direct children, including readable captured Spec
    children. Only generated wrappers are hidden; real module fields remain.
    value_status is available (including null), missing or unavailable. selector
    is owner-supplied generated identity or None for ordinary fields. source is
    the nearest captured ref identity or None. via_ref means a ref ancestor was
    crossed, independently of declaration readonly. errors contains relevant
    logical directed failures, including root and affecting ancestors.
    Reading grants no writes.
    element is the declaring owner's same-read original index/name, or None
    outside declared fixed elements. All containers are detached from the reply.
    """

    input: YamlValue
    input_present: bool
    value: YamlValue
    mode: Literal["key", "inline"] | None
    key: str | None
    locked: bool
    fields: tuple[str, ...]
    errors: tuple[RecipeFormError, ...]
    value_status: Literal["available", "missing", "unavailable"]
    selector: RecipeSelectorWire | None
    source: RecipeSourceWire | None
    via_ref: bool
    element: ElementIdentity | None = None


def encode_form_value(value: object) -> YamlValue:
    """Encode a named Form value for the GUI RPC without evaluating it.

    Format authority: lib/zcu_tools/gui/form/README.md (FormInput/expression
    and complex tags) and tab.describe_cfg RPC field descriptions. Accept
    scalar values, Expr, BaseModel and nested string-keyed mappings/sequences.
    Models retain concrete subclass fields. Unsupported values or non-string
    mapping keys raise TypeError. Schema and source validation stay with GUI.
    """
    if isinstance(value, Expr):
        return {"expr": value.text}
    if isinstance(value, BaseModel):
        return encode_form_value(value.model_dump(mode="python", serialize_as_any=True))
    if isinstance(value, complex):
        return {
            key: encode_form_value(item) for key, item in encode_complex(value).items()
        }
    if isinstance(value, Mapping):
        encoded: dict[str, YamlValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Form input mapping keys must be strings")
            encoded[key] = encode_form_value(item)
        return encoded
    if isinstance(value, (list, tuple)):
        return [encode_form_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported Form input value: {type(value).__name__}")


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


def field_observation(reply: Mapping[str, object]) -> RecipeFieldObservation:
    """Decode a tab.read_recipe_field reply; malformed wire raises incompatible_wire.

    reply must follow the app's typed single-layer wire, including local input
    and value. This validates transport shape, never core annotation semantics.
    """
    try:
        return TypeAdapter(RecipeFieldObservation).validate_python(reply, strict=True)
    except ValueError as error:
        raise GuiRpcError(
            "Invalid Form description", reason="incompatible_wire"
        ) from error


def read_form_field(
    observation: RecipeFieldObservation, path: RecipeFieldPath
) -> RecipeFormField:
    """Project one decoded atomic field read into the author's detached view.

    observation comes from tab.read_recipe_field; path is its logical request.
    Fixed hidden structural tags are not author field paths. Mode/key use only
    owner-supplied selector identity, never authored input or Option inspection.
    """
    description = observation["description"]
    target = description["target"]
    if target is None or (target["hidden"] and target["fixed"]):
        raise ValueError(f"Unknown cfg field: {path!r}")
    raw = observation["input"]
    mode: Literal["key", "inline"] | None = None
    selector = observation["selector"]
    key = None
    if selector is not None:
        if selector["mode"] == "key":
            mode = "key"
            key = selector["key"]
        elif selector["mode"] == "inline":
            mode = "inline"
    return RecipeFormField(
        input=wire_value(raw),
        input_present=observation["input_present"],
        value=wire_value(observation["value"]),
        mode=mode,
        key=key,
        locked=target["selection_fixed"] or target["fixed"] or target["readonly"],
        fields=tuple(
            child["name"]
            for child in description["children"]
            if not (child["hidden"] and child["fixed"])
        ),
        errors=tuple(
            RecipeFormError(tuple(error["path"]), error["message"])
            for error in observation["errors"]
        ),
        value_status=observation["value_status"],
        selector=selector,
        source=observation["source"],
        via_ref=observation["readonly"],
        element=observation.get("element"),
    )
