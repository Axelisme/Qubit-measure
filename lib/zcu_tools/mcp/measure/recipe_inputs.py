"""Validate handwritten recipe inputs and detach typed keyword arguments.

JSON Schema validation belongs to jsonschema. This adapter adds the recipe's
finite leaf boundary and existing number-to-float normalization; it does not
derive schemas from Python signatures or interpret experiment-specific rules.
"""

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite

from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError, ValidationError

from zcu_tools.mcp.measure.recipe import (
    RecipeInputSchema,
    RecipeInputValue,
    RecipeParameter,
    RecipePropertySchema,
    RecipeScalar,
)

_SCALAR_TYPES = frozenset({"string", "number", "integer", "boolean", "null"})


def _types(schema: RecipeInputSchema | RecipePropertySchema) -> tuple[str, ...]:
    declared = schema["type"]
    return (declared,) if isinstance(declared, str) else tuple(declared)


def _validate_property(
    schema: RecipeInputSchema | RecipePropertySchema, name: str
) -> None:
    types = _types(schema)
    if not types or not set(types) <= _SCALAR_TYPES | {"array", "object"}:
        raise ValueError(f"Invalid recipe schema for {name}: unsupported types")
    items = schema.get("items")
    if items is not None and (
        not _types(items) or not set(_types(items)) <= _SCALAR_TYPES
    ):
        raise ValueError(f"Invalid recipe schema for {name}: items must be scalars")
    if "object" not in types:
        return
    additional = schema.get("additionalProperties")
    if "properties" not in schema and not isinstance(additional, dict):
        raise ValueError(
            f"Invalid recipe schema for {name}: object needs declared fields"
        )
    properties = schema.get("properties", {})
    for field, child in properties.items():
        if not field:
            raise ValueError(f"Invalid recipe schema for {name}: empty parameter name")
        _validate_property(child, f"{name}.{field}")
    if isinstance(additional, dict):
        _validate_property(additional, f"{name}.*")
    if not set(schema.get("required", ())) <= properties.keys():
        raise ValueError(
            f"Invalid recipe schema for {name}: required names must be declared"
        )


def _number(value: int | float, name: str) -> float:
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _scalar(
    value: object, name: str, schema: RecipePropertySchema | None
) -> RecipeScalar:
    if value is None or isinstance(value, (str, bool)):
        return value
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must contain only recipe scalar values")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    types = _types(schema) if schema is not None else ()
    if "number" in types:
        return _number(value, name)
    # Keep the existing integer-input contract: integral floats are not integers.
    if "integer" in types and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def copy_recipe_parameters(
    parameters: Mapping[str, object],
) -> dict[str, RecipeParameter]:
    """Copy named scalar/one-level array parameters without GUI or schema work.

    Values must be recipe scalars or lists of those scalars, with finite numbers.
    Names are the mapping's string keys. Raise ValueError for unsupported values;
    nested arrays/models are not accepted. Preserve scalar types, detach lists
    and do not inject defaults. Recipe inputs and analysis updates share this
    scalar value contract; nested invocation inputs belong only to RecipeInputs.
    """
    copied: dict[str, RecipeParameter] = {}
    for name, value in parameters.items():
        if isinstance(value, list):
            copied[name] = [
                _scalar(element, f"{name}[{index}]", None)
                for index, element in enumerate(value)
            ]
        else:
            copied[name] = _scalar(value, name, None)
    return copied


def _input_value(
    value: object, name: str, schema: RecipePropertySchema | None
) -> RecipeInputValue:
    """Detach a declared input, preserving the flat contract for undeclared extras."""
    if isinstance(value, dict):
        if schema is None or "object" not in _types(schema):
            raise ValueError(f"{name} must not contain undeclared objects")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties")
        extra_schema = additional if isinstance(additional, dict) else None
        copied: dict[str, RecipeInputValue] = {}
        for field, child in value.items():
            if not isinstance(field, str):
                raise ValueError(f"{name} must contain only string keys")
            copied[field] = _input_value(
                child, f"{name}.{field}", properties.get(field, extra_schema)
            )
        return copied
    if isinstance(value, list):
        items = schema.get("items") if schema is not None else None
        return [
            _scalar(element, f"{name}[{index}]", items)
            for index, element in enumerate(value)
        ]
    return _scalar(value, name, schema)


class RecipeInputs:
    """One definition's detached handwritten schema and keyword-input codec.

    schema uses Draft 7 validation with scalar, scalar-array and explicitly
    declared nested object properties. Invalid schemas, unsupported types and
    undeclared required names raise ValueError at construction. The schema is copied, so
    later author mutations do not change the admitted input contract.

    normalize validates the schema's constraints, rejects non-finite numbers,
    preserves integer inputs and normalizes number inputs/elements to float.
    No GUI work, signature inspection, defaults injection or cfg validation takes
    place here; experiment-specific cross-field rules remain with the recipe.
    """

    def __init__(self, schema: RecipeInputSchema) -> None:
        try:
            Draft7Validator.check_schema(schema)
        except SchemaError as error:
            raise ValueError(f"Invalid recipe schema: {error.message}") from error
        if schema["type"] != "object":
            raise ValueError("Invalid recipe schema: root must be an object")
        self._schema = deepcopy(schema)
        _validate_property(self._schema, "<arguments>")
        self._validator = Draft7Validator(self._schema)

    def normalize(self, arguments: Mapping[str, object]) -> dict[str, RecipeInputValue]:
        """Return detached typed keywords, or raise ValueError without side effects.

        arguments contains the caller's explicit tool keywords. Omitted optional
        keys stay omitted; explicit null stays None. Unknown keys follow the
        authored additionalProperties policy. Nested objects require a declared
        object schema; undeclared extras remain scalar/scalar-array only. All
        numbers must be finite and integer fields require Python ints, excluding
        bool. Errors name the rejected path or rule. Neither arguments nor any
        nested container is mutated or retained.
        """
        copied = dict(arguments)
        try:
            # Check authored constraints before number-to-float rounding.
            self._validator.validate(copied)
        except ValidationError as error:
            path = ".".join(str(part) for part in error.absolute_path)
            raise ValueError(
                f"Invalid recipe argument {path or '<arguments>'}: {error.message}"
            ) from error
        return {
            name: _input_value(value, name, self._schema["properties"].get(name))
            for name, value in copied.items()
        }
