"""Validate handwritten recipe inputs and detach typed keyword arguments.

JSON Schema validation belongs to jsonschema. This adapter adds the recipe's
finite scalar boundary and existing number-to-float normalization; it does not
derive schemas from Python signatures or interpret experiment-specific rules.
"""

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite

from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError, ValidationError

from zcu_tools.mcp.measure.recipe import (
    RecipeInputSchema,
    RecipeParameter,
    RecipePropertySchema,
    RecipeScalar,
)

_SCALAR_TYPES = frozenset({"string", "number", "integer", "boolean", "null"})


def _types(schema: RecipePropertySchema) -> tuple[str, ...]:
    declared = schema["type"]
    return (declared,) if isinstance(declared, str) else tuple(declared)


def _validate_property(schema: RecipePropertySchema, name: str) -> None:
    types = _types(schema)
    if not types or not set(types) <= _SCALAR_TYPES | {"array"}:
        raise ValueError(f"Invalid recipe schema for {name}: unsupported types")
    items = schema.get("items")
    if items is not None and (
        not _types(items) or not set(_types(items)) <= _SCALAR_TYPES
    ):
        raise ValueError(f"Invalid recipe schema for {name}: items must be scalars")


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
    value contract; definition-specific normalization belongs to RecipeInputs.
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


class RecipeInputs:
    """One definition's detached handwritten schema and keyword-input codec.

    schema uses Draft 7 object validation with scalar or one-level scalar-array
    properties. Invalid JSON Schema, unsupported property types and undeclared
    required names raise ValueError at construction. The schema is copied, so
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
        for name, property_schema in self._schema["properties"].items():
            if not name:
                raise ValueError("Invalid recipe schema: empty parameter name")
            _validate_property(property_schema, name)
        if (
            not set(self._schema.get("required", ()))
            <= self._schema["properties"].keys()
        ):
            raise ValueError("Invalid recipe schema: required names must be declared")
        self._validator = Draft7Validator(self._schema)

    def normalize(self, arguments: Mapping[str, object]) -> dict[str, RecipeParameter]:
        """Return detached typed keywords, or raise ValueError without side effects.

        arguments contains the caller's explicit tool keywords. Omitted optional
        keys stay omitted; explicit null stays None. Unknown keys follow the
        authored additionalProperties flag, but every value must still be a
        RecipeParameter. All numbers must be finite and integer fields require
        Python ints, excluding bool. Errors name the rejected parameter or rule.
        Neither arguments nor any nested array is mutated or retained.
        """
        copied = copy_recipe_parameters(arguments)
        try:
            self._validator.validate(copied)
        except ValidationError as error:
            path = ".".join(str(part) for part in error.absolute_path)
            raise ValueError(
                f"Invalid recipe argument {path or '<arguments>'}: {error.message}"
            ) from error
        normalized: dict[str, RecipeParameter] = {}
        for name, value in copied.items():
            schema = self._schema["properties"].get(name)
            if isinstance(value, list):
                items = schema.get("items") if schema is not None else None
                normalized[name] = [
                    _scalar(element, f"{name}[{index}]", items)
                    for index, element in enumerate(value)
                ]
            else:
                normalized[name] = _scalar(value, name, schema)
        return normalized
