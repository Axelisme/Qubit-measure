"""Workflow metadata and explicit per-catalog name registration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, is_dataclass
from inspect import isfunction
from typing import Protocol

from pydantic import BaseModel, JsonValue, TypeAdapter

from .env import InitEnv, WorkflowEnv
from .models import Capability, Step

_METADATA_ATTRIBUTE = "__workflow_definition__"


class WorkflowStep[P, T, S, R, C](Protocol):
    """A generator callable taking env, fixed plan, tunables, and detached state.

    Yield only via env effects; return Next(record, state), Done, or Aborted.
    Each call receives fresh copies. Exceptions outside Schedule fail the run.
    """

    def __call__(
        self, env: WorkflowEnv[C], plan: P, tun: T, state: S
    ) -> Step[S, R]: ...


@dataclass(frozen=True)
class WorkflowDefinition[P: BaseModel, T: BaseModel, S, R, C]:
    """Declaration-owner metadata retaining the callable's generic pairing.

    name is the nonempty catalog name. plan/tunables are validated model types.
    state is a dataclass type; record is the declared summary type, not an
    encoder. init builds state from InitEnv and plan without effects. requires
    lists optional soc/devices/context capabilities. step is the original
    decorated function. The owner retrieves metadata; callers never cast it.
    """

    name: str
    plan: type[P]
    tunables: type[T]
    state: type[S]
    record: type[R]
    init: Callable[[InitEnv[C], P], S]
    requires: tuple[Capability, ...]
    step: WorkflowStep[P, T, S, R, C]


# Arguments after name are keyword-only and preserve generic type pairings (D12).
def workflow[P: BaseModel, T: BaseModel, S, R, C](  # noqa: PLR0913
    name: str,
    *,
    plan: type[P],
    tunables: type[T],
    state: type[S],
    record: type[R],
    init: Callable[[InitEnv[C], P], S],
    requires: tuple[Capability, ...] = (),
) -> Callable[[WorkflowStep[P, T, S, R, C]], WorkflowStep[P, T, S, R, C]]:
    """Declare a workflow without wrapping its function or running init.

    name must be nonempty. plan must be a frozen extra-forbid Pydantic model.
    All tunables model layers must forbid extras and contain only JSON scalar
    leaves, nested models, lists, or tuples. state must be a dataclass type.
    record is retained without serialization checks. requires accepts only
    soc/devices/context, without duplicates. Invalid declarations raise ValueError;
    a non-function step raises TypeError. No global catalog is populated.
    """
    _validate_declaration(name, plan, tunables, state, requires)

    def decorate(
        step: WorkflowStep[P, T, S, R, C],
    ) -> WorkflowStep[P, T, S, R, C]:
        if not isfunction(step):
            raise TypeError("Workflow step must be a Python function")
        if getattr(step, _METADATA_ATTRIBUTE, None) is not None:
            raise ValueError("Workflow step is already declared")
        definition = WorkflowDefinition(
            name, plan, tunables, state, record, init, requires, step
        )
        setattr(step, _METADATA_ATTRIBUTE, definition)
        return step

    return decorate


def workflow_definition[P: BaseModel, T: BaseModel, S, R, C](
    step: WorkflowStep[P, T, S, R, C],
) -> WorkflowDefinition[P, T, S, R, C]:
    """Retrieve declaration metadata tied to this exact original callable.

    Undeclared functions, copied metadata, and invalid attributes raise ValueError.
    Reflection and heterogeneous metadata stay in this owner; the caller's
    callable signature supplies the generic pairing, without a cast.
    """
    definition = getattr(step, _METADATA_ATTRIBUTE, None)
    if not isinstance(definition, WorkflowDefinition) or definition.step is not step:
        raise ValueError("Workflow function is not declared")
    return definition


class WorkflowRegistry:
    """One explicitly populated catalog; names are unique and insertion-ordered.

    Create a fresh registry for each catalog reload. This registry does not own
    execution, expose an untyped callable lookup, or populate global state.
    """

    def __init__(self) -> None:
        self._names: dict[str, None] = {}

    def add[P: BaseModel, T: BaseModel, S, R, C](
        self, step: WorkflowStep[P, T, S, R, C]
    ) -> None:
        """Register a declared function; reject undeclared or duplicate names."""
        definition = workflow_definition(step)
        if definition.name in self._names:
            raise ValueError(f"Duplicate workflow name: {definition.name!r}")
        self._names[definition.name] = None

    def names(self) -> tuple[str, ...]:
        """Return registered names in insertion order as a detached tuple."""
        return tuple(self._names)


def _validate_declaration(
    name: str,
    plan: type[object],
    tunables: type[object],
    state: object,
    requires: tuple[Capability, ...],
) -> None:
    if not name.strip():
        raise ValueError("Workflow name must not be empty")
    if not issubclass(plan, BaseModel) or (
        plan.model_config.get("frozen") is not True
        or plan.model_config.get("extra") != "forbid"
    ):
        raise ValueError("Workflow plan must be a frozen extra-forbid model")
    if (
        not issubclass(tunables, BaseModel)
        or tunables.model_config.get("extra") != "forbid"
    ):
        raise ValueError("Workflow tunables must be an extra-forbid model")
    if not isinstance(state, type) or not is_dataclass(state):
        raise ValueError("Workflow state must be a dataclass type")
    if len(set(requires)) != len(requires) or any(
        capability not in ("soc", "devices", "context") for capability in requires
    ):
        raise ValueError("Workflow requires must contain unique known capabilities")
    schema = TypeAdapter(JsonValue).validate_python(tunables.model_json_schema())
    if not isinstance(schema, dict):
        raise ValueError("Tunables schema must be an object")
    definitions = schema.get("$defs", {})
    if not isinstance(definitions, dict):
        raise ValueError("Invalid tunables schema definitions")
    _check_schema(schema, definitions, set())


def _check_schema(
    schema: JsonValue, definitions: dict[str, JsonValue], visited: set[str]
) -> None:
    if not isinstance(schema, dict):
        raise ValueError("Tunables require a JSON-compatible schema")
    if "$ref" in schema:
        _check_reference(schema["$ref"], definitions, visited)
    elif "anyOf" in schema:
        alternatives = schema["anyOf"]
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError("Invalid tunables union schema")
        for alternative in alternatives:
            _check_schema(alternative, definitions, visited)
    elif schema.get("type") == "object":
        _check_model_schema(schema, definitions, visited)
    elif schema.get("type") == "array":
        _check_array_schema(schema, definitions, visited)
    elif "enum" in schema and "type" not in schema:
        values = schema["enum"]
        if not isinstance(values, list) or any(
            isinstance(value, (dict, list)) for value in values
        ):
            raise ValueError("Tunables Literal choices must be JSON scalars")
    elif (
        schema.get("type") not in ("string", "number", "integer", "boolean", "null")
        or "format" in schema
    ):
        raise ValueError(
            "Tunables leaves must be JSON scalars, not handles or expressions"
        )


def _check_reference(
    reference: JsonValue, definitions: dict[str, JsonValue], visited: set[str]
) -> None:
    if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
        raise ValueError("Invalid tunables schema reference")
    name = reference.removeprefix("#/$defs/")
    if name not in definitions:
        raise ValueError("Missing tunables schema definition")
    if name not in visited:
        visited.add(name)
        _check_schema(definitions[name], definitions, visited)


def _check_model_schema(
    schema: dict[str, JsonValue], definitions: dict[str, JsonValue], visited: set[str]
) -> None:
    properties = schema.get("properties")
    if schema.get("additionalProperties") is not False or not isinstance(
        properties, dict
    ):
        raise ValueError(
            "Tunables model layers must forbid extras; mappings are not tunables"
        )
    for child in properties.values():
        _check_schema(child, definitions, visited)


def _check_array_schema(
    schema: dict[str, JsonValue], definitions: dict[str, JsonValue], visited: set[str]
) -> None:
    if schema.get("uniqueItems") is True:
        raise ValueError("Tunables arrays must be lists or tuples, not sets")
    if "prefixItems" in schema:
        items = schema["prefixItems"]
        if not isinstance(items, list):
            raise ValueError("Invalid tunables tuple schema")
        for item in items:
            _check_schema(item, definitions, visited)
    elif "items" in schema:
        _check_schema(schema["items"], definitions, visited)
    elif schema.get("maxItems") == 0:
        return
    else:
        raise ValueError("Tunables arrays require a scalar or model element type")
