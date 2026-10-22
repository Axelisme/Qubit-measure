"""Workflow metadata and explicit per-catalog name registration."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import iscoroutinefunction, signature
from typing import Protocol, get_args, get_origin, get_type_hints

from pydantic import BaseModel, JsonValue, TypeAdapter

from .env import WorkflowEnv
from .models import Step
from .tunables import Live

_METADATA_ATTRIBUTE = "__workflow_definition__"


class WorkflowStep[P, T, S, R, C](Protocol):
    """A generator callable taking env, fixed plan, tunables, and detached state.

    Yield only via env effects; return Next(record, state), Done, or Aborted.
    Each call receives fresh copies. Exceptions outside Schedule fail the run.
    """

    def __call__(
        self, env: WorkflowEnv[C], plan: P, tun: T, state: S
    ) -> Step[S, R]: ...


class WorkflowScript[P, T, Summary: BaseModel | None, C](Protocol):
    """An async author receiving fixed plan and live tunables, returning a summary.

    Await env operations and commit BaseModel records explicitly. Unhandled
    exceptions fail the run; returning a value completes it without a new record.
    """

    async def __call__(self, env: WorkflowEnv[C], plan: P, tun: Live[T]) -> Summary: ...


@dataclass(frozen=True)
class WorkflowDefinition[P: BaseModel, T: BaseModel, Summary: BaseModel | None, C]:
    """Declaration-owner metadata retaining the callable's generic pairing.

    name is the nonempty catalog name. plan/tunables are validated model types.
    description is authored catalog help. script is the original coroutine;
    its annotations infer plan/tunables and preserve the returned Summary type.
    Record types belong to individual commit calls, not the declaration.
    """

    name: str
    plan: type[P]
    tunables: type[T]
    description: str
    script: WorkflowScript[P, T, Summary, C]


def workflow(name: str, description: str) -> WorkflowDecorator:
    """Attach signature-derived metadata without executing or wrapping a script.

    The coroutine must take exactly (env, plan, tun), annotated WorkflowEnv[C],
    frozen extra-forbid Plan, and Live[extra-forbid Tunables], and annotate its
    return. Tunables contain JSON scalar leaves, nested models, lists or tuples.
    Invalid models/signatures raise ValueError; non-coroutines raise TypeError.
    name must be nonempty. No global registry is populated.
    """
    return WorkflowDecorator(name, description)


@dataclass(frozen=True)
class WorkflowDecorator:
    """Preserve each script's generic pairing while checking runtime annotations.

    name and description become catalog metadata on the original coroutine.
    This callable is returned by workflow; it does not register or execute code.
    """

    name: str
    description: str

    def __call__[P: BaseModel, T: BaseModel, Summary: BaseModel | None, C](
        self, script: WorkflowScript[P, T, Summary, C]
    ) -> WorkflowScript[P, T, Summary, C]:
        """Validate the async author signature and attach its declaration once."""
        if not iscoroutinefunction(script):
            raise TypeError("Workflow author must be a coroutine function")
        if getattr(script, _METADATA_ATTRIBUTE, None) is not None:
            raise ValueError("Workflow script is already declared")
        parameters = tuple(signature(script).parameters)
        if parameters != ("env", "plan", "tun"):
            raise ValueError("Workflow signature must be (env, plan, tun)")
        hints = get_type_hints(script)
        if get_origin(hints.get("env")) is not WorkflowEnv:
            raise ValueError("Workflow env must be annotated WorkflowEnv[C]")
        if "plan" not in hints:
            raise ValueError("Workflow plan must be annotated with a model")
        plan = hints["plan"]
        live = hints.get("tun")
        if get_origin(live) is not Live or len(get_args(live)) != 1:
            raise ValueError("Workflow tun must be annotated Live[Tunables]")
        tunables = get_args(live)[0]
        if "return" not in hints:
            raise ValueError("Workflow must annotate its returned summary")
        _validate_declaration(self.name, plan, tunables)
        definition = WorkflowDefinition(
            self.name, plan, tunables, self.description, script
        )
        setattr(script, _METADATA_ATTRIBUTE, definition)
        return script


def workflow_definition[P: BaseModel, T: BaseModel, Summary: BaseModel | None, C](
    script: WorkflowScript[P, T, Summary, C],
) -> WorkflowDefinition[P, T, Summary, C]:
    """Retrieve declaration metadata tied to this exact original callable.

    Undeclared functions, copied metadata, and invalid attributes raise ValueError.
    Reflection and heterogeneous metadata stay in this owner; the caller's
    callable signature supplies the generic pairing, without a cast.
    """
    definition = getattr(script, _METADATA_ATTRIBUTE, None)
    if (
        not isinstance(definition, WorkflowDefinition)
        or definition.script is not script
    ):
        raise ValueError("Workflow function is not declared")
    return definition


class WorkflowRegistry:
    """One explicitly populated catalog; names are unique and insertion-ordered.

    Create a fresh registry for each catalog reload. This registry does not own
    execution, expose an untyped callable lookup, or populate global state.
    """

    def __init__(self) -> None:
        self._names: dict[str, None] = {}

    def add[P: BaseModel, T: BaseModel, Summary: BaseModel | None, C](
        self, script: WorkflowScript[P, T, Summary, C]
    ) -> None:
        """Register a declared coroutine; reject undeclared or duplicate names."""
        definition = workflow_definition(script)
        if definition.name in self._names:
            raise ValueError(f"Duplicate workflow name: {definition.name!r}")
        self._names[definition.name] = None

    def names(self) -> tuple[str, ...]:
        """Return registered names in insertion order as a detached tuple."""
        return tuple(self._names)


def _validate_declaration(
    name: str,
    plan: object,
    tunables: object,
) -> None:
    if not name.strip():
        raise ValueError("Workflow name must not be empty")
    if (
        not isinstance(plan, type)
        or not issubclass(plan, BaseModel)
        or (
            plan.model_config.get("frozen") is not True
            or plan.model_config.get("extra") != "forbid"
        )
    ):
        raise ValueError("Workflow plan must be a frozen extra-forbid model")
    if (
        not isinstance(tunables, type)
        or not issubclass(tunables, BaseModel)
        or tunables.model_config.get("extra") != "forbid"
    ):
        raise ValueError("Workflow tunables must be an extra-forbid model")
    schema = TypeAdapter(JsonValue).validate_python(tunables.model_json_schema())
    if not isinstance(schema, dict):
        raise ValueError("Tunables schema must be an object")
    definitions = schema.get("$defs", {})
    if not isinstance(definitions, dict):
        raise ValueError("Invalid tunables schema definitions")
    _check_schema(schema, definitions, set())


# Schema nodes nest models/arrays/references; visited definitions bound this recursive walk.
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
