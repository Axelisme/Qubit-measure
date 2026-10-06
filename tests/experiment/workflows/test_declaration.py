"""Workflow declaration and explicit catalog behavior through public APIs."""

from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict
from zcu_tools.experiment.workflows import (
    Done,
    InitEnv,
    Step,
    WorkflowEnv,
    WorkflowRegistry,
    WorkflowStep,
    workflow,
)


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    count: int = 1


class Knobs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reps: int = 2


class NestedKnobs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cal: Knobs
    spans: tuple[float, ...]
    gate: Literal[1, "auto"] | None = None


@dataclass
class State:
    index: int = 0


def _init(_env: InitEnv[None], _plan: BaseModel) -> State:
    return State()


def _step(
    env: WorkflowEnv[None], plan: BaseModel, tun: BaseModel, state: State
) -> Step[State, object]:
    yield from ()
    return Done()


def _declare[P: BaseModel, T: BaseModel](
    name: str, plan: type[P], tunables: type[T]
) -> WorkflowStep[P, T, State, object, None]:
    def step(
        env: WorkflowEnv[None], plan: P, tun: T, state: State
    ) -> Step[State, object]:
        return _step(env, plan, tun, state)

    return workflow(
        name,
        plan=plan,
        tunables=tunables,
        state=State,
        record=object,
        init=_init,
    )(step)


def test_decorator_preserves_function_and_does_not_execute_init() -> None:
    calls: list[str] = []

    def init(_env: InitEnv[None], _plan: Plan) -> State:
        calls.append("init")
        return State()

    def original(
        env: WorkflowEnv[None], plan: Plan, tun: Knobs, state: State
    ) -> Step[State, object]:
        calls.append("step")
        yield from ()
        return Done()

    declared = workflow(
        "original",
        plan=Plan,
        tunables=Knobs,
        state=State,
        record=object,
        init=init,
        requires=("context",),
    )(original)
    registry = WorkflowRegistry()
    registry.add(declared)

    assert declared is original
    assert calls == []
    assert registry.names() == ("original",)


def test_catalog_is_insertion_ordered_and_can_be_reloaded_independently() -> None:
    first = _declare("first", Plan, Knobs)
    second = _declare("second", Plan, NestedKnobs)
    registry = WorkflowRegistry()
    registry.add(second)
    previous_names = registry.names()
    registry.add(first)
    reloaded = WorkflowRegistry()
    reloaded.add(first)

    assert previous_names == ("second",)
    assert registry.names() == ("second", "first")
    assert reloaded.names() == ("first",)


def test_copied_metadata_does_not_declare_a_different_callable() -> None:
    declared = _declare("original", Plan, Knobs)

    @wraps(declared)
    def copied(
        env: WorkflowEnv[None], plan: Plan, tun: Knobs, state: State
    ) -> Step[State, object]:
        return (yield from declared(env, plan, tun, state))

    registry = WorkflowRegistry()
    with pytest.raises(ValueError, match="not declared"):
        registry.add(copied)
    assert registry.names() == ()


class OpaqueRecord:
    """A record type with no Pydantic schema or JSON encoder."""


def test_record_type_is_not_required_to_have_a_serialization_schema() -> None:
    def step(
        env: WorkflowEnv[None], plan: Plan, tun: Knobs, state: State
    ) -> Step[State, OpaqueRecord]:
        yield from ()
        return Done()

    declared = workflow(
        "opaque-record",
        plan=Plan,
        tunables=Knobs,
        state=State,
        record=OpaqueRecord,
        init=_init,
    )(step)
    registry = WorkflowRegistry()
    registry.add(declared)
    assert registry.names() == ("opaque-record",)


class EmptyTupleKnobs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: tuple[()] = ()


def test_empty_fixed_tuple_is_a_valid_tunable_leaf() -> None:
    registry = WorkflowRegistry()
    registry.add(_declare("empty-tuple", Plan, EmptyTupleKnobs))
    assert registry.names() == ("empty-tuple",)


def test_duplicate_name_is_rejected_without_changing_catalog() -> None:
    registry = WorkflowRegistry()
    registry.add(_declare("same", Plan, Knobs))
    duplicate = _declare("same", Plan, Knobs)

    with pytest.raises(ValueError, match="Duplicate workflow name"):
        registry.add(duplicate)

    assert registry.names() == ("same",)


def test_undeclared_function_is_rejected() -> None:
    registry = WorkflowRegistry()
    with pytest.raises(ValueError, match="not declared"):
        registry.add(_step)
    assert registry.names() == ()


@pytest.mark.parametrize("name", ["", "  "])
def test_empty_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        _declare(name, Plan, Knobs)


class MutablePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = 1


class ExtraPlan(BaseModel):
    model_config = ConfigDict(frozen=True)
    count: int = 1


@pytest.mark.parametrize("plan", [MutablePlan, ExtraPlan])
def test_plan_requires_frozen_extra_forbid_model(plan: type[BaseModel]) -> None:
    with pytest.raises(ValueError, match="frozen extra-forbid"):
        _declare("invalid-plan", plan, Knobs)


class ExtraKnobs(BaseModel):
    count: int = 1


class NestedExtraKnobs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nested: ExtraKnobs


class MappingKnobs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: dict[str, float]


class DateKnobs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: datetime


class SetKnobs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: set[float]


class UnstructuredKnobs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: object


@pytest.mark.parametrize(
    ("knobs", "message"),
    [
        (ExtraKnobs, "extra-forbid"),
        (NestedExtraKnobs, "model layers must forbid extras"),
        (MappingKnobs, "mappings are not tunables"),
        (DateKnobs, "JSON scalars"),
        (SetKnobs, "lists or tuples"),
        (UnstructuredKnobs, "JSON scalars"),
    ],
)
def test_unsupported_tunables_are_rejected(
    knobs: type[BaseModel], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _declare("invalid-knobs", Plan, knobs)


def test_state_must_be_a_dataclass_type() -> None:
    with pytest.raises(ValueError, match="dataclass type"):
        workflow(
            "invalid-state",
            plan=Plan,
            tunables=Knobs,
            state=int,
            record=str,
            init=lambda _env, _plan: 0,
        )


def test_capabilities_cannot_be_repeated() -> None:
    with pytest.raises(ValueError, match="unique known capabilities"):
        workflow(
            "invalid-requires",
            plan=Plan,
            tunables=Knobs,
            state=State,
            record=object,
            init=_init,
            requires=("soc", "soc"),
        )


def test_function_cannot_be_declared_twice() -> None:
    declared = _declare("once", Plan, Knobs)
    decorate = workflow(
        "again",
        plan=Plan,
        tunables=Knobs,
        state=State,
        record=object,
        init=_init,
    )
    with pytest.raises(ValueError, match="already declared"):
        decorate(declared)
