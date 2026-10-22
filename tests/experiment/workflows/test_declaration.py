"""Async workflow declaration and catalog behavior at the authoring seam."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from functools import wraps
from types import CoroutineType

import pytest
from pydantic import BaseModel, ConfigDict
from zcu_tools.experiment import workflows as w
from zcu_tools.experiment.workflows.declaration import workflow_definition


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    count: int = 1


class Knobs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reps: int = 2


class Summary(BaseModel):
    count: int


async def script(env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[Knobs]) -> Summary:
    return Summary(count=plan.count + tun.current.reps)


def test_decorator_infers_models_preserves_coroutine_and_defers_execution() -> None:
    calls: list[str] = []

    async def original(
        env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[Knobs]
    ) -> Summary:
        calls.append("script")
        return Summary(count=plan.count + tun.current.reps)

    try:
        declared = w.workflow("example", "Count configured repetitions")(original)
    except TypeError as error:
        pytest.fail(f"The async authoring declaration must be accepted: {error}")
    definition = workflow_definition(declared)
    registry = w.WorkflowRegistry()
    registry.add(declared)

    assert declared is original
    assert calls == []
    assert definition.plan is Plan
    assert definition.tunables is Knobs
    assert definition.description == "Count configured repetitions"
    assert registry.names() == ("example",)


def test_catalog_rejects_copied_metadata_and_duplicate_names() -> None:
    async def original(
        env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[Knobs]
    ) -> Summary:
        return Summary(count=plan.count + tun.current.reps)

    try:
        declared = w.workflow("original", "A synthetic workflow")(original)
    except TypeError as error:
        pytest.fail(f"The async authoring declaration must be accepted: {error}")

    @wraps(declared)
    async def copied(
        env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[Knobs]
    ) -> Summary:
        return await declared(env, plan, tun)

    registry = w.WorkflowRegistry()
    registry.add(declared)
    with pytest.raises(ValueError, match="not declared"):
        registry.add(copied)
    with pytest.raises(ValueError, match="Duplicate workflow name"):
        registry.add(declared)
    assert registry.names() == ("original",)


@pytest.mark.parametrize("name", ["", "  "])
def test_empty_names_are_rejected(name: str) -> None:
    try:
        decorate = w.workflow(name, "A synthetic workflow")
    except TypeError as error:
        pytest.fail(
            f"The async authoring declaration must accept a description: {error}"
        )
    with pytest.raises(ValueError, match="name must not be empty"):
        decorate(script)


def test_synchronous_author_is_rejected() -> None:
    def synchronous(
        env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[Knobs]
    ) -> CoroutineType[None, None, Summary]:
        return script(env, plan, tun)

    try:
        decorate = w.workflow("synchronous", "Invalid synchronous author")
    except TypeError as error:
        pytest.fail(
            f"The async authoring declaration must accept a description: {error}"
        )
    with pytest.raises(TypeError, match="coroutine"):
        decorate(synchronous)


class MutablePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = 1


class ExtraPlan(BaseModel):
    model_config = ConfigDict(frozen=True)
    count: int = 1


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


class OpaqueKnobs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: object


async def mutable_plan(
    env: w.WorkflowEnv[None], plan: MutablePlan, tun: w.Live[Knobs]
) -> Summary:
    raise AssertionError("Declaration must not execute the author")


async def extra_plan(
    env: w.WorkflowEnv[None], plan: ExtraPlan, tun: w.Live[Knobs]
) -> Summary:
    raise AssertionError("Declaration must not execute the author")


async def extra_knobs(
    env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[ExtraKnobs]
) -> Summary:
    raise AssertionError("Declaration must not execute the author")


async def nested_extra_knobs(
    env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[NestedExtraKnobs]
) -> Summary:
    raise AssertionError("Declaration must not execute the author")


async def mapping_knobs(
    env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[MappingKnobs]
) -> Summary:
    raise AssertionError("Declaration must not execute the author")


async def date_knobs(
    env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[DateKnobs]
) -> Summary:
    raise AssertionError("Declaration must not execute the author")


async def set_knobs(
    env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[SetKnobs]
) -> Summary:
    raise AssertionError("Declaration must not execute the author")


async def opaque_knobs(
    env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[OpaqueKnobs]
) -> Summary:
    raise AssertionError("Declaration must not execute the author")


@pytest.mark.parametrize(
    ("author", "message"),
    [
        (mutable_plan, "frozen extra-forbid"),
        (extra_plan, "frozen extra-forbid"),
        (extra_knobs, "extra-forbid"),
        (nested_extra_knobs, "model layers must forbid extras"),
        (mapping_knobs, "mappings are not tunables"),
        (date_knobs, "JSON scalars"),
        (set_knobs, "lists or tuples"),
        (opaque_knobs, "JSON scalars"),
    ],
)
def test_signature_inference_keeps_model_validation(
    author: Callable[..., CoroutineType[None, None, Summary]], message: str
) -> None:
    try:
        decorate = w.workflow("invalid", "Reject invalid input models")
    except TypeError as error:
        pytest.fail(f"The async declaration must accept a description: {error}")
    with pytest.raises(ValueError, match=message):
        decorate(author)
