"""Async workflow declaration and catalog behavior at the authoring seam."""

from __future__ import annotations

from functools import wraps

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


async def script(env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[Knobs]) -> int:
    return plan.count + tun.current.reps


def test_decorator_infers_models_preserves_coroutine_and_defers_execution() -> None:
    calls: list[str] = []

    async def original(
        env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[Knobs]
    ) -> int:
        calls.append("script")
        return plan.count + tun.current.reps

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
    try:
        declared = w.workflow("original", "A synthetic workflow")(script)
    except TypeError as error:
        pytest.fail(f"The async authoring declaration must be accepted: {error}")

    @wraps(declared)
    async def copied(
        env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[Knobs]
    ) -> int:
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
        pytest.fail(f"The async authoring declaration must accept a description: {error}")
    with pytest.raises(ValueError, match="name must not be empty"):
        decorate(script)


def test_synchronous_author_is_rejected() -> None:
    def synchronous(
        env: w.WorkflowEnv[None], plan: Plan, tun: w.Live[Knobs]
    ) -> int:
        return 1

    try:
        decorate = w.workflow("synchronous", "Invalid synchronous author")
    except TypeError as error:
        pytest.fail(f"The async authoring declaration must accept a description: {error}")
    with pytest.raises(TypeError, match="coroutine"):
        decorate(synchronous)
