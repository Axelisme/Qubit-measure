"""Representation commands respect the enclosing handwritten model's branches."""

from typing import Annotated

import pytest
from pydantic import Field
from zcu_tools.gui.form import (
    FormCommandError,
    FormEnv,
    FormErrors,
    FormModel,
    FormState,
)
from zcu_tools.gui.form_adapter import CenterSpan, LinRange


class FixedSweepForm(FormModel):
    sweep: LinRange = Field(default_factory=LinRange)


class CoordinateSweepForm(FormModel):
    sweep: Annotated[LinRange | CenterSpan, Field(discriminator="kind")] = Field(
        default_factory=LinRange
    )


@pytest.mark.parametrize("command", ["to_center", "to_step"])
def test_single_branch_sweep_does_not_offer_or_install_other_kinds(command):
    env = FormEnv()
    state = FormState(FixedSweepForm.default_input(env))
    before = state.input.to_json()
    description = state.describe(("sweep",), env).target
    assert description is not None
    assert description.commands == (), "A fixed LinRange cannot accept another kind"
    with pytest.raises(FormCommandError):
        state.run_command(("sweep",), command, env)
    assert state.input.to_json() == before


def test_coordinate_only_union_does_not_offer_sampling_conversion():
    env = FormEnv()
    state = FormState(CoordinateSweepForm.default_input(env))
    description = state.describe(("sweep",), env).target
    assert description is not None
    assert [command.name for command in description.commands] == ["to_center"]
    before = state.input.to_json()
    with pytest.raises(FormCommandError):
        state.run_command(("sweep",), "to_step", env)
    assert state.input.to_json() == before
    state.run_command(("sweep",), "to_center", env)
    assert [child.name for child in state.describe(("sweep",), env).children] == [
        "center",
        "span",
        "n",
    ]
    assert state.input.data["sweep"] == {
        "kind": "center",
        "center": 0.5,
        "span": 1.0,
        "n": 3,
    }
    state.run_command(("sweep",), "to_range", env)
    assert not isinstance(state.result(env), FormErrors)
    assert state.input.to_json() == before
