"""Representation commands respect the enclosing handwritten model's branches."""

from typing import Annotated

import pytest
from pydantic import Field
from zcu_tools.gui.form import (
    CenterSpan,
    FormCommandError,
    FormEnv,
    FormErrors,
    FormModel,
    FormState,
    LinRange,
)


class FixedSweepForm(FormModel):
    sweep: LinRange = Field(default_factory=LinRange)


class CoordinateSweepForm(FormModel):
    sweep: Annotated[LinRange | CenterSpan, Field(discriminator="kind")] = Field(
        default_factory=LinRange
    )


@pytest.mark.parametrize(
    ("field", "command"), [("start", "to_center"), ("n", "to_step")]
)
def test_single_branch_sweep_does_not_offer_or_install_other_kinds(field, command):
    env = FormEnv()
    state = FormState(FixedSweepForm.default_input(env))
    before = state.input.to_json()
    description = state.describe(("sweep", field), env).target
    assert description is not None
    assert description.commands == (), "A fixed LinRange cannot accept another kind"
    with pytest.raises(FormCommandError):
        state.run_command(("sweep", field), command, env)
    assert state.input.to_json() == before


def test_coordinate_only_union_does_not_offer_sampling_conversion():
    env = FormEnv()
    state = FormState(CoordinateSweepForm.default_input(env))
    description = state.describe(("sweep", "n"), env).target
    assert description is not None
    assert description.commands == (), "Expts-only annotations cannot accept Step"
    before = state.input.to_json()
    with pytest.raises(FormCommandError):
        state.run_command(("sweep", "n"), "to_step", env)
    assert state.input.to_json() == before
    state.run_command(("sweep", "start"), "to_center", env)
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
    state.run_command(("sweep", "center"), "to_range", env)
    assert not isinstance(state.result(env), FormErrors)
    assert state.input.to_json() == before
