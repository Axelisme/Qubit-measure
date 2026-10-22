"""Representation commands respect the enclosing handwritten model's branches."""

from typing import Annotated

import pytest
from pydantic import Field
from zcu_tools.gui.form import (
    FormCommandError,
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormState,
)
from zcu_tools.gui.form_adapter import CenterSpan, LinRange, SweepInput


class RequiredSweepForm(FormModel):
    sweep: SweepInput


class DefaultSweepForm(FormModel):
    sweep: SweepInput = Field(default_factory=lambda: CenterSpan(center=3, span=2, n=5))


class ExplicitSweepForm(FormModel):
    sweep: SweepInput = CenterSpan(center=3, span=2, n=5)


def test_required_sweep_fresh_input_selects_range_without_coordinates():
    env = FormEnv()
    input = RequiredSweepForm.default_input(env)
    assert input.data == {"sweep": {"kind": "linrange"}}
    state = FormState(input)
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert result.messages == {
        ("sweep", name): "Field required" for name in ("start", "stop", "n")
    }
    description = state.describe(("sweep",), env)
    assert description.target is not None
    assert description.target.selected == 0
    assert [field.name for field in description.children] == [
        "start",
        "stop",
        "n",
    ]


@pytest.mark.parametrize("model", [DefaultSweepForm, ExplicitSweepForm])
def test_sweep_explicit_defaults_keep_authored_coordinates(model):
    env = FormEnv()
    input = model.default_input(env)
    assert input.data == {"sweep": {"kind": "center", "center": 3, "span": 2, "n": 5}}
    result = FormState(input).result(env)
    assert not isinstance(result, FormErrors)
    assert (result.sweep.start, result.sweep.stop, result.sweep.n) == (2, 4, 5)


@pytest.mark.parametrize(
    "raw",
    [
        {"kind": "linrange", "start": 2, "stop": 4, "n": 3},
        {"kind": "center", "center": 3, "span": 2, "n": 3},
        {"kind": "linrange_step", "start": 2, "stop": 4, "step": 1},
        {"kind": "center_step", "center": 3, "span": 2, "step": 1},
    ],
)
def test_sweep_authored_branch_retains_values(raw):
    env = FormEnv()
    state = FormState(FormInput(RequiredSweepForm, {"sweep": raw}))
    result = state.result(env)
    assert not isinstance(result, FormErrors)
    assert (result.sweep.start, result.sweep.stop, result.sweep.n) == (2, 4, 3)
    assert state.input.data == {"sweep": raw}


@pytest.mark.parametrize(
    ("raw", "missing"),
    [
        ({"kind": "linrange", "n": 3}, ("start", "stop")),
        ({"kind": "linrange", "start": 2, "stop": 4}, ("n",)),
        ({"kind": "center", "center": 3, "span": 2}, ("n",)),
    ],
)
def test_sweep_authored_partial_input_requires_missing_coordinates(raw, missing):
    env = FormEnv()
    state = FormState(FormInput(RequiredSweepForm, {"sweep": raw}))
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert result.messages == {("sweep", name): "Field required" for name in missing}


def test_required_sweep_coordinates_can_roundtrip_through_commands():
    env = FormEnv()
    state = FormState(RequiredSweepForm.default_input(env))
    state.write({("sweep",): {"kind": "linrange", "start": 2, "stop": 4, "n": 3}}, env)
    state.run_command(("sweep",), "to_center", env)
    centered = state.result(env)
    assert not isinstance(centered, FormErrors)
    assert (centered.sweep.start, centered.sweep.stop, centered.sweep.n) == (2, 4, 3)
    state.run_command(("sweep",), "to_range", env)
    assert state.input.data == {
        "sweep": {"kind": "linrange", "start": 2, "stop": 4, "n": 3}
    }
    result = state.result(env)
    assert not isinstance(result, FormErrors)


class FixedSweepForm(FormModel):
    sweep: LinRange = Field(default_factory=lambda: LinRange(start=0, stop=1, n=3))


class CoordinateSweepForm(FormModel):
    sweep: Annotated[LinRange | CenterSpan, Field(discriminator="kind")] = Field(
        default_factory=lambda: LinRange(start=0, stop=1, n=3)
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
