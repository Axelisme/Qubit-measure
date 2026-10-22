"""Hidden and fixed metadata preserve authored input and validation."""

from typing import Annotated, Literal

import pytest
from pydantic import Field
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormState,
)
from zcu_tools.gui.form_adapter import LinRange


def test_hidden_field_stays_in_description_input_and_validation() -> None:
    class HiddenForm(FormModel):
        shown: float = 1.0
        gain: Annotated[float, Display(hidden=True)] = Field(default=0.5, gt=0)

    env = FormEnv()
    state = FormState(HiddenForm.default_input(env))
    fields = {field.path[-1]: field for field in state.describe((), env).children}
    assert fields["gain"].hidden and not fields["shown"].hidden
    state.write({("shown",): 2.0}, env)
    assert state.input.data == {"shown": 2.0, "gain": 0.5}
    state.write({("gain",): -1.0}, env)
    errors = state.result(env)
    assert isinstance(errors, FormErrors)
    assert set(errors.messages) == {("gain",)}


def test_root_fixed_tag_is_hidden_without_changing_defaults_or_validation() -> None:
    class RangeForm(LinRange):
        fixed: Literal[2] = 2

    env = FormEnv()
    state = FormState(
        FormInput(
            RangeForm, {"kind": "linrange", "start": 0, "stop": 1, "n": 3, "fixed": 2}
        )
    )
    fields = {field.path[-1]: field for field in state.describe((), env).children}
    assert fields["kind"].hidden and fields["kind"].fixed_value == "linrange"
    assert not fields["fixed"].hidden
    assert state.input.data["kind"] == "linrange"
    state.write({("kind",): "other"}, env)
    errors = state.result(env)
    assert isinstance(errors, FormErrors)
    assert set(errors.messages) == {("kind",)}


def test_display_group_rejects_an_empty_name() -> None:
    with pytest.raises(ValueError, match="group"):
        Display(group="")
