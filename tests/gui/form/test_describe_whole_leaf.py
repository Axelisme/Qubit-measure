"""Authored option fields never turn whole leaves into child-editable models."""

from collections.abc import Mapping

import pytest
from pydantic import BaseModel, JsonValue
from zcu_tools.gui.form import (
    FormEnv,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    Option,
)


class SuggestedFields(BaseModel):
    frequency: float


class WholeLeafForm(FormModel):
    mapping: dict[str, float]
    sequence: tuple[float, ...]

    @classmethod
    def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[Option, ...]]:
        return {
            (name,): (Option("Object", {}, fields=SuggestedFields),)
            for name in ("mapping", "sequence")
        }


@pytest.mark.parametrize("name", ["mapping", "sequence"])
def test_option_fields_cannot_open_whole_leaf_descendants(name: str) -> None:
    assert hasattr(FormState, "describe"), "whole leaves need single-layer description"
    # A malformed sequence draft is still readable; metadata cannot change its shape.
    state = FormState(FormInput(WholeLeafForm, {name: {}}))
    env = FormEnv()
    description = state.describe((name,), env)
    assert description.target is not None
    assert description.target.kind == name
    assert description.target.selected == 0
    assert description.children == ()
    with pytest.raises(ValueError, match=name):
        state.describe((name, "frequency"), env)
    assert state.input.data == state.base.data == {name: {}}
