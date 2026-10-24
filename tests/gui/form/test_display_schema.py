"""Hidden and fixed metadata preserve authored input and validation."""

from typing import Annotated, Literal

import pytest
from pydantic import Field
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormErrors,
    FormInput,
    FormInputRejected,
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


def test_field_label_retains_alias_selector() -> None:
    class OverlayForm(FormModel):
        type Choice = Annotated[Literal["a", "b"], Display(selector=False)]

        choice: Annotated[Choice, Display(label="Mode")] = "a"

    env = FormEnv()
    field = FormState(OverlayForm.default_input(env)).describe((), env).children[0]
    assert field.title == "Mode"
    assert field.selector is False
    assert field.selected == 0


def test_explicit_selector_true_overrides_false_without_losing_alias_label() -> None:
    class OverlayForm(FormModel):
        type Choice = Annotated[
            Literal["a", "b"], Display(label="Alias mode", selector=False)
        ]

        choice: Annotated[Choice, Display(selector=True)] = "a"

    env = FormEnv()
    field = FormState(OverlayForm.default_input(env)).describe((), env).children[0]
    assert field.selector is True
    assert field.title == "Alias mode"


def test_explicit_false_and_none_clear_only_named_display_properties() -> None:
    class OverlayForm(FormModel):
        type Gain = Annotated[
            float,
            Display(
                label="Alias gain",
                description="Gain hint",
                unit="V",
                placeholder="Voltage",
                decimals=2,
                group="Advanced",
                hidden=True,
                readonly=True,
                selector=False,
            ),
        ]

        gain: Annotated[
            Gain,
            Display(
                label=None,
                description=None,
                unit=None,
                placeholder=None,
                decimals=None,
                group=None,
                hidden=False,
                readonly=False,
            ),
        ] = 1.0

    env = FormEnv()
    state = FormState(OverlayForm.default_input(env))
    field = state.describe((), env).children[0]
    assert (field.title, field.description, field.unit, field.placeholder) == (
        "Gain",
        None,
        None,
        None,
    )
    assert (field.decimals, field.group, field.hidden, field.readonly) == (
        None,
        None,
        False,
        False,
    )
    assert field.selector is False
    state.write({("gain",): 2.0}, env)
    assert state.input.data == {"gain": 2.0}


def test_three_annotated_layers_overlay_from_inner_to_outer() -> None:
    class OverlayForm(FormModel):
        type Inner = Annotated[
            float, Display(label="Inner", unit="V", decimals=3, selector=False)
        ]
        type Middle = Annotated[Inner, Display(label="Middle", unit="mV")]

        gain: Annotated[Middle, Display(label="Outer", decimals=None)] = 1.0

    env = FormEnv()
    field = FormState(OverlayForm.default_input(env)).describe((), env).children[0]
    assert (field.title, field.unit, field.decimals) == ("Outer", "mV", None)
    assert field.selector is False


def test_label_overlay_preserves_alias_readonly_admission() -> None:
    class OverlayForm(FormModel):
        type Gain = Annotated[float, Display(readonly=True)]

        gain: Annotated[Gain, Display(label="Gain")] = 1.0

    env = FormEnv()
    state = FormState(OverlayForm.default_input(env))
    with pytest.raises(FormInputRejected, match="Readonly"):
        state.write({("gain",): 2.0}, env)
    assert state.input.data == {"gain": 1.0}


def test_label_overlay_preserves_model_list_element_labels() -> None:
    class Tone(FormModel):
        name: str

    class OverlayForm(FormModel):
        type Tones = Annotated[list[Tone], Display(element_label="name")]

        tones: Annotated[Tones, Display(label="Participants")] = [Tone(name="Q1")]

    env = FormEnv()
    state = FormState(OverlayForm.default_input(env))
    field = state.describe((), env).children[0]
    assert field.elements is not None
    assert field.elements[0].name == "Q1"
    assert state.describe(("tones", 0), env).target is not None
    assert state.describe(("tones",), env).children[0].title == "Q1"


@pytest.mark.parametrize(
    "outer, expected", [(Display(), False), (Display(selector=True), True)]
)
def test_omitted_and_explicit_default_remain_distinct_on_same_annotation(
    outer: Display,
    expected: bool,
) -> None:
    class OverlayForm(FormModel):
        type Choice = Annotated[
            Literal["a", "b"], Display(label="Mode", selector=False)
        ]

        inherited: Annotated[Choice, Display()] = "a"
        explicit: Annotated[Choice, Display(selector=True)] = "a"
        repeated: Annotated[Choice, outer] = "a"

    env = FormEnv()
    fields = FormState(OverlayForm.default_input(env)).describe((), env).children
    assert fields[0].selector is False
    assert fields[1].selector is True
    assert fields[2].selector is expected
