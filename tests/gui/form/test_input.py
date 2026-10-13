"""Persistence and isolation through the public form input envelope."""

import re
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, JsonValue
from zcu_tools.gui.form import FormEnv, FormInput, FormModel, FormState


class InlineTone(BaseModel):
    frequency: float = 100.0


class InlineRamp(BaseModel):
    style: Literal["ramp"] = "ramp"
    length: float = 1.0


class InlineFlat(BaseModel):
    style: Literal["flat"] = "flat"
    length: float = 1.0


type InlineShape = Annotated[InlineRamp | InlineFlat, Field(discriminator="style")]
type UntaggedShape = str | InlineShape


class UntaggedToneForm(FormModel):
    tone: str | InlineTone = "Q1.pulses.pi"


class NullableToneForm(FormModel):
    tone: str | InlineTone | None = None


class AliasedShapeForm(FormModel):
    shape: UntaggedShape


class NestedToneForm(FormModel):
    section: UntaggedToneForm | None = None


class SafeBranch(BaseModel):
    kind: Literal["safe"] = "safe"


class UntaggedBranch(BaseModel):
    kind: Literal["other"] = "other"
    tone: str | InlineTone


class InactiveToneForm(FormModel):
    section: Annotated[SafeBranch | UntaggedBranch, Field(discriminator="kind")] = (
        Field(default_factory=SafeBranch)
    )


@pytest.mark.parametrize(
    ("model", "path"),
    [
        (UntaggedToneForm, "tone"),
        (NullableToneForm, "tone"),
        (AliasedShapeForm, "shape"),
        (NestedToneForm, "section.tone"),
        (InactiveToneForm, "section.tone"),
    ],
)
@pytest.mark.parametrize("entry", ["schema", "defaults", "state"])
def test_form_entry_rejects_untagged_string_or_model_with_field_path(
    model: type[FormModel], path: str, entry: str
) -> None:
    with pytest.raises(ValueError, match=rf"{re.escape(path)}:.*str.*Model"):
        if entry == "schema":
            model.input_schema(FormEnv())
        elif entry == "defaults":
            model.default_input(FormEnv())
        else:
            FormState(FormInput(model, {}))


class InputForm(FormModel):
    """Ordinary values used to observe incomplete saved input."""

    required: float
    optional: float | None = None
    samples: tuple[float, ...] = ()


def test_input_round_trip_preserves_missing_null_and_scalar_envelopes() -> None:
    data: dict[str, JsonValue] = {
        "optional": None,
        "samples": [1.0, {"expr": "seed * 2"}],
        "amplitude": {"__complex__": [1.0, -2.0]},
    }
    original = FormInput(InputForm, data)
    data["samples"] = []
    exposed = original.data
    exposed["optional"] = 42.0

    restored = FormInput.from_json(InputForm, original.to_json())
    assert restored.model is InputForm
    assert restored.data == {
        "optional": None,
        "samples": [1.0, {"expr": "seed * 2"}],
        "amplitude": {"__complex__": [1.0, -2.0]},
    }
    assert "required" not in restored.data
    assert original.data == restored.data


@pytest.mark.parametrize("text", ["[]", "null", "{", '{"required": NaN}'])
def test_saved_input_rejects_invalid_json_envelopes(text: str) -> None:
    with pytest.raises(ValueError, match="JSON|json|object|finite"):
        FormInput.from_json(InputForm, text)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_input_rejects_nonfinite_json_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="finite|JSON|json"):
        FormInput(InputForm, {"required": value})


def test_input_keeps_unknown_fields_and_invalid_values_for_restoration() -> None:
    data: dict[str, JsonValue] = {
        "required": {"expr": 42},
        "optional": "not a number",
        "old_name": "saved before a model change",
    }
    assert (
        FormInput.from_json(InputForm, FormInput(InputForm, data).to_json()).data
        == data
    )
