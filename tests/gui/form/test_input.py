"""Persistence and isolation through the public form input envelope."""

import json
import re
from collections.abc import Callable
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, JsonValue, create_model
from zcu_tools.gui.form import (
    FormCommandContext,
    FormCommandSnapshot,
    FormConversionResult,
    FormEnv,
    FormInput,
    FormModel,
    FormState,
    InputConverter,
)
from zcu_tools.gui.form.commands import ReadonlyFormInput


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
@pytest.mark.parametrize("entry", ["defaults", "state"])
def test_form_entry_rejects_untagged_string_or_model_with_field_path(
    model: type[FormModel], path: str, entry: str
) -> None:
    operations: dict[str, Callable[[], object]] = {
        "defaults": lambda: model.default_input(FormEnv()),
        "state": lambda: FormState(FormInput(model, {})),
    }
    operation = operations[entry]
    with pytest.raises(ValueError, match=rf"{re.escape(path)}:.*str.*Model"):
        operation()


@pytest.mark.parametrize("with_override", [False, True])
def test_json_codec_keeps_raw_fields_and_typed_selection_paths(with_override: bool) -> None:
    raw: dict[str, JsonValue] = {"data": {"expr": "unbound"}, "selections": [1, 2], "tones": [{"shape": {"style": "ramp", "length": "unfinished"}}]}
    selections = {("tones", 0, "shape"): {"style": "flat"}} if with_override else {}
    original = FormInput(FormModel, raw, selections=selections)

    text = original.to_json()

    assert json.loads(text) == {"data": raw, "selections": [{"path": ["tones", 0, "shape"], "value": {"style": "flat"}}] if with_override else []}
    assert FormInput.from_json(FormModel, text).same_as(original)


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


class DraftForm(FormModel):
    shape: InlineShape
    optional: float | None = None
    required: float


def test_saved_selection_and_branch_restore_the_authored_invalid_draft() -> None:
    env = FormEnv()
    original = FormState(
        FormInput(
            DraftForm,
            {
                "shape": {"style": "ramp", "length": {"expr": "unbound + 2"}},
                "optional": None,
            },
        )
    )
    original.write({("shape",): {"style": "flat"}}, env)
    restored = FormState(FormInput.from_json(DraftForm, original.input.to_json()))
    assert restored.input.same_as(original.input)
    target = restored.describe(("shape",), env).target
    assert target is not None
    assert target.selected == 1
    assert target.branch == {"style": "ramp"}
    assert restored.input.data["optional"] is None
    assert "required" not in restored.input.data
    restored.write({("shape", "length"): "not a number"}, env)
    invalid = FormInput.from_json(DraftForm, restored.input.to_json())
    assert invalid.same_as(restored.input)


def test_restore_uses_current_trusted_declaration_without_running_converter() -> None:
    def forbidden(
        old: FormCommandSnapshot,
        selection: ReadonlyFormInput,
        context: FormCommandContext,
    ) -> FormConversionResult:
        raise AssertionError("Restore must not execute a converter")

    def declaration() -> type[FormModel]:
        return create_model(
            "RegeneratedDraft",
            __base__=DraftForm,
            shape=(
                Annotated[InlineShape, InputConverter(forbidden)],
                Field(discriminator="style"),
            ),
        )

    previous = declaration()
    trusted = declaration()
    original = FormInput(
        previous,
        {"shape": {"style": "ramp", "length": -3}},
        selections={("shape",): {"style": "flat"}},
    )
    restored = FormInput.from_json(trusted, original.to_json())
    assert restored.model is trusted
    assert restored.data == original.data
    assert restored.selections == original.selections
    target = FormState(restored).describe(("shape",), FormEnv()).target
    assert target is not None and target.selected == 1


def test_saved_draft_rejects_changed_type_identity_before_installation() -> None:
    previous = create_model("TypedDraft", __base__=FormModel, value=(float, ...))
    changed = create_model("TypedDraft", __base__=FormModel, value=(str, ...))
    saved = FormInput(previous, {"value": "invalid even for the original type"})
    with pytest.raises(ValueError, match="identity|incompatible"):
        FormInput.from_json(changed, saved.to_json())


@pytest.mark.parametrize(
    ("raw", "selections"),
    [
        ({"shape": {"style": "unknown"}}, {}),
        ({"shape": {"style": "ramp"}}, {("shape",): {"style": "unknown"}}),
        ({"shape": {"style": "ramp"}}, {("removed",): "flat"}),
    ],
    ids=["branch", "literal-selection", "selection-path"],
)
def test_restore_rejects_annotation_incompatible_branch_or_selection(
    raw, selections
) -> None:
    saved = FormInput(DraftForm, raw, selections=selections).to_json()
    with pytest.raises(ValueError, match="identity|incompatible"):
        FormInput.from_json(DraftForm, saved)


def test_untagged_model_union_draft_restores_without_guessing_a_branch() -> None:
    class Left(BaseModel):
        left: float

    class Right(BaseModel):
        right: float

    class Other(BaseModel):
        other: float

    model = create_model("UnionDraft", __base__=FormModel, branch=(Left | Right, ...))
    original = FormInput(model, {"branch": {"left": "unfinished"}})
    restored = FormInput.from_json(model, original.to_json())
    assert restored.same_as(original)
    assert FormState(restored).preview(FormEnv()).errors is not None
    changed = create_model("UnionDraft", __base__=FormModel, branch=(Left | Other, ...))
    with pytest.raises(ValueError, match="identity|incompatible"):
        FormInput.from_json(changed, original.to_json())


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
