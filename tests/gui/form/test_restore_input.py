"""Name-and-shape restoration through the public FormState input boundary."""

from enum import StrEnum
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, JsonValue
from zcu_tools.gui.form import FormEnv, FormInput, FormModel, FormPath, FormState


class Tone(BaseModel):
    name: str
    amount: float = 1
    phase: float = 9


class Ramp(BaseModel):
    kind: Literal["ramp"] = "ramp"
    length: float = 2
    added: int = 7


class Flat(BaseModel):
    kind: Literal["flat"] = "flat"
    length: float = 3


class RestoreForm(FormModel):
    tones: list[Tone] = Field(default_factory=list)
    shape: Annotated[Ramp | Flat, Field(discriminator="kind")] = Field(
        default_factory=Ramp
    )
    mode: Literal["keep", "new"] = "new"
    value: float = 5
    required: float
    added: int = 11


@pytest.mark.parametrize("saved_length", [0, 1, 3])
def test_restore_list_keeps_saved_positions_and_length(saved_length: int) -> None:
    base = FormInput(
        RestoreForm, {"tones": [{"name": "fresh", "amount": 99}], "added": 11}
    )
    state = FormState(base)
    saved = FormInput(
        RestoreForm,
        {
            "tones": [
                {"name": f"Q{i}", "amount": {"expr": "missing + 1"}, "retired": 6}
                for i in range(saved_length)
            ]
        },
    )

    dropped = state.restore_input(saved, FormEnv())

    assert state.input.data["tones"] == [
        {"name": f"Q{i}", "amount": {"expr": "missing + 1"}, "phase": 9}
        for i in range(saved_length)
    ]
    assert set(dropped) == {("tones", i, "retired") for i in range(saved_length)}
    assert state.input.data["added"] == 11
    assert state.base.same_as(base)


def test_restore_list_uses_captured_defaults_at_the_saved_position() -> None:
    base = FormInput(
        RestoreForm, {"tones": [{"name": "fresh", "amount": 99, "phase": 123}]}
    )
    state = FormState(base)
    saved = FormInput(RestoreForm, {"tones": [{"name": "Q1", "amount": 17}]})

    assert state.restore_input(saved, FormEnv()) == ()
    assert state.input.data["tones"] == [{"name": "Q1", "amount": 17, "phase": 123}]


@pytest.mark.parametrize(
    ("raw", "expected", "dropped"),
    [
        (None, None, ()),
        ([], [], ()),
        (
            [{"x": "unfinished", "removed": 4}],
            [{"x": "unfinished", "added": 7}],
            (("items", 0, "removed"),),
        ),
    ],
)
def test_restore_nullable_list_recurses_its_only_non_null_branch(
    raw: JsonValue, expected: JsonValue, dropped: tuple[FormPath, ...]
) -> None:
    class Item(BaseModel):
        x: int = 1
        added: int = 7

    class NullableList(FormModel):
        items: list[Item] | None = None

    state = FormState(FormInput(NullableList, {"items": None}))
    saved = FormInput(NullableList, {"items": raw})

    actual_dropped = state.restore_input(saved, FormEnv())

    assert state.input.data == {"items": expected}
    assert actual_dropped == dropped


def test_restore_keeps_invalid_leaf_raw_and_reports_only_shape_or_removed_fields() -> (
    None
):
    state = FormState(RestoreForm.default_input(FormEnv()))
    saved = FormInput(
        RestoreForm,
        {
            "value": "unfinished",
            "required": {"expr": "unbound"},
            "retired": 1,
            "shape": [1, 2],
        },
    )

    dropped = state.restore_input(saved, FormEnv())

    assert state.input.data["value"] == "unfinished"
    assert state.input.data["required"] == {"expr": "unbound"}
    assert state.input.data["shape"] == {"kind": "ramp", "length": 2, "added": 7}
    assert set(dropped) == {("retired",), ("shape",)}


def test_restore_rejects_object_to_scalar_shape_but_keeps_missing_required() -> None:
    state = FormState(RestoreForm.default_input(FormEnv()))
    saved = FormInput(RestoreForm, {"value": {"old_field": 3}})

    assert state.restore_input(saved, FormEnv()) == (("value",),)
    assert state.input.data["value"] == 5
    assert "required" not in state.input.data


def test_restore_invalid_selection_uses_current_default_and_reports_once() -> None:
    state = FormState(RestoreForm.default_input(FormEnv()))
    saved = FormInput(
        RestoreForm,
        {"shape": {"kind": "flat", "length": 88}, "mode": "keep"},
        selections={("shape",): {"kind": "retired"}, ("mode",): "retired"},
    )

    dropped = state.restore_input(saved, FormEnv())

    assert state.input.data["shape"] == {"kind": "ramp", "length": 2, "added": 7}
    assert state.input.data["mode"] == "new"
    assert state.input.selections == {}
    assert set(dropped) == {("shape",), ("mode",)}


def test_saved_enum_selection_is_accepted_by_its_current_annotation() -> None:
    class Speed(StrEnum):
        slow = "slow"
        fast = "fast"

    class EnumForm(FormModel):
        speed: Speed = Speed.slow

    authored = FormInput(EnumForm, {"speed": "slow"}, selections={("speed",): "fast"})
    saved = FormInput.from_json(EnumForm, authored.to_json())
    state = FormState(EnumForm.default_input(FormEnv()))

    assert state.restore_input(saved, FormEnv()) == ()
    assert state.input.same_as(authored)


def test_restore_untagged_union_keeps_raw_without_guessing_or_filling_branch() -> None:
    class Left(BaseModel):
        left: float
        added: int = 8

    class Right(BaseModel):
        right: float

    class Untagged(FormModel):
        branch: Left | Right

    state = FormState(Untagged.default_input(FormEnv()))
    saved = FormInput(Untagged, {"branch": {"left": "unfinished", "unknown": 4}})

    assert state.restore_input(saved, FormEnv()) == ()
    assert state.input.data == saved.data
