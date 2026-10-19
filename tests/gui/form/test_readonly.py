"""Readonly declarations guard ordinary Form mutations, not saved-input restore."""

from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, JsonValue
from zcu_tools.gui.form import (
    Display,
    FormCommandContext,
    FormCommandResult,
    FormCommandSnapshot,
    FormEnv,
    FormInput,
    FormInputRejected,
    FormModel,
    FormState,
    InputCommand,
)


def rename(
    _snapshot: FormCommandSnapshot, _context: FormCommandContext
) -> FormCommandResult:
    """Attempt a new name through the ordinary command seam."""
    return FormCommandResult(None, None, "changed")


class Tone(BaseModel):
    name: Annotated[
        str,
        Display(readonly=True),
        InputCommand("rename", "Rename", (), rename),
    ]
    gain: float = 1


class ReadonlyForm(FormModel):
    tones: list[Tone]
    mode: Annotated[Literal["a", "b"], Display(readonly=True)] = "a"
    editable: int = 0


@pytest.fixture
def readonly_state() -> FormState[ReadonlyForm]:
    return FormState(
        FormInput(
            ReadonlyForm,
            {
                "tones": [{"name": "Q0", "gain": 1}, {"name": "Q1", "gain": 2}],
                "mode": "a",
                "editable": 0,
            },
        )
    )


def test_list_positions_keep_distinct_readonly_values_and_allow_same_value_writes(
    readonly_state: FormState[ReadonlyForm],
) -> None:
    state = readonly_state
    env = FormEnv()
    state.write({("tones", 0, "name"): "Q0", ("tones", 1, "name"): "Q1"}, env)
    assert not state.is_dirty
    state.select(("mode",), "a", env)
    for index in (0, 1):
        field = state.describe(("tones", index, "name"), env).target
        assert field is not None and field.readonly and not field.fixed
    with pytest.raises(FormInputRejected) as caught:
        state.write({("tones", 1, "name"): "Q0"}, env)
    assert getattr(caught.value, "reason", None) == "readonly"
    assert getattr(caught.value, "current", None) == "Q1"
    assert caught.value.path == ("tones", 1, "name")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("tones", 0, "name"), "changed"),
        (("tones", 0), {"name": "changed", "gain": 3}),
        (("tones",), [{"name": "changed", "gain": 3}, {"name": "Q1", "gain": 2}]),
        (("tones", 0), {"gain": 3}),
        (("tones",), []),
    ],
)
def test_ancestor_and_batch_writes_cannot_change_or_drop_readonly_children(
    readonly_state: FormState[ReadonlyForm],
    path: tuple[str | int, ...],
    value: JsonValue,
) -> None:
    state = readonly_state
    before = state.input
    notifications = []
    state.subscribe(lambda: notifications.append(state.input))
    with pytest.raises(FormInputRejected) as caught:
        state.write({("editable",): 9, path: value}, FormEnv())
    assert caught.value.path == path
    assert getattr(caught.value, "reason", None) == "readonly"
    assert getattr(caught.value, "current", None) == "Q0"
    assert state.input.same_as(before)
    assert state.base.same_as(before)
    assert notifications == []


def mutate_readonly(state: FormState[ReadonlyForm], operation: str) -> None:
    """Submit the same forbidden changes through each public mutation seam."""
    env = FormEnv()
    if operation == "prepare":
        state.prepare_write({("tones", 0, "name"): "changed"}, env)
    elif operation == "install":
        candidate = state.input.data
        candidate["mode"] = "b"
        state.install_input(FormInput(ReadonlyForm, candidate))
    elif operation == "command":
        state.run_command(("tones", 0, "name"), "rename", env)
    else:
        state.select(("mode",), "b", env)


@pytest.mark.parametrize("operation", ["prepare", "install", "command", "select"])
def test_every_ordinary_mutation_entrypoint_enforces_readonly(
    readonly_state: FormState[ReadonlyForm], operation: str
) -> None:
    state = readonly_state
    before = state.input
    with pytest.raises(FormInputRejected) as caught:
        mutate_readonly(state, operation)
    assert getattr(caught.value, "reason", None) == "readonly"
    assert state.input.same_as(before)
    assert state.base.same_as(before)


def test_restore_preserves_saved_readonly_values_and_uses_them_for_later_admission(
    readonly_state: FormState[ReadonlyForm],
) -> None:
    state = readonly_state
    base = state.base
    saved = FormInput(
        ReadonlyForm,
        {"tones": [{"name": "saved0", "gain": 4}, {"name": "saved1", "gain": 5}]},
    )
    assert state.restore_input(saved, FormEnv()) == ()
    assert state.input.data["tones"] == saved.data["tones"]
    assert state.base.same_as(base)
    state.write({("tones", 1, "name"): "saved1"}, FormEnv())
    with pytest.raises(FormInputRejected) as caught:
        state.write({("tones", 1, "name"): "Q1"}, FormEnv())
    assert getattr(caught.value, "current", None) == "saved1"


class Inline(BaseModel):
    kind: Literal["inline"] = "inline"
    name: Annotated[str, Display(readonly=True)] = "fixed"
    gain: float = 1


class Ref(BaseModel):
    kind: Literal["ref"] = "ref"
    key: str


class BranchForm(FormModel):
    module: Annotated[Inline | Ref, Field(discriminator="kind")]


class ReadonlyBranchForm(FormModel):
    module: Annotated[Inline | Ref, Field(discriminator="kind"), Display(readonly=True)]


def test_readonly_children_do_not_freeze_ancestor_branch_choices() -> None:
    state = FormState(
        FormInput(BranchForm, {"module": {"kind": "inline", "name": "A"}})
    )
    env = FormEnv()
    state.select(("module",), {"kind": "ref"}, env)
    assert state.input.selections == {("module",): {"kind": "ref"}}
    state.write({("module",): {"kind": "ref", "key": "library-key"}}, env)
    assert state.input.data == {"module": {"kind": "ref", "key": "library-key"}}
    state.write({("module",): {"kind": "inline", "name": "B"}}, env)
    assert state.input.data == {"module": {"kind": "inline", "name": "B"}}
    with pytest.raises(FormInputRejected):
        state.write({("module", "name"): "C"}, env)


def mutate_branch(state: FormState[ReadonlyBranchForm], operation: str) -> None:
    """Submit subtree, selector or child changes to the same readonly model."""
    env = FormEnv()
    if operation == "write":
        state.write({("module",): {"kind": "ref", "key": "library-key"}}, env)
    elif operation == "select":
        state.select(("module",), {"kind": "ref"}, env)
    else:
        state.write({("module", "gain"): 2}, env)


@pytest.mark.parametrize("operation", ["write", "select", "child"])
def test_readonly_model_freezes_its_entire_subtree_and_selector(operation: str) -> None:
    state = FormState(
        FormInput(
            ReadonlyBranchForm, {"module": {"kind": "inline", "name": "A", "gain": 1}}
        )
    )
    before = state.input
    with pytest.raises(FormInputRejected) as caught:
        mutate_branch(state, operation)
    assert getattr(caught.value, "current", None) == before.data["module"]
    assert state.input.same_as(before)


def test_ancestor_write_with_unchanged_readonly_children_can_edit_other_values(
    readonly_state: FormState[ReadonlyForm],
) -> None:
    readonly_state.write({("tones", 0): {"name": "Q0", "gain": 8}}, FormEnv())
    assert readonly_state.input.data["tones"] == [
        {"name": "Q0", "gain": 8},
        {"name": "Q1", "gain": 2},
    ]


def test_readonly_model_same_selection_after_restore_remains_admissible() -> None:
    state = FormState(
        FormInput(ReadonlyBranchForm, {"module": {"kind": "inline", "name": "fresh"}})
    )
    saved = FormInput(
        ReadonlyBranchForm,
        {"module": {"kind": "inline", "name": "saved"}},
        selections={("module",): {"kind": "inline"}},
    )
    assert state.restore_input(saved, FormEnv()) == ()
    state.select(("module",), {"kind": "inline"}, FormEnv())
    assert state.input.data == saved.data
    with pytest.raises(FormInputRejected):
        state.select(("module",), {"kind": "ref"}, FormEnv())


def test_nullable_ancestor_can_deactivate_without_allowing_readonly_child_edits() -> (
    None
):
    class NullableForm(FormModel):
        item: Tone | None

    state = FormState(FormInput(NullableForm, {"item": {"name": "Q0", "gain": 1}}))
    with pytest.raises(FormInputRejected):
        state.write({("item", "name"): "Q1"}, FormEnv())
    state.write({("item",): None}, FormEnv())
    assert state.input.data == {"item": None}


def test_readonly_json_comparison_does_not_treat_boolean_as_the_same_number() -> None:
    class NumericForm(FormModel):
        value: Annotated[list[int], Display(readonly=True)]

    state = FormState(FormInput(NumericForm, {"value": [1]}))
    with pytest.raises(FormInputRejected) as caught:
        state.write({("value",): [True]}, FormEnv())
    assert getattr(caught.value, "current", None) == [1]
    assert state.input.data == {"value": [1]}
