"""Neutral model lists expose positions without assigning element identities."""

from typing import Annotated

import pytest
from pydantic import BaseModel, Field
from zcu_tools.gui.form import (
    FormCommandContext,
    FormCommandResult,
    FormCommandSnapshot,
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormState,
    InputCommand,
)


def double_gain(
    _snapshot: FormCommandSnapshot, context: FormCommandContext
) -> FormCommandResult:
    """Return a replacement for the indexed scalar mounted by the declaration."""
    gain = context.evaluate(())
    assert isinstance(gain, (int, float))
    return FormCommandResult(selection=None, structure=None, input=gain * 2)


class Item(BaseModel):
    name: str
    gain: Annotated[
        float,
        Field(gt=0),
        InputCommand("double", "Double", ((),), double_gain),
    ]


class Section(FormModel):
    items: list[Item]
    participants: list[str]


class ListForm(FormModel):
    section: Section


def list_state() -> FormState[ListForm]:
    """Two ordinary items deliberately share a name; participants is opaque."""
    return FormState(
        FormInput(
            ListForm,
            {
                "section": {
                    "items": [{"name": "same", "gain": 1}, {"name": "same", "gain": 2}],
                    "participants": [],
                }
            },
        )
    )


def test_plain_model_list_describes_and_edits_original_positions():
    state = list_state()
    env = FormEnv()
    target = state.describe(("section", "items"), env).target
    assert target is not None and target.can_expand
    assert target.elements is not None
    assert [item.path for item in target.elements] == [
        ("section", "items", 0),
        ("section", "items", 1),
    ]
    name = state.describe(("section", "items", 1, "name"), env).target
    assert name is not None and not name.fixed
    state.write({("section", "items", 1, "gain"): {"expr": "3 * 2"}}, env)
    assert state.preview(env).values[("section", "items", 1, "gain")] == 6
    assert isinstance(state.result(env), ListForm)
    state.write({("section", "items", 0, "gain"): -1}, env)
    errors = state.result(env)
    assert isinstance(errors, FormErrors)
    assert ("section", "items", 0, "gain") in errors.messages
    state.write({("section", "items"): [{"name": "new", "gain": 4}]}, env)
    assert len(state.describe(("section", "items"), env).children) == 1
    state.write({("section", "items"): []}, env)
    assert state.describe(("section", "items"), env).children == ()


def test_plain_model_list_runs_indexed_command_atomically():
    state = list_state()
    env = FormEnv()
    target = state.describe(("section", "items"), env).target
    assert target is not None and target.can_expand
    events = []
    state.subscribe(lambda: events.append(state.input))
    state.run_command(("section", "items", 1, "gain"), "double", env)
    assert state.preview(env).values[("section", "items", 1, "gain")] == 4
    assert state.preview(env).values[("section", "items", 0, "gain")] == 1
    assert len(events) == 1


@pytest.mark.parametrize("index", [-1, 2, True, "1"])
def test_plain_model_list_rejects_bad_index_with_original_write_path(index):
    state = list_state()
    env = FormEnv()
    target = state.describe(("section", "items"), env).target
    assert target is not None and target.can_expand
    path = ("section", "items", index, "gain")
    before = state.input
    with pytest.raises(ValueError, match="path") as caught:
        state.write({path: 9}, env)
    assert getattr(caught.value, "path", None) == path
    assert state.input.same_as(before)
