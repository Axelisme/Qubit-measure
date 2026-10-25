"""Suggested scalar values never restrict the accepted value domain."""

from typing import Annotated

from zcu_tools.gui.form import (
    FormEnv,
    FormInput,
    FormModel,
    FormState,
    InputSuggestions,
    Option,
)


class SuggestedInput(FormModel):
    name: Annotated[
        str,
        InputSuggestions(lambda _: (Option("First", "one"), Option("Second", "two"))),
    ] = "outside"


def test_suggestions_preserve_unlisted_values_and_do_not_validate_membership():
    env = FormEnv()
    state = FormState(SuggestedInput.default_input(env))
    target = state.describe(("name",), env).target
    assert target is not None
    assert target.options is not None
    assert [(item.label, item.value) for item in target.options] == [
        ("First", "one"),
        ("Second", "two"),
    ]
    assert target.selected is None
    assert target.choices is None
    assert state.result(env) == SuggestedInput(name="outside")
    state.select(("name",), "two", env)
    assert state.result(env) == SuggestedInput(name="two")
    state.write({("name",): "another arbitrary value"}, env)
    assert state.result(env) == SuggestedInput(name="another arbitrary value")
    assert FormInput.from_json(SuggestedInput, state.input.to_json()).data == {
        "name": "another arbitrary value"
    }


def test_text_command_changes_presentation_only_and_notifies_views():
    env = FormEnv()
    state = FormState(SuggestedInput.default_input(env))
    before = state.input.to_json()
    notifications = []
    state.subscribe(lambda: notifications.append(state.is_dirty))
    target = state.describe(("name",), env).target
    assert target is not None
    assert [(item.name, item.label) for item in target.commands] == [
        ("toggle_text", "Edit as text")
    ]
    state.run_command(("name",), "toggle_text", env)
    target = state.describe((), env).children[0]
    assert not target.selector
    assert target.commands[0].label == "Choose from list"
    assert state.input.to_json() == before
    assert not state.is_dirty
    assert state.result(env) == SuggestedInput(name="outside")
    assert notifications == [False]
    state.write({("name",): "typed name"}, env)
    state.run_command(("name",), "toggle_text", env)
    target = state.describe(("name",), env).target
    assert target is not None and target.selector
    assert state.result(env) == SuggestedInput(name="typed name")
    restored = FormState(FormInput.from_json(SuggestedInput, state.input.to_json()))
    assert restored.describe((), env).children[0].selector
