"""Active child capability is available before describing the next layer."""

from collections.abc import Mapping
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, JsonValue
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    Option,
)
from zcu_tools.gui.form_adapter import KeyRef


class Child(BaseModel):
    value: float


class Empty(BaseModel):
    pass


class Inline(FormModel):
    kind: Literal["inline"] = "inline"
    module: Child


class EmptyInline(FormModel):
    kind: Literal["empty"] = "empty"
    module: Empty


class TagOnly(FormModel):
    kind: Literal["tag-only"] = "tag-only"


class CapabilityForm(FormModel):
    ordinary: Child
    empty: Empty
    optional: Child | None = None
    hidden: Annotated[Child, Display(hidden=True)]
    selection: Annotated[
        Inline | EmptyInline | TagOnly | KeyRef | None, Field(discriminator="kind")
    ] = None
    scalar: float
    sequence: list[Child]
    mapping: dict[str, Child]

    @classmethod
    def options(cls, _env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        return {
            ("selection",): (
                Option("None", None),
                Option("Ref", {"kind": "ref", "name": "saved"}),
                Option("Inline", {"kind": "inline"}, fields=Inline),
                Option("Empty", {"kind": "empty"}, fields=EmptyInline),
                Option("Tag only", {"kind": "tag-only"}, fields=TagOnly),
            )
        }


@pytest.mark.parametrize(
    "name,data,expected,nonempty",
    [
        ("ordinary", {}, True, True),
        ("ordinary", {"ordinary": {"value": "invalid"}}, True, True),
        ("empty", {}, True, False),
        ("optional", {}, False, False),
        ("optional", {"optional": None}, False, False),
        ("optional", {"optional": {}}, True, True),
        ("hidden", {}, True, True),
        ("selection", {}, False, False),
        ("selection", {"selection": None}, False, False),
        ("selection", {"selection": {"kind": "unknown"}}, False, False),
        ("selection", {"selection": {"kind": "ref", "name": "saved"}}, False, False),
        ("selection", {"selection": {"kind": "inline"}}, True, True),
        ("selection", {"selection": {"kind": "empty"}}, True, False),
        ("selection", {"selection": {"kind": "tag-only"}}, True, False),
        ("scalar", {}, False, False),
        ("sequence", {"sequence": [{"value": 1}]}, False, False),
        ("mapping", {"mapping": {"key": {"value": 1}}}, False, False),
    ],
)
def test_expand_capability_matches_parent_and_target_without_changing_draft(
    name: str, data: dict[str, JsonValue], expected: bool, nonempty: bool
) -> None:
    state = FormState(FormInput(CapabilityForm, data))
    env = FormEnv()
    row = next(
        child for child in state.describe((), env).children if child.name == name
    )
    assert row.can_expand is expected
    description = state.describe((name,), env)
    assert description.target is not None
    assert description.target.can_expand is expected
    assert bool(description.children) is nonempty
    assert state.input.data == state.base.data == data


class SourceGuardedChild(FormModel):
    value: float

    @classmethod
    def choices(cls, _env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        raise AssertionError("child choices must not read captured sources yet")

    @classmethod
    def options(cls, _env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        raise AssertionError("child options must not read captured sources yet")

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        del env
        raise AssertionError("description must not request defaults")


class SourceGuardedForm(FormModel):
    child: SourceGuardedChild
    selection: SourceGuardedChild

    @classmethod
    def options(cls, _env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        return {("selection",): (Option("Active", {}, fields=SourceGuardedChild),)}


@pytest.mark.parametrize("data", [{}, {"child": {}, "selection": {}}])
def test_parent_capability_does_not_execute_child_source_hooks(
    data: dict[str, JsonValue],
) -> None:
    state = FormState(FormInput(SourceGuardedForm, data))
    rows = state.describe((), FormEnv()).children
    assert {row.name: row.can_expand for row in rows} == {
        "child": True,
        "selection": "selection" in data,
    }
    assert state.input.data == state.base.data == data


class WrapperShape(FormModel):
    module: Empty


class SelectorShape(WrapperShape):
    @classmethod
    def options(cls, _env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        return {("module",): ()}


class TextShape(FormModel):
    mode: str


class ChoiceShape(TextShape):
    @classmethod
    def choices(cls, _env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        return {("mode",): ("x",)}


class DynamicShapeForm(FormModel):
    wrapper: WrapperShape
    selector: SelectorShape
    text: TextShape
    choice: ChoiceShape

    @classmethod
    def options(cls, _env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        return {
            ("wrapper",): (Option("Empty", {}, fields=WrapperShape),),
            ("selector",): (Option("Selector", {}, fields=SelectorShape),),
            ("text",): (Option("Text", {"mode": "x"}, fields=TextShape),),
            ("choice",): (Option("Choice", {"mode": "x"}, fields=ChoiceShape),),
        }


@pytest.mark.parametrize(
    "name,value,child_names",
    [
        ("wrapper", {}, []),
        ("selector", {}, ["module"]),
        ("text", {"mode": "x"}, ["mode"]),
        ("choice", {"mode": "x"}, []),
    ],
)
def test_expand_capability_preserves_dynamic_wrapper_and_pinned_shape(
    name: str, value: JsonValue, child_names: list[str]
) -> None:
    state = FormState(FormInput(DynamicShapeForm, {name: value}))
    env = FormEnv()
    row = next(
        child for child in state.describe((), env).children if child.name == name
    )
    assert row.can_expand is True
    description = state.describe((name,), env)
    assert description.target is not None
    assert description.target.can_expand is True
    assert [child.name for child in description.children] == child_names
    assert state.input.data == state.base.data == {name: value}
