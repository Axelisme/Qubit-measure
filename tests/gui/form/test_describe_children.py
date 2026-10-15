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
    KeyRef,
    Option,
)


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
    def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
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
    "name,data,expected",
    [
        ("ordinary", {}, True),
        ("ordinary", {"ordinary": {"value": "invalid"}}, True),
        ("empty", {}, False),
        ("optional", {}, False),
        ("optional", {"optional": None}, False),
        ("optional", {"optional": {}}, True),
        ("hidden", {}, True),
        ("selection", {}, False),
        ("selection", {"selection": None}, False),
        ("selection", {"selection": {"kind": "unknown"}}, False),
        ("selection", {"selection": {"kind": "ref", "name": "saved"}}, False),
        ("selection", {"selection": {"kind": "inline"}}, True),
        ("selection", {"selection": {"kind": "empty"}}, False),
        ("selection", {"selection": {"kind": "tag-only"}}, False),
        ("scalar", {}, False),
        ("sequence", {"sequence": [{"value": 1}]}, False),
        ("mapping", {"mapping": {"key": {"value": 1}}}, False),
    ],
)
def test_child_capability_matches_target_children_without_changing_draft(
    name: str, data: dict[str, JsonValue], expected: bool
) -> None:
    state = FormState(FormInput(CapabilityForm, data))
    env = FormEnv()
    row = next(child for child in state.describe((), env).children if child.name == name)
    assert row.has_children is expected
    description = state.describe((name,), env)
    assert description.target is not None
    assert description.target.has_children is expected
    assert bool(description.children) is expected
    assert state.input.data == state.base.data == data


class SourceGuardedChild(FormModel):
    value: float

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        raise AssertionError("child choices must not read captured sources yet")

    @classmethod
    def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        raise AssertionError("child options must not read captured sources yet")

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        raise AssertionError("description must not request defaults")


class SourceGuardedForm(FormModel):
    child: SourceGuardedChild
    selection: SourceGuardedChild

    @classmethod
    def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        return {("selection",): (Option("Active", {}, fields=SourceGuardedChild),)}


@pytest.mark.parametrize("data", [{}, {"child": {}, "selection": {}}])
def test_parent_capability_does_not_execute_child_source_hooks(
    data: dict[str, JsonValue],
) -> None:
    state = FormState(FormInput(SourceGuardedForm, data))
    rows = state.describe((), FormEnv()).children
    assert {row.name: row.has_children for row in rows} == {
        "child": True,
        "selection": "selection" in data,
    }
    assert state.input.data == state.base.data == data
