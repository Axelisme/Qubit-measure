"""Authored form identity includes detached selections, not just value leaves."""

from typing import Literal

from pydantic import BaseModel, Field, JsonValue
from zcu_tools.gui.form import FormEnv, FormInput, FormModel, FormState


class First(BaseModel):
    kind: Literal["first"] = "first"
    amount: float = 2


class Second(BaseModel):
    kind: Literal["second"] = "second"
    required: float


class Choice(FormModel):
    branch: First | Second = Field(discriminator="kind")


def test_selection_only_identity_dirty_accept_discard_and_noop() -> None:
    raw: dict[str, JsonValue] = {"branch": {"kind": "first", "amount": 2}}
    env = FormEnv()
    base = FormInput(Choice, raw)
    state = FormState(base)
    assert hasattr(state, "is_dirty"), "Dirty observation must include selections"
    assert hasattr(base, "same_as"), "Form input needs one authored equality contract"
    events: list[bool] = []
    state.subscribe(lambda: events.append(state.is_dirty))
    assert not state.is_dirty
    assert base.same_as(FormInput(Choice, raw))
    state.select(("branch",), {"kind": "first"}, env)
    assert events == []
    state.select(("branch",), {"kind": "second"}, env)
    assert state.is_dirty
    assert state.input.data == state.base.data == raw
    assert state.changes(env) == (), "Value-leaf changes are not a dirty predicate"
    assert not state.input.same_as(base)
    assert state.input.same_as(
        FormInput(Choice, raw, selections=state.input.selections)
    )
    description = state.describe(("branch",), env)
    assert description.target is not None
    assert description.target.branch == {"kind": "first"}
    accepted = state.input
    state.reset(accepted)
    assert not state.is_dirty
    state.select(("branch",), {"kind": "first"}, env)
    assert state.is_dirty
    state.reset(state.base)
    assert not state.is_dirty
    restored = state.describe(("branch",), env).target
    assert restored is not None and restored.selected == 1
    state.reset(state.base)
    state.select(("branch",), {"kind": "second"}, env)
    assert events == [True, False, True, False]
    assert state.input.data == raw

    class Other(Choice):
        pass

    assert not base.same_as(FormInput(Other, raw))
    assert not base.same_as(FormInput(Choice, {"branch": {"kind": "first"}}))
