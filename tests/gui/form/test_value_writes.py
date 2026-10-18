"""Value writes replace raw subtrees; only explicit selections run converters."""

from typing import Annotated, Literal

from pydantic import Field
from zcu_tools.gui import form as api


class First(api.FormModel):
    kind: Literal["first"] = "first"
    amount: float


class Second(api.FormModel):
    kind: Literal["second"] = "second"
    amount: float = Field(gt=0)
    extra: float = 10


class ValueForm(api.FormModel):
    branch: First | Second = Field(discriminator="kind")
    sibling: float

    @classmethod
    def options(cls, _env: api.FormEnv) -> dict[api.FormPath, tuple[api.Option, ...]]:
        return {("sibling",): (api.Option("Seven", 7),)}


def test_value_write_replaces_raw_and_clears_only_target_selection_overrides():
    state = api.FormState(
        api.FormInput(
            ValueForm,
            {"branch": {"kind": "first", "amount": {"expr": "1 / 0"}}, "sibling": 7},
            selections={
                ("branch",): {"kind": "second"},
                ("branch", "amount"): 9,
                ("sibling",): 7,
            },
        )
    )
    before, base = state.input, state.base
    events = []
    state.subscribe(lambda: events.append(state.input))
    candidate = state.prepare_write(
        {("branch",): {"kind": "second", "amount": -2}}, api.FormEnv()
    )
    assert candidate.data["branch"] == {"kind": "second", "amount": -2}
    assert candidate.selections == {("sibling",): 7}
    assert state.input.same_as(before)
    assert events == []
    state.install_input(candidate)
    assert state.input.data["sibling"] == 7
    assert state.base.same_as(base)
    assert len(events) == 1
    errors = state.result(api.FormEnv())
    assert isinstance(errors, api.FormErrors)
    assert ("branch", "amount") in errors.messages


def test_explicit_selection_without_converter_preserves_branch_and_authored_input():
    assert hasattr(api.FormState, "select"), "Missing explicit selection operation"
    assert hasattr(api.FormState, "prepare_selection"), (
        "Missing detached selection preparation"
    )
    state = api.FormState(
        api.FormInput(
            ValueForm,
            {"branch": {"kind": "first", "amount": {"expr": "1 / 0"}}, "sibling": 7},
        )
    )
    before = state.input
    candidate = state.prepare_selection(("branch",), {"kind": "second"}, api.FormEnv())
    assert state.input.same_as(before)
    assert candidate.data == before.data
    state.select(("branch",), {"kind": "second"}, api.FormEnv())
    assert state.input.data == before.data
    assert state.input.selections == {("branch",): {"kind": "second"}}
    state.write({("branch",): {"kind": "second", "amount": 4}}, api.FormEnv())
    assert state.input.data["branch"] == {"kind": "second", "amount": 4}
    assert state.input.selections == {}


def test_value_writes_do_not_execute_converter_but_selection_does():
    assert hasattr(api.FormState, "select"), "Missing explicit selection operation"
    calls = []

    def convert(_old, _new, _context):
        calls.append("convert")
        return api.FormConversionResult(Second, {"kind": "second", "amount": 8})

    class Converted(ValueForm):
        branch: Annotated[
            First | Second, Field(discriminator="kind"), api.InputConverter(convert)
        ]

    state = api.FormState(
        api.FormInput(
            Converted, {"branch": {"kind": "first", "amount": 1}, "sibling": 7}
        )
    )
    state.write({("branch",): {"kind": "second", "amount": 3}}, api.FormEnv())
    assert calls == []
    assert state.input.data["branch"] == {"kind": "second", "amount": 3}
    state.select(("branch",), {"kind": "second"}, api.FormEnv())
    assert calls == ["convert"]
    assert state.input.data["branch"] == {"kind": "second", "amount": 8}
