"""Exact subtree replacement is explicit and separate from partial writes."""

from typing import Literal

import pytest
from pydantic import Field, JsonValue
from zcu_tools.gui import form as api


class First(api.FormModel):
    kind: Literal["first"] = "first"
    amount: float


class Second(api.FormModel):
    kind: Literal["second"] = "second"
    amount: float = Field(gt=0)
    extra: float = 10


class ReplacementForm(api.FormModel):
    branch: First | Second = Field(discriminator="kind")
    sibling: float


def replacement_state() -> api.FormState[ReplacementForm]:
    assert hasattr(api.FormState, "prepare_replace"), (
        "Missing exact replacement preparation"
    )
    assert hasattr(api.FormState, "replace"), "Missing explicit subtree replacement"
    return api.FormState(
        api.FormInput(
            ReplacementForm,
            {"branch": {"kind": "first", "amount": {"expr": "1 / 0"}}, "sibling": 7},
            selections={
                ("branch",): {"kind": "second"},
                ("branch", "amount"): 9,
                ("sibling",): 7,
            },
        )
    )


def test_exact_replacement_clears_only_target_selections_without_old_evaluation():
    state = replacement_state()
    before, base = state.input, state.base
    events = []
    state.subscribe(lambda: events.append(state.input))
    value: JsonValue = {"kind": "second", "amount": -2}
    candidate = state.prepare_replace({("branch",): value}, api.FormEnv())
    assert state.input.same_as(before)
    assert events == []
    state.install_input(candidate)
    assert state.input.data == {
        "branch": {"kind": "second", "amount": -2},
        "sibling": 7,
    }
    assert state.input.selections == {("sibling",): 7}
    assert state.base.same_as(base)
    assert len(events) == 1
    errors = state.result(api.FormEnv())
    assert isinstance(errors, api.FormErrors)
    assert ("branch", "amount") in errors.messages
    target = state.describe(("branch",), api.FormEnv()).target
    assert target is not None and target.branch == {"kind": "second"}


def test_partial_selection_still_preserves_old_branch_and_raw_input():
    state = replacement_state()
    state.write({("branch",): {"kind": "second"}}, api.FormEnv())
    assert state.input.data["branch"] == {"kind": "first", "amount": {"expr": "1 / 0"}}
    state.replace({("branch",): {"kind": "second", "amount": 4}}, api.FormEnv())
    assert state.input.data["branch"] == {"kind": "second", "amount": 4}


@pytest.mark.parametrize(
    "value", [{"kind": "other", "amount": 2}, {"kind": "second", "unknown": 3}]
)
def test_annotation_rejection_keeps_draft_base_selections_and_notifications(value):
    state = replacement_state()
    before, base = state.input, state.base
    events = []
    state.subscribe(lambda: events.append(state.input))
    with pytest.raises(ValueError, match="(structure|field)"):
        state.replace({("branch",): value}, api.FormEnv())
    assert state.input.same_as(before)
    assert state.base.same_as(base)
    assert events == []


def test_replacement_runs_owner_admission_on_complete_candidate():
    replacement_state()

    class Owner(ReplacementForm):
        @classmethod
        def check_input_change(
            cls, before: dict[str, JsonValue], after: dict[str, JsonValue]
        ) -> None:
            if before["branch"] != after["branch"]:
                raise api.FormInputRejected(("branch",), "Owner refuses replacement")

    state = api.FormState(
        api.FormInput(Owner, {"branch": {"kind": "first", "amount": 1}, "sibling": 7})
    )
    before = state.input
    events = []
    state.subscribe(lambda: events.append(state.input))
    with pytest.raises(api.FormInputRejected, match="Owner refuses") as caught:
        state.replace(
            {("branch",): {"kind": "second", "amount": 2}, ("sibling",): 8},
            api.FormEnv(),
        )
    assert caught.value.path == ("branch",)
    assert state.input.same_as(before)
    assert state.base.same_as(before)
    assert events == []
