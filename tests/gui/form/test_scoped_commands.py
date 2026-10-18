"""Commands install one annotation-admitted subtree from readonly draft input."""

from __future__ import annotations

from collections.abc import Mapping
from operator import setitem
from typing import Literal

import pytest
from pydantic import BaseModel, Field
from zcu_tools.gui import form as api


class Original(BaseModel):
    kind: Literal["old"] = "old"
    amount: float
    samples: list[int] = []


class Converted(BaseModel):
    kind: Literal["new"] = "new"
    amount: float = Field(gt=0)


class Foreign(BaseModel):
    kind: Literal["foreign"] = "foreign"


def conversion(*args) -> api.FormCommandResult:
    assert len(args) == 2, (
        "Commands need a readonly snapshot and local evaluate context"
    )
    old, context = args
    assert old.selection == {"kind": "old"}
    assert old.structure is Original
    assert old.input["amount"] == {"expr": "6 / 2"}
    with pytest.raises(TypeError):
        setitem(old.input["amount"], "expr", "0")
    with pytest.raises(TypeError):
        setitem(old.input["samples"], 0, 99)
    assert context.evaluate(("amount",)) == 3
    return api.FormCommandResult(
        selection={"kind": "new"}, structure=Converted, input={"amount": -3}
    )


class CommandForm(api.FormModel):
    value: Original | Converted = Field(discriminator="kind")
    other: float

    @classmethod
    def options(
        cls, _env: api.FormEnv
    ) -> Mapping[api.FormPath, tuple[api.Option, ...]]:
        return {
            ("value",): (
                api.Option(
                    "Old",
                    {"kind": "old"},
                    Original,
                    (api.OptionCommand("convert", "Convert", conversion),),
                ),
                api.Option("New", {"kind": "new"}, Converted),
            )
        }


def test_command_reads_only_participants_and_publishes_complete_invalid_draft():
    env = api.FormEnv()
    state = api.FormState(
        api.FormInput(
            CommandForm,
            {
                "value": {"kind": "old", "amount": {"expr": "6 / 2"}, "samples": [1]},
                "other": {"expr": "1 / 0"},
            },
        )
    )
    events = []

    def published():
        target = state.describe(("value",), env).target
        assert target is not None
        events.append((state.input.data, target.selected))

    state.subscribe(published)
    state.run_command(("value",), "convert", env)
    assert state.input.data == {
        "value": {"kind": "new", "amount": -3},
        "other": {"expr": "1 / 0"},
    }
    assert events == [(state.input.data, 1)]
    errors = state.result(env)
    assert isinstance(errors, api.FormErrors)
    assert ("other",) in errors.messages
    state.write({("other",): 1}, env)
    errors = state.result(env)
    assert isinstance(errors, api.FormErrors)
    assert set(errors.messages) == {("value", "amount")}
    assert state.input.data["value"] == {"kind": "new", "amount": -3}
    assert len(events) == 2


@pytest.mark.parametrize(
    "defect", ["foreign", "sibling", "nested", "selection", "evaluation"]
)
def test_command_rejects_bad_structure_scope_and_required_evaluation_atomically(defect):
    def rejected(*args):
        assert len(args) == 2, "Commands need the scoped callable contract"
        _, context = args
        if defect == "evaluation":
            context.evaluate(("amount",))
        return api.FormCommandResult(
            selection={"kind": "foreign" if defect == "selection" else "new"},
            structure=Foreign if defect == "foreign" else Converted,
            input={
                "amount": 1,
                **({"other": 9} if defect == "sibling" else {}),
                **({"amount": {"child": 3}} if defect == "nested" else {}),
            },
        )

    class Rejected(api.FormModel):
        value: Original | Converted = Field(discriminator="kind")
        other: float = 7

        @classmethod
        def options(
            cls, _env: api.FormEnv
        ) -> Mapping[api.FormPath, tuple[api.Option, ...]]:
            return {
                ("value",): (
                    api.Option(
                        "Old",
                        {"kind": "old"},
                        Original,
                        (api.OptionCommand("reject", "Reject", rejected),),
                    ),
                    api.Option("New", {"kind": "new"}, Converted),
                )
            }

    env = api.FormEnv()
    state = api.FormState(
        api.FormInput(Rejected, {"value": {"kind": "old", "amount": {"expr": "1 / 0"}}})
    )
    before = state.input.to_json()
    events = []
    state.subscribe(lambda: events.append(state.input.data))
    with pytest.raises(api.FormCommandError):
        state.run_command(("value",), "reject", env)
    assert state.input.to_json() == before
    assert events == []


def test_command_cannot_remove_nonnullable_model_structure():
    class Required(api.FormModel):
        value: Original | Converted = Field(discriminator="kind")

        @classmethod
        def options(
            cls, _env: api.FormEnv
        ) -> Mapping[api.FormPath, tuple[api.Option, ...]]:
            return {
                ("value",): (
                    api.Option(
                        "Old",
                        {"kind": "old"},
                        Original,
                        (
                            api.OptionCommand(
                                "remove",
                                "Remove",
                                lambda _old, _context: api.FormCommandResult(
                                    None, None, None
                                ),
                            ),
                        ),
                    ),
                    api.Option("New", {"kind": "new"}, Converted),
                )
            }

    env = api.FormEnv()
    state = api.FormState(
        api.FormInput(Required, {"value": {"kind": "old", "amount": 3, "samples": [1]}})
    )
    before = state.input.to_json()
    notifications = []
    state.subscribe(lambda: notifications.append(state.input.data))
    with pytest.raises(api.FormCommandError):
        state.run_command(("value",), "remove", env)
    assert state.input.to_json() == before
    target = state.describe(("value",), env).target
    assert target is not None
    assert target.selected == 0
    assert notifications == []
