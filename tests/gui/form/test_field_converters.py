"""Field selection is independent of child drafts; conversions are atomic."""

from operator import setitem
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, create_model
from zcu_tools.gui import form as api


class First(BaseModel):
    kind: Literal["first"] = "first"
    amount: float
    samples: list[int] = []


class Second(BaseModel):
    kind: Literal["second"] = "second"
    amount: float = Field(default=99, gt=0)
    required: float


class Plain(api.FormModel):
    value: First | Second = Field(discriminator="kind")
    unrelated: float = 1


@pytest.mark.parametrize(
    "batch", [{("value",): {"kind": "second"}}, {("value", "kind"): "second"}]
)
def test_selection_without_converter_preserves_branch_and_authored_children(batch):
    env = api.FormEnv()
    raw = {"value": {"kind": "first", "amount": {"expr": "6 / 2"}, "samples": [1]}}
    state = api.FormState(api.FormInput(Plain, raw))
    events = []
    state.subscribe(lambda: events.append(state.describe(("value",), env)))
    state.write(batch, env)
    assert state.input.data == raw, "A selection must not rebuild child defaults"
    description = state.describe(("value",), env)
    assert description.target is not None and description.target.selected == 1
    assert [child.name for child in description.children] == ["amount", "samples"]
    assert len(events) == 1
    state.write({("value", "amount"): -5}, env)
    assert state.input.data["value"] == {"kind": "first", "amount": -5, "samples": [1]}
    detached = api.FormState(state.input)
    assert detached.describe(("value",), env).target == description.target
    state.reset(state.base)
    assert state.describe(("value",), env).target.selected == 0


def converted_model(run):
    assert hasattr(api, "InputConverter"), (
        "Fields need one converter for all selections"
    )
    assert hasattr(api, "FormConversionResult"), (
        "Converters must return named structure/input"
    )
    return create_model(
        "ConvertedForm",
        __base__=Plain,
        value=(
            Annotated[First | Second, api.InputConverter(run)],
            Field(discriminator="kind"),
        ),
    )


def test_one_converter_receives_readonly_snapshot_target_and_local_evaluation():
    calls = []

    def convert(old, selection, context):
        calls.append(selection)
        assert old.structure is First
        assert old.selection == {"kind": "first"}
        with pytest.raises(TypeError):
            setitem(old.input["amount"], "expr", "0")
        with pytest.raises(TypeError):
            setitem(selection, "kind", "first")
        if selection == {"kind": "second"}:
            return api.FormConversionResult(
                structure=Second,
                input={"kind": "second", "amount": -context.evaluate(("amount",))},
            )
        return api.FormConversionResult(structure=First, input=old.copy_input())

    model = converted_model(convert)
    env = api.FormEnv()
    state = api.FormState(
        api.FormInput(
            model,
            {
                "value": {"kind": "first", "amount": {"expr": "6 / 2"}},
                "unrelated": {"expr": "1 / 0"},
            },
        )
    )
    state.describe((), env)
    state.preview(env)
    state.result(env)
    assert calls == [], "Reads must not invoke a converter"
    observed = []
    state.subscribe(
        lambda: observed.append((state.input.data, state.describe(("value",), env)))
    )
    state.write({("value",): {"kind": "second"}}, env)
    assert state.input.data["value"] == {"kind": "second", "amount": -3}
    assert len(calls) == len(observed) == 1
    errors = state.result(env)
    assert isinstance(errors, api.FormErrors)
    assert ("unrelated",) in errors.messages
    assert state.input.data["unrelated"] == {"expr": "1 / 0"}
    state.write({("unrelated",): 1}, env)
    errors = state.result(env)
    assert isinstance(errors, api.FormErrors)
    assert set(errors.messages) == {("value", "amount"), ("value", "required")}
    assert state.input.data["value"] == {"kind": "second", "amount": -3}
    assert len(calls) == 1
    assert len(observed) == 2


@pytest.mark.parametrize("defect", ["structure", "scope", "evaluation", "exception"])
def test_converter_failure_keeps_selection_structure_input_and_notifications(defect):
    def convert(old, selection, context):
        if defect == "exception":
            raise ValueError("cannot convert")
        if defect == "evaluation":
            context.evaluate(("amount",))
        return api.FormConversionResult(
            structure=Plain if defect == "structure" else Second,
            input={
                "kind": "second",
                "amount": 1,
                **({"unrelated": 2} if defect == "scope" else {}),
            },
        )

    model = converted_model(convert)
    env = api.FormEnv()
    state = api.FormState(
        api.FormInput(model, {"value": {"kind": "first", "amount": {"expr": "1 / 0"}}})
    )
    before = state.describe(("value",), env)
    raw = state.input.data
    events = []
    state.subscribe(lambda: events.append(state.input.data))
    with pytest.raises(ValueError, match=".+"):
        state.write({("value",): {"kind": "second"}}, env)
    assert state.input.data == raw
    assert state.describe(("value",), env) == before
    assert events == []


def test_converter_handles_every_scalar_selection_and_rejects_duplicate_declarations():
    assert hasattr(api, "InputConverter"), (
        "Fields need one converter for all selections"
    )
    calls = []

    def convert(old, selection, context):
        calls.append((old.copy_input(), selection))
        return api.FormConversionResult(structure=None, input=selection)

    class Scalar(api.FormModel):
        mode: Annotated[Literal["a", "b", "c"], api.InputConverter(convert)] = "a"

    env = api.FormEnv()
    state = api.FormState(Scalar.default_input(env))
    state.write({("mode",): "b"}, env)
    state.write({("mode",): "c"}, env)
    assert calls == [("a", "b"), ("b", "c")]
    duplicate = create_model(
        "Duplicate",
        __base__=api.FormModel,
        mode=(
            Annotated[
                Literal["a", "b"],
                api.InputConverter(convert),
                api.InputConverter(convert),
            ],
            "a",
        ),
    )
    with pytest.raises(ValueError, match="one converter"):
        duplicate.default_input(env)
