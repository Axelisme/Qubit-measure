"""Generic source hooks and detached transaction preparation at public seams."""

from collections.abc import Mapping
from typing import Annotated

import pytest
from pydantic import JsonValue, create_model
from pydantic_core import PydanticCustomError
from zcu_tools.gui import form


class CapturedName(form.FormModel):
    """A source consumer unrelated to module/library identity."""

    name: str

    @classmethod
    def check_sources(cls, data: Mapping[str, JsonValue], env: form.FormEnv) -> None:
        del env
        if data.get("name") == "missing":
            raise form.field_error(("name",), "Unavailable captured name")


class SourceTree(form.FormModel):
    child: CapturedName
    count: int = 1


def test_source_hook_directs_nested_failures_and_restore_reverts_only_consumer() -> (
    None
):
    env = form.FormEnv()
    state = form.FormState[form.FormModel](
        form.FormInput(SourceTree, {"child": {"name": "ok"}})
    )
    before = state.input.data
    with pytest.raises(PydanticCustomError) as failure:
        state.write({("child", "name"): "missing", ("count",): 2}, env)
    assert failure.value.context is not None and failure.value.context["path"] == (
        "child",
        "name",
    )
    assert state.input.data == before
    invalid = form.FormState(form.FormInput(SourceTree, {"child": {"name": "missing"}}))
    errors = invalid.result(env)
    assert isinstance(errors, form.FormErrors)
    assert ("child", "name") in errors.messages
    assert state.write_compatible({"child": {"name": "missing"}, "count": 4}, env) == (
        ("child", "name"),
    )
    assert state.input.data == {"child": {"name": "ok"}, "count": 4}


def test_prepare_install_is_detached_preserves_base_and_notifies_once() -> None:
    env = form.FormEnv()
    state = form.FormState[form.FormModel](
        form.FormInput(SourceTree, {"child": {"name": "ok"}})
    )
    before = state.base.data
    notices: list[str] = []
    state.subscribe(lambda: notices.append("changed"))
    candidate = state.prepare_write({("count",): -2}, env)
    assert state.input.data == before and notices == []
    state.install_input(candidate)
    state.install_input(candidate)
    assert state.input.data["count"] == -2
    assert state.base.data == before and notices == ["changed"]
    with pytest.raises(TypeError):
        state.install_input(form.FormInput(CapturedName, {"name": "ok"}))


def test_input_policy_disables_expression_across_description_defaults_and_restore() -> (
    None
):
    model = create_model(
        "LiteralText",
        __base__=form.FormModel,
        text=(Annotated[str, form.InputPolicy(expression=False)], {"expr": "1 / 0"}),
    )
    env = form.FormEnv()
    state = form.FormState(model.default_input(env))
    assert state.input.data == {"text": {"expr": "1 / 0"}}
    description = state.describe(("text",), env)
    assert description.target is not None and not description.target.expression
    assert isinstance(state.result(env), form.FormErrors)
    restored = form.FormState(form.FormInput(model, {"text": "$literal"}))
    assert restored.write_compatible(state.input.data, env) == (("text",),)
    assert restored.input.data == {"text": "$literal"}
