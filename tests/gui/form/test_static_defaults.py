"""Public static input defaults preserve omission and detached JSON values."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field
from zcu_tools.gui import form as form_api
from zcu_tools.gui.form import FormEnv, FormState


class StaticChild(BaseModel):
    """Nullable defaults remain absent, without changing validated values."""

    kind: Literal["child"] = "child"
    optional: int | None = None
    values: list[int] = Field(default_factory=lambda: [2])


class OtherChild(BaseModel):
    """A distinct ordinary branch triggers static input defaults."""

    kind: Literal["other"] = "other"


class StaticForm(form_api.FormModel):
    """Ordinary tagged model branches use the same static defaults."""

    child: Annotated[OtherChild | StaticChild, Field(discriminator="kind")] = OtherChild()


def test_public_static_defaults_omit_none_and_return_detached_values() -> None:
    assert hasattr(form_api, "static_defaults"), "Missing public static_defaults"
    first = form_api.static_defaults(StaticChild)
    assert first == {"kind": "child", "values": [2]}
    first["values"] = [9]
    assert form_api.static_defaults(StaticChild) == {"kind": "child", "values": [2]}
    assert StaticChild.model_validate(form_api.static_defaults(StaticChild)) == StaticChild()


def test_enabling_ordinary_model_keeps_none_absent_and_validated_value_unchanged() -> None:
    assert hasattr(form_api, "static_defaults"), "Missing public static_defaults"
    env = FormEnv()
    state = FormState(StaticForm.default_input(env))
    state.write({("child", "kind"): "child"}, env)
    assert state.input.data == {"child": {"kind": "child", "values": [2]}}
    assert state.result(env) == StaticForm(child=StaticChild())
