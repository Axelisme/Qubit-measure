"""Public static input defaults preserve omission and detached JSON values."""

from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, JsonValue
from zcu_tools.gui import form as form_api
from zcu_tools.gui.form import FormEnv, FormModel, FormState, InputDefault

from tests.gui._form_converters import RebuildingForm


class StaticChild(BaseModel):
    """Nullable defaults remain absent, without changing validated values."""

    kind: Literal["child"] = "child"
    optional: int | None = None
    values: list[int] = Field(default_factory=lambda: [2])


class OtherChild(BaseModel):
    """A distinct ordinary branch triggers static input defaults."""

    kind: Literal["other"] = "other"


class StaticForm(RebuildingForm):
    """Ordinary tagged model branches use the same static defaults."""

    branch_models = {("child",): (OtherChild, StaticChild)}

    child: Annotated[OtherChild | StaticChild, Field(discriminator="kind")] = (
        OtherChild()
    )


def test_public_static_defaults_omit_none_and_return_detached_values() -> None:
    assert hasattr(form_api, "static_defaults"), "Missing public static_defaults"
    first = form_api.static_defaults(StaticChild)
    assert first == {"kind": "child", "values": [2]}
    first["values"] = [9]
    assert form_api.static_defaults(StaticChild) == {"kind": "child", "values": [2]}
    assert (
        StaticChild.model_validate(form_api.static_defaults(StaticChild))
        == StaticChild()
    )


def test_input_default_metadata_is_detached_and_defers_to_declared_defaults() -> None:
    source: dict[str, JsonValue] = {"items": [1]}
    calls: list[str] = []

    def fresh() -> JsonValue:
        calls.append("fresh")
        return source

    def outer() -> JsonValue:
        calls.append("outer")
        return {"items": [2]}

    class MetadataForm(FormModel):
        type FreshValue = Annotated[dict[str, list[int]], InputDefault(fresh)]

        missing: FreshValue
        overridden: Annotated[FreshValue, InputDefault(outer)]
        explicit: FreshValue = {"items": [3]}
        factory: FreshValue = Field(default_factory=lambda: {"items": [4]})
        optional: Annotated[FreshValue | None, InputDefault(fresh)] = None

    result = form_api.static_defaults(MetadataForm)
    assert result == {
        "missing": {"items": [1]},
        "overridden": {"items": [2]},
        "explicit": {"items": [3]},
        "factory": {"items": [4]},
    }
    assert calls == ["fresh", "outer"]
    assert result["missing"] is not source
    source["items"] = [9]
    assert result["missing"] == {"items": [1]}
    assert form_api.static_defaults(MetadataForm)["missing"] == {"items": [9]}


def test_input_default_factory_failure_propagates() -> None:
    def unavailable() -> JsonValue:
        raise ValueError("fresh input unavailable")

    class MetadataForm(FormModel):
        value: Annotated[int, InputDefault(unavailable)]

    with pytest.raises(ValueError, match="fresh input unavailable"):
        form_api.static_defaults(MetadataForm)


def test_enabling_ordinary_model_keeps_none_absent_and_validated_value_unchanged() -> (
    None
):
    assert hasattr(form_api, "static_defaults"), "Missing public static_defaults"
    env = FormEnv()
    state = FormState(StaticForm.default_input(env))
    state.write({("child", "kind"): "child"}, env)
    assert state.input.data == {"child": {"kind": "child"}}
    assert state.result(env) == StaticForm(child=StaticChild())
