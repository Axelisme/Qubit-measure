"""Requested-path errors remain distinct from form declaration failures."""

from collections.abc import Mapping
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, JsonValue
from zcu_tools.gui import form
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    Option,
)


class Child(BaseModel):
    value: float


class First(BaseModel):
    kind: Literal["first"] = "first"
    child: Child


class Second(BaseModel):
    kind: Literal["second"] = "second"
    other: float


class PathForm(FormModel):
    child: Child
    optional: Child | None = None
    selected: Annotated[First | Second, Field(discriminator="kind")]
    scalar: float
    sequence: list[Child]
    mapping: dict[str, Child]


@pytest.mark.parametrize(
    "path",
    [
        ("unknown",),
        ("unknown", "value"),
        ("child", "unknown", "value"),
        ("optional", "value"),
        ("selected", "child", "value"),
        ("scalar", "value"),
        ("sequence", "value"),
        ("mapping", "key", "value"),
    ],
)
def test_describe_path_error_exposes_full_requested_path_and_valueerror_ancestry(
    path: tuple[str, ...],
) -> None:
    state = FormState(FormInput(PathForm, {"selected": {"kind": "second"}}))
    with pytest.raises(ValueError, match="form field path") as failure:
        state.describe(path, FormEnv())
    assert isinstance(failure.value, form.FormPathError)
    assert failure.value.path == path
    assert str(failure.value)
    assert state.input.data == state.base.data == {"selected": {"kind": "second"}}


class InvalidDisplay(FormModel):
    value: Annotated[str, Display(decimals=2)]


class InvalidDeclaration(FormModel):
    value: float

    @classmethod
    def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[Option, ...]]:
        return {("unknown",): ()}


class NonfiniteDeclaration(FormModel):
    value: float

    @classmethod
    def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[Option, ...]]:
        return {("value",): (Option("Invalid", float("inf")),)}


@pytest.mark.parametrize(
    "model,message",
    [
        (InvalidDisplay, "decimals"),
        (InvalidDeclaration, "Unknown option field path"),
        (NonfiniteDeclaration, "finite"),
    ],
)
def test_describe_declaration_failures_are_not_request_path_errors(
    model: type[FormModel], message: str
) -> None:
    data: dict[str, JsonValue] = {}
    state = FormState(FormInput(model, data))
    with pytest.raises(ValueError, match=message) as failure:
        state.describe((), FormEnv())
    assert not isinstance(failure.value, form.FormPathError)
    assert state.input.data == state.base.data == data
