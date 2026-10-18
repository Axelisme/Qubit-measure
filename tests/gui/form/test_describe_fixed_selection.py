"""Declaration-fixed branches stay distinct from fixed complete option values."""

from collections.abc import Mapping
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, JsonValue
from zcu_tools.gui.form import (
    FormEnv,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    Option,
)


class First(BaseModel):
    kind: Literal["first"] = "first"
    value: float


class Second(BaseModel):
    kind: Literal["second"] = "second"
    value: float


class SingleInline(BaseModel):
    kind: Literal["inline"] = "inline"
    module: First


class MultiInline(BaseModel):
    kind: Literal["inline"] = "inline"
    module: Annotated[First | Second, Field(discriminator="kind")]


class Reference(BaseModel):
    kind: Literal["ref"] = "ref"
    name: str


class SelectionForm(FormModel):
    single: SingleInline
    multi: MultiInline
    selectable: Annotated[Reference | SingleInline, Field(discriminator="kind")]
    optional: SingleInline | None = None
    nullable_multi: MultiInline | None = None
    nullable_selectable: Annotated[
        Reference | SingleInline | None, Field(discriminator="kind")
    ] = None
    number: float
    literal: Literal["only"] = "only"
    optional_literal: Literal["only"] | None = None

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        return {("number",): (1.0,)}

    @classmethod
    def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[Option, ...]]:
        first: JsonValue = {"kind": "inline", "module": {"kind": "first"}}
        second: JsonValue = {"kind": "inline", "module": {"kind": "second"}}
        return {
            ("single",): (Option("Inline", first, fields=SingleInline),),
            ("multi",): (
                Option("First", first, fields=MultiInline),
                Option("Second", second, fields=MultiInline),
            ),
            ("selectable",): (Option("Inline", first, fields=SingleInline),),
            ("optional",): (Option("Inline", first, fields=SingleInline),),
            ("nullable_multi",): (
                Option("None", None),
                Option("First", first, fields=MultiInline),
                Option("Second", second, fields=MultiInline),
            ),
            ("nullable_selectable",): (Option("Inline", first, fields=SingleInline),),
        }


@pytest.mark.parametrize(
    "name,selection_fixed,fixed,fixed_value",
    [
        ("single", True, True, {"kind": "inline", "module": {"kind": "first"}}),
        ("multi", True, False, None),
        ("selectable", False, False, None),
        ("optional", True, False, None),
        ("nullable_selectable", False, False, None),
        ("number", False, False, None),
        ("literal", True, True, "only"),
        ("optional_literal", True, False, None),
    ],
)
@pytest.mark.parametrize("present", [False, True])
def test_fixed_branch_and_complete_option_value_depend_on_annotation_not_menu_size(
    name: str,
    selection_fixed: bool,
    fixed: bool,
    fixed_value: JsonValue,
    present: bool,
) -> None:
    authored: dict[str, JsonValue] = {
        "single": {"kind": "inline", "module": {"kind": "first", "value": 2}},
        "multi": {"kind": "inline", "module": {"kind": "second", "value": 3}},
        "selectable": {"kind": "inline", "module": {"kind": "first", "value": 4}},
        "optional": {"kind": "inline", "module": {"kind": "first", "value": 5}},
        "nullable_selectable": {
            "kind": "inline",
            "module": {"kind": "first", "value": 6},
        },
        "number": 1.0,
        "literal": "only",
        "optional_literal": "only",
    }
    data = authored if present else {}
    state = FormState(FormInput(SelectionForm, data))
    env = FormEnv()
    root_row = next(row for row in state.describe((), env).children if row.name == name)
    assert root_row.selection_fixed is selection_fixed
    assert root_row.fixed is fixed
    assert root_row.fixed_value == fixed_value
    description = state.describe((name,), env)
    assert description.target is not None
    assert description.target.selection_fixed is selection_fixed
    assert description.target.fixed is fixed
    assert description.target.fixed_value == fixed_value
    if present and name == "single":
        assert [child.path for child in description.children] == [
            ("single", "module", "value")
        ]
    if present and name == "multi":
        assert [child.name for child in description.children] == ["module"]
        assert description.children[0].selection_fixed is False
        assert description.children[0].fixed is False
    assert state.input.data == state.base.data == data


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
def test_nullable_single_branch_keeps_inner_selector_editable(enabled: bool) -> None:
    current: JsonValue = (
        {"kind": "inline", "module": {"kind": "first", "value": 2}} if enabled else None
    )
    state = FormState(FormInput(SelectionForm, {"nullable_multi": current}))
    env = FormEnv()
    root_row = next(
        row for row in state.describe((), env).children if row.name == "nullable_multi"
    )
    description = state.describe(("nullable_multi",), env)
    assert description.target == root_row
    assert root_row.nullable is True
    assert root_row.selection_fixed is True
    assert root_row.fixed is False
    assert root_row.fixed_value is None
    assert root_row.can_expand is enabled
    assert state.input.data == state.base.data == {"nullable_multi": current}

    if not enabled:
        assert description.children == ()
        state.write(
            {("nullable_multi",): {"kind": "inline", "module": {"kind": "first"}}}, env
        )
        assert state.describe(("nullable_multi",), env).children == ()
        state.install_input(
            FormInput(
                SelectionForm,
                {
                    "nullable_multi": {
                        "kind": "inline",
                        "module": {"kind": "first", "value": 2},
                    }
                },
            )
        )
    inner = state.describe(("nullable_multi", "module"), env).target
    assert inner is not None
    assert inner.selection_fixed is False
    assert inner.fixed is False
    assert inner.options is not None
    assert [option.value for option in inner.options] == [
        {"kind": "first"},
        {"kind": "second"},
    ]
    state.write({("nullable_multi", "module", "kind"): "second"}, env)
    state.write({("nullable_multi", "module", "value"): 7.0}, env)
    switched = state.describe(("nullable_multi", "module"), env).target
    assert switched is not None and switched.selected == 1
    assert state.preview(env).values[("nullable_multi", "module", "value")] == 7.0

    state.write({("nullable_multi",): None}, env)
    disabled = state.describe(("nullable_multi",), env)
    assert disabled.target is not None
    assert disabled.target.selection_fixed is True
    assert disabled.target.nullable is True
    assert disabled.target.fixed is False
    assert disabled.children
    assert state.input.data["nullable_multi"] == {
        "kind": "inline",
        "module": {"kind": "first", "value": 7.0},
    }
