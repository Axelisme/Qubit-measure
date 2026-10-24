"""Single-layer descriptions at the public state/command seam, without sources."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, JsonValue
from zcu_tools.gui.form import (
    Display,
    FormCommandContext,
    FormCommandResult,
    FormCommandSnapshot,
    FormConstraints,
    FormEnv,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    Option,
    OptionCommand,
)
from zcu_tools.gui.form_adapter import KeyRef, LinRange


class Speed(StrEnum):
    slow = "slow"
    fast = "fast"


class Settings(BaseModel):
    frequency: Annotated[
        float,
        Display(label="Frequency", unit="MHz", decimals=3, placeholder="Enter Hz"),
        Field(gt=0, le=100),
    ]
    label: Annotated[str, Field(min_length=2, max_length=8, pattern="^[a-z]+$")]


type BoundedNumber = Annotated[float, Field(ge=1, lt=10)]
type BoundedText = Annotated[str, Field(min_length=2, max_length=8, pattern="^[a-z]+$")]


class NullableConstraints(FormModel):
    number: Annotated[float, Field(gt=0, le=100)] | None = Field(
        default=None, lt=80, title="Optional number"
    )
    number_alias: BoundedNumber | None = Field(default=None, le=8)
    text: (
        Annotated[str, Field(min_length=2, max_length=8, pattern="^[a-z]+$")] | None
    ) = None
    text_alias: BoundedText | None = None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("number", FormConstraints(gt=0, le=100, lt=80)),
        ("number_alias", FormConstraints(ge=1, le=8, lt=10)),
        (
            "text",
            FormConstraints(min_length=2, max_length=8, pattern="^[a-z]+$"),
        ),
        (
            "text_alias",
            FormConstraints(min_length=2, max_length=8, pattern="^[a-z]+$"),
        ),
    ],
)
def test_nullable_inner_constraints_are_described(
    name: str, expected: FormConstraints
) -> None:
    state = FormState(FormInput(NullableConstraints, {}))
    target = state.describe((name,), FormEnv()).target
    assert target is not None
    assert target.constraints == expected
    assert target.nullable
    assert state.input.data == state.base.data == {}


class Inline(FormModel):
    kind: Literal["inline"] = "inline"
    module: Settings


class NullableInline(FormModel):
    kind: Literal["nullable"] = "nullable"
    module: Settings | None = None


class SelectableInline(FormModel):
    kind: Literal["selectable"] = "selectable"
    module: Settings

    @classmethod
    def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[Option, ...]]:
        return {("module",): ()}


class DescribeForm(FormModel):
    settings: Annotated[Settings, Display(group="Advanced")]
    optional: Settings | None = None
    selection: Annotated[
        KeyRef | Inline | NullableInline | SelectableInline | None,
        Field(discriminator="kind"),
    ] = None
    samples: tuple[float, ...] = ()
    mapping: dict[str, float] = Field(default_factory=dict)
    speed: Speed | None = None
    count: Literal[1, 2] = 1
    flag: bool = False
    amplitude: complex = 1j
    secret: Annotated[str, Display(hidden=True, description="Internal text")] = "x"
    grid: LinRange

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        del env
        raise AssertionError("description must not request defaults")

    @classmethod
    def options(cls, _env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        def convert(
            _old: FormCommandSnapshot, _context: FormCommandContext
        ) -> FormCommandResult:
            return FormCommandResult(
                {"kind": "inline"}, Inline, {"module": {"frequency": 4, "label": "new"}}
            )

        return {
            ("selection",): (
                Option("None", None),
                Option(
                    "Saved",
                    {"kind": "ref", "name": "saved"},
                    commands=(OptionCommand("convert", "Convert", convert),),
                ),
                Option("Inline", {"kind": "inline"}, fields=Inline),
                Option("Nullable", {"kind": "nullable"}, fields=NullableInline),
                Option("Selectable", {"kind": "selectable"}, fields=SelectableInline),
            )
        }


def test_root_model_leaf_are_single_layer_and_keep_display_constraints() -> None:
    assert hasattr(FormState, "describe"), "FormState must offer single-layer describe"
    env = FormEnv()
    state = FormState(FormInput(DescribeForm, {}))
    root = state.describe((), env)
    assert root.path == () and root.target is None
    assert [field.path for field in root.children][:3] == [
        ("settings",),
        ("optional",),
        ("selection",),
    ]
    settings = root.children[0]
    assert settings.kind == "model" and settings.title == "Settings"
    assert settings.group == "Advanced"
    model = state.describe(("settings",), env)
    assert [field.path for field in model.children] == [
        ("settings", "frequency"),
        ("settings", "label"),
    ]
    leaf = state.describe(("settings", "frequency"), env)
    assert leaf.children == ()
    assert leaf.target is not None
    assert leaf.target.kind == "number" and leaf.target.expression
    assert leaf.target.title == "Frequency" and leaf.target.unit == "MHz"
    assert leaf.target.decimals == 3 and leaf.target.placeholder == "Enter Hz"
    assert leaf.target.constraints.gt == 0 and leaf.target.constraints.le == 100
    label = state.describe(("settings", "label"), env).target
    assert label is not None
    assert label.constraints.min_length == 2 and label.constraints.max_length == 8
    assert label.constraints.pattern == "^[a-z]+$"
    assert state.input.data == state.base.data == {}


@pytest.mark.parametrize(
    ("name", "kind", "expression"),
    [
        ("samples", "sequence", False),
        ("mapping", "mapping", False),
        ("flag", "boolean", False),
        ("amplitude", "complex", True),
    ],
)
def test_whole_leaves_describe_without_opening_container_contents(
    name: str, kind: str, expression: bool
) -> None:
    assert hasattr(FormState, "describe"), "whole leaves need field metadata"
    state = FormState(FormInput(DescribeForm, {}))
    leaf = state.describe((name,), FormEnv())
    assert leaf.target is not None
    assert leaf.target.kind == kind and leaf.target.expression is expression
    assert leaf.children == ()


def test_literal_enum_fixed_and_hidden_metadata_do_not_change_input() -> None:
    assert hasattr(FormState, "describe"), "picked and fixed leaves need metadata"
    state = FormState(FormInput(DescribeForm, {}))
    env = FormEnv()
    count = state.describe(("count",), env).target
    speed = state.describe(("speed",), env).target
    secret = state.describe(("secret",), env).target
    tag = state.describe(("grid", "kind"), env).target
    assert (
        count is not None
        and speed is not None
        and secret is not None
        and tag is not None
    )
    assert count.options is not None and [item.value for item in count.options] == [
        1,
        2,
    ]
    assert speed.options is not None
    assert [item.value for item in speed.options] == [None, "slow", "fast"]
    assert speed.nullable and not speed.expression and speed.selected is None
    assert secret.hidden and secret.description == "Internal text"
    assert tag.fixed and tag.fixed_value == "linrange" and tag.hidden
    assert state.input.data == {}


@pytest.mark.parametrize(
    ("value", "paths"),
    [
        (
            {"kind": "inline", "module": {"frequency": -2}},
            [("selection", "module", "frequency"), ("selection", "module", "label")],
        ),
        ({"kind": "nullable", "module": None}, [("selection", "module")]),
        ({"kind": "selectable", "module": {}}, [("selection", "module")]),
        ({"kind": "ref", "name": "saved"}, []),
        ({"kind": "missing"}, []),
        (None, []),
    ],
)
def test_selected_fields_flatten_only_one_ordinary_wrapper(
    value: JsonValue, paths: list[tuple[str, ...]]
) -> None:
    assert hasattr(FormState, "describe"), (
        "selected fields need lazy full-path projection"
    )
    state = FormState(FormInput(DescribeForm, {"selection": value}))
    description = state.describe(("selection",), FormEnv())
    assert [child.path for child in description.children] == paths
    assert description.target is not None and description.target.options is not None
    assert [item.label for item in description.target.options] == [
        "None",
        "Saved",
        "Inline",
        "Nullable",
        "Selectable",
    ]
    assert description.target.selected == (
        next(
            (
                index
                for index, tag in enumerate(
                    ("ref", "inline", "nullable", "selectable"), 1
                )
                if tag == value.get("kind")
            ),
            None,
        )
        if isinstance(value, dict)
        else 0
    )
    assert state.input.data == state.base.data == {"selection": value}


@pytest.mark.parametrize("value", [None, 42, {"kind": "unknown"}])
def test_invalid_or_disabled_target_is_readable_but_descendant_is_not(
    value: JsonValue,
) -> None:
    assert hasattr(FormState, "describe"), "invalid drafts must remain describable"
    state = FormState(FormInput(DescribeForm, {"selection": value, "optional": None}))
    assert state.describe(("selection",), FormEnv()).target is not None
    assert state.describe(("selection",), FormEnv()).children == ()
    for path in (("selection", "module"), ("optional", "frequency")):
        with pytest.raises(ValueError, match="selection|optional"):
            state.describe(path, FormEnv())


@pytest.mark.parametrize(
    "path", [("unknown",), ("samples", "0"), ("mapping", "key"), ("count", "child")]
)
def test_unknown_and_whole_leaf_descendants_fail_with_path(
    path: tuple[str, ...],
) -> None:
    assert hasattr(FormState, "describe"), "invalid paths must fail at describe entry"
    state = FormState(FormInput(DescribeForm, {}))
    with pytest.raises(ValueError, match=path[0]):
        state.describe(path, FormEnv())


class Local(FormModel):
    mode: str
    parent: int

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        return {("mode",): ()}


class Reused(FormModel):
    left: Local
    right: Local

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        return {("left", "mode"): ("outer",), ("left", "parent"): (2,)}

    @classmethod
    def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[Option, ...]]:
        return {("left", "parent"): (Option("First", 2), Option("Second", 2))}


def test_local_override_choices_field_isolation_first_match_and_detachment() -> None:
    assert hasattr(FormState, "describe"), "description must own effective choices"
    state = FormState(FormInput(Reused, {"left": {"parent": 2}, "right": {}}))
    env = FormEnv()
    left = state.describe(("left",), env)
    right = state.describe(("right",), env)
    assert left.children[0].choices == right.children[0].choices == ()
    assert not left.children[0].expression
    assert left.children[1].choices == (2,) and left.children[1].selected == 0
    assert right.children[1].choices is None and right.children[1].expression
    selection = FormState(
        FormInput(DescribeForm, {"selection": {"kind": "ref", "name": "saved"}})
    )
    target = selection.describe(("selection",), env).target
    assert target is not None and target.options is not None
    option = target.options[1]
    assert [(command.name, command.label) for command in option.commands] == [
        ("convert", "Convert")
    ]
    assert isinstance(option.value, dict)
    option.value["name"] = "changed"
    again = selection.describe(("selection",), env).target
    assert again is not None and again.options is not None
    assert again.options[1].value == {"kind": "ref", "name": "saved"}
    events: list[dict[str, JsonValue]] = []
    selection.subscribe(lambda: events.append(selection.input.data))
    selection.run_command(("selection",), "convert", env)
    assert selection.input.data == {
        "selection": {"kind": "inline", "module": {"frequency": 4, "label": "new"}}
    }
    assert events == [selection.input.data]
    assert selection.describe(("selection", "module"), env).children[0].path == (
        "selection",
        "module",
        "frequency",
    )
    with pytest.raises(ValueError, match="command"):
        selection.run_command(("selection",), "convert", env)
    assert events == [selection.input.data]
