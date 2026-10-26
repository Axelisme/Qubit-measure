"""String input envelopes resolve names without changing authored input."""

from collections.abc import Mapping
from typing import Annotated, Literal

import pytest
from pydantic import Field, JsonValue, create_model
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormState,
)
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class NameSection(FormModel):
    """Nested ordinary strings use the same input contract."""

    name: str


class NamesForm(FormModel):
    """Names also occur inside whole containers and nullable sections."""

    name: str = Field(default="literal", min_length=2)
    optional: str | None = None
    names: tuple[str, ...] = ()
    mapping: dict[str, str] = Field(default_factory=dict)
    section: NameSection | None = None


@pytest.fixture
def name_env() -> FormEnv:
    """Capture authored aliases without requiring devices or live sources."""
    context = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="2.0",
            schema_version=1,
            general=PointGeneral(created_at="2026-10-11T00:00:00Z"),
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
        ),
        target=SavedPoint(entry_id="entry", point="synthetic"),
        roles=None,
        storage=None,
    ).bind_aliases((AliasRow("device", "yoko1"), AliasRow("unbound", "")), {})
    return FormEnv(parameters=context)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("name", {"expr": "$device"}, "yoko1"),
        ("optional", {"expr": "$device"}, "yoko1"),
        ("names", [{"expr": "$device"}, "$device"], ("yoko1", "$device")),
        ("mapping", {"rack.left": {"expr": "$device"}}, {"rack.left": "yoko1"}),
        ("section", {"name": {"expr": "$device"}}, NameSection(name="yoko1")),
    ],
)
def test_string_envelopes_share_preview_result_and_restore(
    name_env: FormEnv, field: str, value: JsonValue, expected: object
) -> None:
    state = FormState(NamesForm.default_input(name_env))
    authored: dict[str, JsonValue] = {field: value}
    state.write({(field,): value}, name_env)
    result = state.result(name_env)
    assert isinstance(result, NamesForm), result
    assert getattr(result, field) == expected
    preview = state.preview(name_env)
    assert preview.errors is None
    path = ("section", "name") if field == "section" else (field,)
    assert preview.values[path] == ("yoko1" if field == "section" else expected)
    saved = FormInput.from_json(NamesForm, FormInput(NamesForm, authored).to_json())
    restored = FormState(NamesForm.default_input(name_env))
    assert (
        restored.restore_input(FormInput(restored.input.model, saved.data), name_env)
        == ()
    )
    assert restored.input.data[field] == value
    assert restored.result(name_env) == result
    # A literal beginning with $ never opts in to evaluation.
    state.write({("name",): "$device", ("optional",): None}, name_env)
    literal = state.result(name_env)
    assert isinstance(literal, NamesForm)
    assert literal.name == "$device" and literal.optional is None


@pytest.mark.parametrize(
    "literal",
    [{"expr": "$device"}, {"expr": "1 / 0", "other": "$device"}],
    ids=["envelope-shaped", "multiple-keys"],
)
@pytest.mark.parametrize("annotation", [dict[str, str], Mapping[str, str]])
def test_literal_mapping_expr_key_is_data_through_defaults_write_preview_and_restore(
    name_env: FormEnv, literal: dict[str, JsonValue], annotation: object
) -> None:
    model = create_model(
        "LiteralMapping",
        __base__=FormModel,
        mapping=(annotation, literal),
        name=(str, ""),
    )
    state = FormState(model.default_input(name_env))
    assert state.input.data["mapping"] == literal
    state.write({("mapping",): literal, ("name",): {"expr": "$device"}}, name_env)
    result = state.result(name_env)
    assert not isinstance(result, FormErrors), result
    assert result.model_dump() == {"mapping": literal, "name": "yoko1"}
    preview = state.preview(name_env)
    assert preview.errors is None
    assert preview.values[("mapping",)] == literal
    saved = FormInput.from_json(model, state.input.to_json())
    restored = FormState(model.default_input(name_env))
    assert restored.restore_input(saved, name_env) == ()
    assert restored.input.data == state.input.data
    assert restored.result(name_env) == result


@pytest.mark.parametrize("expression", ["$missing", "$unbound", "1 + 2"])
def test_string_expression_failures_keep_authored_input_and_other_preview_values(
    name_env: FormEnv, expression: str
) -> None:
    state = FormState(FormInput(NamesForm, {"name": {"expr": "$device"}}))
    assert state.preview(name_env).values[("name",)] == "yoko1"
    state.write({("name",): {"expr": expression}, ("optional",): "ordinary"}, name_env)
    errors = state.result(name_env)
    assert isinstance(errors, FormErrors)
    assert ("name",) in errors.messages
    if expression.startswith("$"):
        assert expression[1:] in errors.messages[("name",)]
    preview = state.preview(name_env)
    assert preview.errors == errors
    assert preview.values[("optional",)] == "ordinary"
    assert state.input.data["name"] == {"expr": expression}


def test_string_expressions_obey_pydantic_constraints(name_env: FormEnv) -> None:
    state = FormState(NamesForm.default_input(name_env))
    assert name_env.parameters is not None
    short = FormEnv(
        parameters=name_env.parameters.bind_aliases((AliasRow("device", "x"),), {})
    )
    state.write({("name",): {"expr": "$device"}}, short)
    errors = state.result(short)
    assert isinstance(errors, FormErrors)
    assert "at least 2" in errors.messages[("name",)]


class FirstSelector(FormModel):
    """An ordinary string tag selects the short-name branch."""

    kind: Literal["first"] = "first"
    name: str = Field(default="short", max_length=10)


class SecondSelector(FormModel):
    """Another string tag selects a branch with a different name constraint."""

    kind: Literal["second"] = "second"
    name: str = Field(default="a sufficiently long name", min_length=20)


class SelectorForm(FormModel):
    """Selectors remain ordinary tags while their value leaves allow Expr."""

    branch: Annotated[FirstSelector | SecondSelector, Field(discriminator="kind")]


class NumericFirstSelector(FormModel):
    """An ordinary numeric tag selects the short-name branch."""

    kind: Literal[1] = 1
    name: str = Field(default="short", max_length=10)


class NumericSecondSelector(FormModel):
    """A second numeric tag selects a branch with a different name constraint."""

    kind: Literal[2] = 2
    name: str = Field(default="a sufficiently long name", min_length=20)


class NumericSelectorForm(FormModel):
    """Numeric selectors have the same ordinary-only schema as string selectors."""

    branch: Annotated[
        NumericFirstSelector | NumericSecondSelector, Field(discriminator="kind")
    ]


@pytest.mark.parametrize(
    ("model", "tag"),
    [
        (SelectorForm, "first"),
        (SelectorForm, "second"),
        (NumericSelectorForm, 1),
        (NumericSelectorForm, 2),
    ],
)
def test_structural_selectors_reject_expression_envelopes(
    name_env: FormEnv, model: type[FormModel], tag: JsonValue
) -> None:
    name = "short" if tag in ("first", 1) else "a sufficiently long name"
    ordinary: dict[str, JsonValue] = {"branch": {"kind": tag, "name": name}}
    result = FormState(FormInput(model, ordinary)).result(name_env)
    assert isinstance(result, model), result


class SingleSelectorForm(FormModel):
    """Even a single explicitly tagged model remains a structural selector."""

    branch: Annotated[FirstSelector, Field(discriminator="kind")]


def test_single_model_selector_rejects_expression_without_evaluating_tag(
    name_env: FormEnv,
) -> None:
    ordinary: dict[str, JsonValue] = {
        "branch": {"kind": "first", "name": {"expr": "$device"}}
    }
    state = FormState(FormInput(SingleSelectorForm, ordinary))
    assert isinstance(state.result(name_env), SingleSelectorForm)
    envelope: dict[str, JsonValue] = {
        "branch": {"kind": {"expr": "$device"}, "name": "short"}
    }
    assert name_env.parameters is not None
    env = FormEnv(
        parameters=name_env.parameters.bind_aliases((AliasRow("device", "first"),), {})
    )
    invalid = FormState(FormInput(SingleSelectorForm, envelope)).result(env)
    assert isinstance(invalid, FormErrors)
    assert ("branch",) in invalid.messages
    assert "kind" in invalid.messages[("branch",)]


class ContainerSelectorForm(FormModel):
    """Containers keep tagged models as whole leaves during restore."""

    items: list[Annotated[FirstSelector, Field(discriminator="kind")]]
    pair: tuple[Annotated[FirstSelector, Field(discriminator="kind")], ...]
    by_name: Mapping[str, Annotated[FirstSelector, Field(discriminator="kind")]]
    note: str = "unchanged"


class ScalarEnvelopeForm(FormModel):
    """Malformed envelopes retain field-specific input diagnostics."""

    name: str = "literal"
    amount: float = 1.0
    phasor: complex = 1j


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("name", "String expression requires {expr: text}"),
        ("amount", "Numeric expression requires {expr: text}"),
        ("phasor", "Numeric expression requires {expr: text}"),
    ],
)
def test_malformed_expression_errors_preserve_scalar_input_contract(
    name_env: FormEnv, field: str, message: str
) -> None:
    state = FormState(FormInput(ScalarEnvelopeForm, {field: {"expr": 3}}))
    errors = state.result(name_env)
    assert isinstance(errors, FormErrors)
    assert errors.messages[(field,)] == message
