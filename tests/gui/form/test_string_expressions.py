"""String input envelopes resolve names without changing authored input."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

import pytest
from jsonschema import Draft202012Validator
from pydantic import Field, JsonValue
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPath,
    FormState,
)
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import EntryDraft
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class DeviceName(StrEnum):
    """A string enum retains Pydantic membership after evaluation."""

    YOKO = "yoko1"


class NameSection(FormModel):
    """Nested ordinary strings use the same input contract."""

    name: str


class NamesForm(FormModel):
    """Names also occur inside whole containers and nullable sections."""

    name: str = Field(default="literal", min_length=2)
    optional: str | None = None
    mode: Literal["yoko1", "other"] = "yoko1"
    device: DeviceName = DeviceName.YOKO
    names: tuple[str, ...] = ()
    mapping: dict[str, str] = Field(default_factory=dict)
    section: NameSection | None = None

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Restrict mode even when Pydantic accepts a different literal."""
        return {("mode",): ("yoko1",)}


@pytest.fixture
def name_env() -> FormEnv:
    """Capture authored aliases without requiring devices or live sources."""
    context = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="1.0",
            general=PointGeneral(created_at="2026-10-11T00:00:00Z"),
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
        ),
        target=EntryDraft(entry_id="entry"),
        roles=None,
    ).bind_aliases((AliasRow("device", "yoko1"), AliasRow("unbound", None)), {})
    return FormEnv(parameters=context)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("name", {"expr": "$device"}, "yoko1"),
        ("optional", {"expr": "$device"}, "yoko1"),
        ("mode", {"expr": "$device"}, "yoko1"),
        ("device", {"expr": "$device"}, DeviceName.YOKO),
        ("names", [{"expr": "$device"}, "$device"], ("yoko1", "$device")),
        ("mapping", {"rack.left": {"expr": "$device"}}, {"rack.left": "yoko1"}),
        ("section", {"name": {"expr": "$device"}}, NameSection(name="yoko1")),
    ],
)
def test_string_envelopes_share_schema_preview_result_and_restore(
    name_env: FormEnv, field: str, value: JsonValue, expected: object
) -> None:
    state = FormState(NamesForm.default_input(name_env))
    authored: dict[str, JsonValue] = {field: value}
    assert Draft202012Validator(state.input_schema(name_env)).is_valid(authored)
    state.write({(field,): value}, name_env)
    result = state.result(name_env)
    assert isinstance(result, NamesForm), result
    assert getattr(result, field) == expected
    preview = state.preview(name_env)
    assert preview.errors is None
    path = ("section", "name") if field == "section" else (field,)
    assert preview.values[path] == ("yoko1" if field == "section" else expected)
    saved = FormInput.from_json(NamesForm, state.input.to_json())
    restored = FormState(NamesForm.default_input(name_env))
    assert restored.write_compatible(saved.data, name_env) == ()
    assert restored.input.data[field] == value
    assert restored.result(name_env) == result
    # A literal beginning with $ never opts in to evaluation.
    state.write({("name",): "$device", ("optional",): None}, name_env)
    literal = state.result(name_env)
    assert isinstance(literal, NamesForm)
    assert literal.name == "$device" and literal.optional is None
    invalid = {field: {"expr": 3}}
    assert not Draft202012Validator(state.input_schema(name_env)).is_valid(invalid)


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


def test_string_expressions_obey_choices_and_pydantic_constraints(
    name_env: FormEnv,
) -> None:
    state = FormState(FormInput(NamesForm, {"mode": {"expr": "$device"}}))
    assert isinstance(state.result(name_env), NamesForm)
    assert name_env.parameters is not None
    other = FormEnv(
        parameters=name_env.parameters.bind_aliases((AliasRow("device", "other"),), {})
    )
    errors = state.result(other)
    assert isinstance(errors, FormErrors)
    assert "choices" in errors.messages[("mode",)]
    short = FormEnv(
        parameters=name_env.parameters.bind_aliases((AliasRow("device", "x"),), {})
    )
    state.write({("name",): {"expr": "$device"}, ("mode",): "yoko1"}, short)
    errors = state.result(short)
    assert isinstance(errors, FormErrors)
    assert "at least 2" in errors.messages[("name",)]
