"""Defaults retain Expr text for Eval leaves and direct values for choices."""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

import pytest
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
from zcu_tools.gui.session.context_types import (
    ContextSourceFault,
    ContextSourceUnavailableError,
    SavedPoint,
)
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

from tests.gui._form_converters import RebuildingForm

SEEDED_EXPRESSIONS: dict[str, JsonValue] = {
    "device": {"expr": "$device"},
    "level": {"expr": "general.ext.seed"},
    "doubled": {"expr": "general.ext.seed * 2"},
}


class Seeded(FormModel):
    """Required leaves whose initial values come from an alias, a path and math."""

    device: str = Field(min_length=1)
    level: float = Field(gt=0)
    doubled: float = Field(gt=0)
    authored: float = 1.0

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        data = super().defaults(env)
        data.update(SEEDED_EXPRESSIONS)
        return data


class Holder(FormModel):
    """A nested section whose defaults come from its own model's hook."""

    section: Seeded

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        data = super().defaults(env)
        data["section"] = Seeded.defaults(env)
        return data


class Manual(FormModel):
    """The initial branch: no expression defaults."""

    kind: Literal["manual"] = "manual"
    offset: float = 0.0


class Derived(FormModel):
    """A branch whose level default reads the captured point."""

    kind: Literal["derived"] = "derived"
    level: float = Field(gt=0)
    offset: float = 0.0

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        data = super().defaults(env)
        data["level"] = {"expr": "general.ext.seed + 1"}
        return data


class Switching(RebuildingForm):
    """Switch between the branches above."""

    branch_models = {("branch",): (Manual, Derived)}

    branch: Annotated[Manual | Derived, Field(discriminator="kind")] = Field(
        default_factory=Manual
    )


def environment(
    seed: JsonValue = 4.0,
    *,
    device: str = "yoko1",
    faults: tuple[ContextSourceFault, ...] = (),
) -> FormEnv:
    """Capture a point whose general.ext holds seed (omitted when None)."""
    ext: dict[str, JsonValue] = {} if seed is None else {"seed": seed}
    context = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="2.0",
            schema_version=1,
            general=PointGeneral(created_at="2026-10-13T00:00:00Z", ext=ext),
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
        ),
        target=SavedPoint(entry_name="entry", point="synthetic"),
        roles=None,
    ).bind_aliases((AliasRow("device", device),), {})
    if not faults:
        return FormEnv(parameters=context)
    return FormEnv(
        parameters=ParameterContext(
            point=context.document(),
            library=context.library,
            target=context.target,
            roles=None,
            source_faults=faults,
        )
    )


class NumericChoice(FormModel):
    """A required numeric leaf accepts only the captured choices."""

    level: float

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Offer only the numeric value 2.0."""
        return {("level",): (2.0,)}

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        """Derive the choice's initial value from an expression."""
        del env
        return {"level": {"expr": "2"}}


def test_accepted_choice_expression_default_becomes_direct() -> None:
    env = FormEnv()
    initial = NumericChoice.default_input(env)
    assert initial.data == {"level": 2.0}
    assert FormState(initial).result(env) == NumericChoice(level=2.0)


class StaticNumericChoice(NumericChoice):
    """An omitted choice falls back to its static value during validation."""

    level: float = 2.0


@pytest.mark.parametrize("model", [NumericChoice, StaticNumericChoice])
def test_nonmember_choice_expression_default_is_left_unfilled(
    model: type[NumericChoice],
) -> None:
    env = FormEnv()
    initial = model.default_input(env, defaults={"level": {"expr": "3"}})
    assert initial.data == {}
    result = FormState(initial).result(env)
    if model is StaticNumericChoice:
        assert result == StaticNumericChoice(level=2.0)
    else:
        assert result == FormErrors({("level",): "Field required"})


class NestedChoice(FormModel):
    """The enclosing model declares a choice for a nested text-field leaf."""

    section: Derived

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Restrict only this occurrence of the nested leaf."""
        return {("section", "level"): (2.0,)}

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        return {"section": Derived.defaults(env)}


def test_nested_choice_expression_default_becomes_direct() -> None:
    env = environment(1.0)
    initial = NestedChoice.default_input(env)
    assert initial.data == {"section": {"kind": "derived", "level": 2.0, "offset": 0.0}}
    assert FormState(initial).result(env) == NestedChoice(section=Derived(level=2.0))


class ChoiceBranch(NumericChoice):
    """A branch with local numeric choices and expression defaults."""

    kind: Literal["choice"] = "choice"

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        return {**super().defaults(env), "kind": "choice"}


class ChoiceSwitching(RebuildingForm):
    """Switching to the choice branch uses its declared defaults."""

    branch_models = {("branch",): (Manual, ChoiceBranch)}
    branch: Annotated[Manual | ChoiceBranch, Field(discriminator="kind")] = Field(
        default_factory=Manual
    )


def test_branch_choice_expression_default_becomes_direct() -> None:
    env = FormEnv()
    state = FormState(ChoiceSwitching.default_input(env))
    state.select(("branch",), {"kind": "choice"}, env)
    assert state.input.data == {"branch": {"kind": "choice", "level": 2.0}}
    assert state.result(env) == ChoiceSwitching(branch=ChoiceBranch(level=2.0))


def test_restored_choice_expression_remains_invalid_authored_input() -> None:
    env = FormEnv()
    state = FormState(NumericChoice.default_input(env))
    saved = FormInput(NumericChoice, {"level": {"expr": "2"}})
    assert state.restore_input(saved, env) == ()
    assert state.input.data == {"level": {"expr": "2"}}
    assert state.result(env) == FormErrors(
        {("level",): "Choice fields take a listed value, not an expression"}
    )


def test_default_input_keeps_alias_path_and_arithmetic_expressions() -> None:
    env = environment(4.0)
    data = Seeded.default_input(env).data
    assert data == {**SEEDED_EXPRESSIONS, "authored": 1.0}
    result = FormState(Seeded.default_input(env)).result(env)
    assert result == Seeded(device="yoko1", level=4.0, doubled=8.0)


def test_nested_section_defaults_keep_their_expressions() -> None:
    env = environment(2.5)
    assert Holder.default_input(env).data == {
        "section": {**SEEDED_EXPRESSIONS, "authored": 1.0}
    }


def test_branch_switch_preserves_new_expression_defaults() -> None:
    env = environment(4.0)
    state = FormState(Switching.default_input(env))
    state.write({("branch", "offset"): {"expr": "general.ext.seed / 8"}}, env)
    state.select(("branch",), {"kind": "derived"}, env)
    assert state.input.data == {
        "branch": {
            "kind": "derived",
            "level": {"expr": "general.ext.seed + 1"},
            "offset": 0.0,
        }
    }
    result = state.result(env)
    assert isinstance(result, Switching)
    assert result.branch == Derived(level=5.0, offset=0.0)


def test_compatible_restore_into_new_branch_keeps_default_expression() -> None:
    env = environment(4.0)
    state = FormState(Switching.default_input(env))
    saved = FormInput(Switching, {"branch": {"kind": "derived", "offset": 2.0}})
    assert state.restore_input(saved, env) == ()
    assert state.input.data["branch"] == {
        "kind": "derived",
        "level": {"expr": "general.ext.seed + 1"},
        "offset": 2.0,
    }
    assert state.preview(environment(9.0)).values[("branch", "level")] == 10.0


def test_reset_rebuilds_defaults_instead_of_retaining_edits() -> None:
    env = environment(4.0)
    state = FormState(Seeded.default_input(env))
    state.write({("level",): 99.0}, env)
    updated = environment(9.0)
    state.reset(Seeded.default_input(updated))
    assert state.input.data == {**SEEDED_EXPRESSIONS, "authored": 1.0}
    assert state.result(updated) == Seeded(device="yoko1", level=9.0, doubled=18.0)
    assert not state.is_dirty


class Inline(RebuildingForm):
    """A synthetic inline wrapper supplies nested expression defaults."""

    kind: Literal["inline"] = "inline"
    section: Derived

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        return {"kind": "inline", "section": Derived.defaults(env)}


class InlineSwitching(RebuildingForm):
    """Fresh inline selection uses the target wrapper's defaults."""

    branch_models = {("branch",): (Manual, Inline)}
    branch: Annotated[Manual | Inline, Field(discriminator="kind")] = Field(
        default_factory=Manual
    )


def test_inline_switch_preserves_nested_expression_default() -> None:
    env = environment(4.0)
    state = FormState(InlineSwitching.default_input(env))
    state.select(("branch",), {"kind": "inline"}, env)
    assert state.input.data["branch"] == {
        "kind": "inline",
        "section": {
            "kind": "derived",
            "level": {"expr": "general.ext.seed + 1"},
            "offset": 0.0,
        },
    }
    assert (
        state.preview(environment(9.0)).values[("branch", "section", "level")] == 10.0
    )


def test_failed_expression_uses_existing_static_default() -> None:
    env = environment()
    raw: dict[str, JsonValue] = {**SEEDED_EXPRESSIONS, "authored": {"expr": "1 / 0"}}
    state = FormState(Seeded.default_input(env, defaults=raw))
    assert "authored" not in state.input.data
    assert state.result(env) == Seeded(device="yoko1", level=4.0, doubled=8.0)


def test_authored_expressions_keep_their_text() -> None:
    env = environment(4.0)
    state = FormState(Seeded.default_input(env))
    state.write({("level",): {"expr": "general.ext.seed + 1"}}, env)
    assert state.input.data["level"] == {"expr": "general.ext.seed + 1"}
    result = state.result(env)
    assert isinstance(result, Seeded) and result.level == 5.0


@pytest.mark.parametrize(
    ("env", "unfilled"),
    [
        pytest.param(FormEnv(), ("device", "level", "doubled"), id="no-capture"),
        pytest.param(environment(4.0, device=""), ("device",), id="unbound-alias"),
        pytest.param(environment(None), ("level", "doubled"), id="missing-path"),
        pytest.param(environment("text"), ("level", "doubled"), id="non-numeric"),
        pytest.param(environment(-1.0), ("level", "doubled"), id="out-of-range"),
    ],
)
def test_uncomputable_or_invalid_defaults_leave_only_field_required(
    env: FormEnv, unfilled: tuple[str, ...]
) -> None:
    data = Seeded.default_input(env).data
    assert all(name not in data for name in unfilled)
    result = FormState(Seeded.default_input(env)).result(env)
    assert isinstance(result, FormErrors)
    assert result.messages == {(name,): "Field required" for name in unfilled}


def test_invalid_branch_default_leaves_the_branch_field_unfilled() -> None:
    env = environment(-3.0)
    state = FormState(Switching.default_input(env))
    state.select(("branch",), {"kind": "derived"}, env)
    assert state.input.data == {"branch": {"kind": "derived", "offset": 0.0}}
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert result.messages == {("branch", "level"): "Field required"}


def test_unavailable_captured_source_still_raises() -> None:
    fault = ContextSourceFault(
        "point", Path("/unavailable/parameters.yaml"), "OSError", "unreadable"
    )
    env = environment(4.0, faults=(fault,))
    with pytest.raises(ContextSourceUnavailableError):
        Seeded.default_input(env)
