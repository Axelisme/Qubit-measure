"""Expression defaults become direct values when input is built or a branch opens."""

from pathlib import Path
from typing import Annotated, Literal

import pytest
from pydantic import Field, JsonValue
from zcu_tools.gui.form import FormEnv, FormErrors, FormModel, FormState
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
            general=PointGeneral(created_at="2026-10-13T00:00:00Z", ext=ext),
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
        ),
        target=SavedPoint(entry_id="entry", point="synthetic"),
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


def test_default_input_holds_evaluated_alias_path_and_arithmetic_values() -> None:
    env = environment(4.0)
    data = Seeded.default_input(env).data
    assert data == {"device": "yoko1", "level": 4.0, "doubled": 8.0, "authored": 1.0}
    result = FormState(Seeded.default_input(env)).result(env)
    assert result == Seeded(device="yoko1", level=4.0, doubled=8.0)


def test_nested_section_defaults_are_evaluated_too() -> None:
    env = environment(2.5)
    assert Holder.default_input(env).data == {
        "section": {"device": "yoko1", "level": 2.5, "doubled": 5.0, "authored": 1.0}
    }


def test_branch_switch_evaluates_new_defaults() -> None:
    env = environment(4.0)
    state = FormState(Switching.default_input(env))
    state.write({("branch", "offset"): {"expr": "general.ext.seed / 8"}}, env)
    state.select(("branch",), {"kind": "derived"}, env)
    assert state.input.data == {
        "branch": {
            "kind": "derived",
            "level": 5.0,
            "offset": 0.0,
        }
    }
    result = state.result(env)
    assert isinstance(result, Switching)
    assert result.branch == Derived(level=5.0, offset=0.0)


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
    assert all(not isinstance(value, dict) for value in data.values())
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
        "point", Path("/unavailable/point.yaml"), "OSError", "unreadable"
    )
    env = environment(4.0, faults=(fault,))
    with pytest.raises(ContextSourceUnavailableError):
        Seeded.default_input(env)
