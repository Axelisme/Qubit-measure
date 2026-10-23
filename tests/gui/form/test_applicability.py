"""Conditional nullable subtrees retain drafts but do not consume values or sources."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

import pytest
from pydantic import Field, JsonValue
from zcu_tools.gui import form as core


class Drive(core.FormModel):
    """An applicable drive consumes a required positive amplitude and source."""

    gain: float = Field(gt=0)

    @classmethod
    def check_sources(cls, data: Mapping[str, JsonValue], env: core.FormEnv) -> None:
        """Model a captured source failure without touching external resources."""
        if data.get("gain") == "missing-source":
            raise core.field_error(("gain",), "Drive source unavailable")


class ConditionalForm(core.FormModel):
    """A neutral membership declaration controls a nullable subtree only."""

    selection: list[int]
    drive: Annotated[Drive | None, core.Display(collapsed=False)] = None
    readout: float = Field(gt=0)

    @classmethod
    def applicability(
        cls, data: Mapping[str, JsonValue]
    ) -> tuple[core.FieldApplicability, ...]:
        """Declare membership, leaving its evaluation to the core."""
        return (core.FieldApplicability(("selection",), 1, (("drive",),)),)


def state_for(
    drive: JsonValue, selection: JsonValue = None
) -> core.FormState[ConditionalForm]:
    """Build a draft directly, without evaluating its disabled source."""
    return core.FormState(
        core.FormInput(
            ConditionalForm,
            {
                "selection": [0] if selection is None else selection,
                "drive": drive,
                "readout": 6000,
            },
        )
    )


@pytest.mark.parametrize(
    "draft",
    [{"gain": {"expr": "unknown_value"}}, {"gain": -1}, {"gain": "missing-source"}],
)
def test_inapplicable_subtree_preserves_draft_without_evaluation_or_validation(
    draft: JsonValue,
) -> None:
    state = state_for(draft)
    before = state.input.to_json()
    result = state.result(core.FormEnv())
    assert not isinstance(result, core.FormErrors), result
    assert result.drive is None
    assert state.preview(core.FormEnv()).errors is None
    assert state.input.to_json() == before
    target = state.describe(("drive",), core.FormEnv()).target
    assert target is not None and getattr(target, "disabled", False)
    if draft == {"gain": "missing-source"}:
        return
    state.write({("selection",): [0, 1]}, core.FormEnv())
    errors = state.preview(core.FormEnv()).errors
    assert errors is not None and ("drive", "gain") in errors.messages
    assert state.input.data["drive"] == draft
    target = state.describe(("drive",), core.FormEnv()).target
    assert target is not None and not target.disabled


def test_default_input_preserves_inapplicable_expression_draft() -> None:
    data: dict[str, JsonValue] = {
        "selection": [0],
        "drive": {"gain": {"expr": "unknown_value"}},
        "readout": 6000,
    }
    initial = ConditionalForm.default_input(core.FormEnv(), defaults=data)
    assert initial.data == data


def test_membership_change_uses_same_input_and_restores_valid_expression() -> None:
    state = state_for({"gain": {"expr": "2 * 3"}})
    state.write({("selection",): [1]}, core.FormEnv())
    result = state.result(core.FormEnv())
    assert not isinstance(result, core.FormErrors), result
    assert result.drive is not None and result.drive.gain == 6
    state.write({("selection",): [0]}, core.FormEnv())
    result = state.result(core.FormEnv())
    assert not isinstance(result, core.FormErrors), result
    assert result.drive is None
    assert state.input.data["drive"] == {"gain": {"expr": "2 * 3"}}


@pytest.mark.parametrize("selection", [[0], [0, 1]])
def test_unconditional_readout_still_validates(selection: list[int]) -> None:
    state = state_for({"gain": 2}, [value for value in selection])
    state.write({("readout",): None}, core.FormEnv())
    errors = state.preview(core.FormEnv()).errors
    assert errors is not None and set(errors.messages) == {("readout",)}
    target = state.describe(("drive",), core.FormEnv()).target
    assert target is not None and getattr(target, "disabled", None) == (
        1 not in selection
    )


@pytest.mark.parametrize("selection", ["invalid", [{"expr": "unknown_index"}]])
def test_invalid_dependency_is_not_hidden_by_disabled_projection(
    selection: JsonValue,
) -> None:
    state = state_for({"gain": {"expr": "unknown_gain"}}, selection)
    errors = state.preview(core.FormEnv()).errors
    assert errors is not None
    assert any(path[:1] == ("selection",) for path in errors.messages)
    assert all(path[:1] == ("selection",) for path in errors.messages)
