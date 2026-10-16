"""Only text fields take Expr input; picked values and library keys take none.

A union mixing text and picked values has no control, so it is rejected.
"""

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    FormWritePath,
)
from zcu_tools.gui.form_adapter import ModuleRef
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

from ._module_models import SyntheticModuleInput


class Speed(StrEnum):
    """A string Enum: its values are picked from a dropdown."""

    FAST = "fast"
    SLOW = "slow"


class PickedForm(FormModel):
    """Every dropdown/checkbox leaf kind beside the four text field kinds."""

    unit: Literal["native", "mA"] = "native"
    nqz: Literal[1, 2] = 2
    optional_unit: Literal["native", "mA"] | None = None
    speed: Speed = Speed.FAST
    enabled: bool = True
    gain: float = 1.0
    device: str = "yoko1"
    drive: SyntheticModuleInput = Field(
        default_factory=lambda: ModuleRef(name="Q1.pulses.pi")
    )
    amount: float = 2.0
    count: int = 3
    phasor: complex = 1j
    name: str = "literal"

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Offer a numeric and a string choice leaf."""
        return {("gain",): (1.0, 2.0), ("device",): ("yoko1", "yoko2")}


TEXT_FIELDS = ("amount", "count", "phasor", "name")


@pytest.fixture
def env() -> FormEnv:
    """Capture aliases that resolve to legal values, and the referenced pulse."""
    pulse = PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.2, waveform="Q1.waveforms.flat")
    library = LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="2.0",
            components={
                "Q1": ComponentLibrary(
                    pulses={"pi": pulse, "pi2": pulse},
                    waveforms={"flat": ConstWaveformCfg(length=1.0)},
                )
            },
        )
    )
    parameters = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="2.0",
            general=PointGeneral(created_at="2026-10-13T00:00:00Z"),
        ),
        library=library,
        target=SavedPoint(entry_id="entry", point="synthetic"),
        roles=None,
        storage=None,
    ).bind_aliases(
        (
            AliasRow("unit", "mA"),
            AliasRow("speed", "slow"),
            AliasRow("device", "yoko2"),
        ),
        {},
    )
    return FormEnv(parameters=parameters, library=library)


def _object(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    "field",
    [
        "unit",
        "nqz",
        "optional_unit",
        "speed",
        "enabled",
        "gain",
        "device",
        "drive",
    ],
)
def test_dropdown_and_checkbox_leaves_have_no_expr_alternative(
    env: FormEnv, field: str
) -> None:
    path = ("drive", "name") if field == "drive" else (field,)
    target = FormState(PickedForm.default_input(env)).describe(path, env).target
    assert target is not None and not target.expression


@pytest.mark.parametrize("field", TEXT_FIELDS)
def test_text_fields_keep_their_expr_alternative(env: FormEnv, field: str) -> None:
    target = FormState(PickedForm.default_input(env)).describe((field,), env).target
    assert target is not None and target.expression


# Each expression resolves without error, so a result that evaluated it would
# hold its value there; the field's error shows that it did not.
PICKED_WRITES: tuple[tuple[FormWritePath, str], ...] = (
    (("unit",), "$unit"),
    (("nqz",), "1"),
    (("optional_unit",), "$unit"),
    (("speed",), "$speed"),
    (("enabled",), "1"),
    (("gain",), "2"),
    (("device",), "$device"),
    (("drive", "name"), "$device"),
)


@pytest.mark.parametrize(("path", "text"), PICKED_WRITES)
def test_expr_on_a_picked_leaf_is_an_error_at_that_leaf_without_evaluation(
    env: FormEnv, path: FormWritePath, text: str
) -> None:
    state = FormState(PickedForm.default_input(env))
    envelope: JsonValue = {"expr": text}
    state.write({path: envelope}, env)
    errors = state.result(env)
    assert isinstance(errors, FormErrors)
    assert path in errors.messages
    # The authored envelope stays in input for the user to replace.
    assert _read(state.input.data, path) == envelope


@pytest.mark.parametrize(
    ("field", "text", "value"),
    [
        ("amount", "1 / 4", 0.25),
        ("count", "2 + 3", 5),
        ("phasor", "1 + 2j", 1 + 2j),
        ("name", "$device", "yoko2"),
    ],
)
def test_text_field_expressions_still_evaluate(
    env: FormEnv, field: str, text: str, value: object
) -> None:
    state = FormState(PickedForm.default_input(env))
    state.write({(field,): {"expr": text}}, env)
    result = state.result(env)
    assert isinstance(result, PickedForm), result
    assert getattr(result, field) == value


def test_compatible_restore_drops_expr_on_picked_leaves_and_library_keys(
    env: FormEnv,
) -> None:
    state = FormState(PickedForm.default_input(env))
    before = state.input.data
    saved: dict[str, JsonValue] = {
        "unit": {"expr": "$unit"},
        "speed": {"expr": "$speed"},
        "drive": {"kind": "ref", "name": {"expr": "$device"}},
        "amount": {"expr": "1 / 4"},
    }
    dropped = state.write_compatible(saved, env)
    assert set(dropped) == {("unit",), ("speed",), ("drive", "name")}
    assert state.input.data == {**before, "amount": {"expr": "1 / 4"}}


class AutoGainForm(FormModel):
    """A number or a picked "auto": a union of text and picked values."""

    gain: float | Literal["auto"] = 1.0


class FlagCountForm(FormModel):
    """A checkbox value or a typed integer: bool is picked, int is text."""

    count: bool | int = 1


class Level(BaseModel):
    """A nested section holding a mixed union leaf."""

    model_config = ConfigDict(extra="forbid")

    level: Speed | str = "fast"


class NestedMixedForm(FormModel):
    """A mixed union leaf one model level down."""

    section: Level = Field(default_factory=Level)


MIXED_FORMS: tuple[tuple[type[FormModel], dict[str, JsonValue], str], ...] = (
    (AutoGainForm, {"gain": 1.0}, "gain"),
    (FlagCountForm, {"count": 1}, "count"),
    (NestedMixedForm, {"section": {"level": "fast"}}, "section.level"),
)
MIXED_MESSAGE = "a union of text and picked values is not supported"


@pytest.mark.parametrize(("form", "data", "path"), MIXED_FORMS)
def test_description_rejects_a_union_of_text_and_picked_values(
    env: FormEnv, form: type[FormModel], data: dict[str, JsonValue], path: str
) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(path)}: {MIXED_MESSAGE}$"):
        FormState(FormInput(form, data)).describe(tuple(path.split(".")), env)


@pytest.mark.parametrize(("form", "data", "path"), MIXED_FORMS)
def test_result_rejects_a_union_of_text_and_picked_values(
    env: FormEnv, form: type[FormModel], data: dict[str, JsonValue], path: str
) -> None:
    state = FormState(FormInput(form, data))
    with pytest.raises(ValueError, match=f"^{re.escape(path)}: {MIXED_MESSAGE}$"):
        state.result(env)


def _read(data: Mapping[str, JsonValue], path: FormWritePath) -> JsonValue:
    current: JsonValue = dict(data)
    for part in path:
        current = _object(current)[part]
    return current
