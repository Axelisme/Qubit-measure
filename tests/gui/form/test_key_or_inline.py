"""Tagged SyntheticModuleInput schema and captured library key suggestions."""

from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from zcu_tools.cfg_model import (
    ConstWaveformCfg,
    DirectReadoutCfg,
    PulseSpec,
)
from zcu_tools.gui.form import (
    FormEnv,
    FormModel,
    FormPath,
)
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)

from ._module_models import SyntheticModuleInput


class FlatShape(BaseModel):
    """First tagged branch; its tag has no default, like a required Literal."""

    model_config = ConfigDict(extra="forbid")

    style: Literal["flat"]
    length: float = 1.0


class RampShape(BaseModel):
    """Second tagged branch with a required branch-only field."""

    model_config = ConfigDict(extra="forbid")

    style: Literal["ramp"] = "ramp"
    length: float = 1.0
    sigma: float


type Shape = Annotated[FlatShape | RampShape, Field(discriminator="style")]


def _library() -> LibrarySnapshot:
    """Two pulses, a direct readout and a waveform under component Q1."""
    pulse = PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.2, waveform="Q1.waveforms.flat")
    return LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="1.0",
            components={
                "Q1": ComponentLibrary(
                    pulses={"pi": pulse, "pi2": pulse},
                    readout={
                        "direct": DirectReadoutCfg(
                            ro_ch=0, ro_length=1.0, ro_freq=100.0
                        )
                    },
                    waveforms={"flat": ConstWaveformCfg(length=1.0)},
                )
            },
        )
    )


class ModuleForm(FormModel):
    """A SyntheticModuleInput beside a plain tagged union, which is not key-or-inline."""

    drive: SyntheticModuleInput
    either: Shape = Field(default_factory=lambda: FlatShape(style="flat"))


class ChoiceModuleForm(FormModel):
    """A SyntheticModuleInput whose ref name declares choices, one not in the library."""

    drive: SyntheticModuleInput

    @classmethod
    def choices(cls, _env: FormEnv, /) -> dict[FormPath, tuple[JsonValue, ...]]:
        return {("drive", "name"): ("Q1.pulses.pi2", "Q1.pulses.renamed")}


def _ref_name(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Return the drive field's ref-branch name property node."""
    properties = schema["properties"]
    assert isinstance(properties, dict)
    field = properties["drive"]
    assert isinstance(field, dict)
    branches = field["oneOf"]
    assert isinstance(branches, list)
    ref = next(
        branch
        for branch in branches
        if isinstance(branch, dict) and branch["x-form-discriminator-values"] == ["ref"]
    )
    assert isinstance(ref, dict)
    ref_properties = ref["properties"]
    assert isinstance(ref_properties, dict)
    name = ref_properties["name"]
    assert isinstance(name, dict)
    return name


def test_schema_marks_a_module_input_field_as_key_or_inline() -> None:
    schema = ModuleForm.input_schema(FormEnv())
    properties = schema["properties"]
    assert isinstance(properties, dict)
    drive = properties["drive"]
    either = properties["either"]
    assert isinstance(drive, dict)
    assert isinstance(either, dict)

    assert drive["x-form-key-or-inline"] is True
    assert "x-form-key-or-inline" not in either


def test_module_ref_name_lists_every_captured_module_key() -> None:
    name = _ref_name(ModuleForm.input_schema(FormEnv(library=_library())))

    # Inline holds any ModuleSpec, so every module key matches; waveforms do not.
    assert name["x-form-library-keys"] == [
        "Q1.pulses.pi",
        "Q1.pulses.pi2",
        "Q1.readout.direct",
    ]


def test_module_ref_name_keys_keep_only_declared_choices_in_the_library() -> None:
    name = _ref_name(ChoiceModuleForm.input_schema(FormEnv(library=_library())))

    assert name["x-form-library-keys"] == ["Q1.pulses.pi2"]


@pytest.mark.parametrize("form", [ModuleForm, ChoiceModuleForm])
def test_module_ref_name_has_no_keys_without_a_captured_library(
    form: type[FormModel],
) -> None:
    assert "x-form-library-keys" not in _ref_name(form.input_schema(FormEnv()))
