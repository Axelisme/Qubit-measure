"""Captured source failures and module-reference switching at public seams."""

from collections.abc import Mapping

import pytest
from pydantic import Field, JsonValue
from pydantic_core import PydanticCustomError
from zcu_tools.cfg_model import PulseSpec
from zcu_tools.device import BaseDeviceInfo, FakeDeviceInfo
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    field_error,
)
from zcu_tools.gui.form_adapter import ModuleRef
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)

from tests.gui._form_converters import RebuildingForm

from ._module_models import KeyPulseSpec, SyntheticModuleInline, SyntheticModuleInput


class ModuleForm(RebuildingForm):
    """A generic library reference, unrelated to a particular workflow role."""

    branch_models = {("drive",): (ModuleRef, SyntheticModuleInline)}

    drive: SyntheticModuleInput = Field(
        default_factory=lambda: ModuleRef(name="Q1.pulses.pi")
    )


class RestoreModuleForm(FormModel):
    """Fresh partial input does not require a library until a ref name is supplied."""

    drive: SyntheticModuleInput
    readout: ModuleRef
    averages: int = 1
    delay: float = 0.0


@pytest.mark.parametrize("capture", ["absent", "unmatched", "matched"])
def test_compatible_restore_reports_unmatched_sources_and_keeps_other_leaves(
    capture: str,
) -> None:
    env = FormEnv(library=library() if capture != "absent" else None)
    name = "Q1.pulses.pi" if capture != "unmatched" else "Q1.pulses.removed"
    saved = FormInput(
        RestoreModuleForm,
        {
            "drive": {"kind": "ref", "name": name},
            "readout": {"kind": "ref", "name": name},
            "averages": 17,
            "delay": "wrong type",
            "removed": 4,
        },
    )
    state = FormState(RestoreModuleForm.default_input(env))
    before_base = state.base.data
    notifications: list[dict[str, JsonValue]] = []
    state.subscribe(lambda: notifications.append(state.input.data))

    dropped = state.restore_input(FormInput(state.input.model, saved.data), env)

    expected_names = {"name": name}
    assert state.input.data == {
        "drive": {"kind": "ref", **expected_names},
        "readout": {"kind": "ref", **expected_names},
        "averages": 17,
        "delay": "wrong type",
    }
    expected_dropped: set[FormPath] = {("removed",)}
    assert set(dropped) == expected_dropped
    assert state.base.data == before_base
    assert notifications == [state.input.data]


def test_compatible_restore_keeps_current_reference_for_unmatched_saved_name() -> None:
    env = FormEnv(library=library())
    state = FormState(ModuleForm.default_input(env))
    before = state.input.data
    notifications: list[dict[str, JsonValue]] = []
    state.subscribe(lambda: notifications.append(state.input.data))

    dropped = state.restore_input(
        FormInput(
            state.input.model, {"drive": {"kind": "ref", "name": "Q1.pulses.removed"}}
        ),
        env,
    )

    assert dropped == ()
    assert state.input.data == {
        **before,
        "drive": {"kind": "ref", "name": "Q1.pulses.removed"},
    }
    assert notifications == [state.input.data]
    errors = state.result(env)
    assert isinstance(errors, FormErrors)
    assert ("drive", "name") in errors.messages


class DeviceChoiceForm(FormModel):
    """Declare device membership against captured infos, without polling."""

    device: str = "flux"

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Require captured device capability at the consuming field."""
        if not env.devices:
            raise field_error(("device",), "Missing captured devices for device")
        return {("device",): tuple(env.devices)}


def library(gain: float = 0.25) -> LibrarySnapshot:
    """Build memory-only Specs with an intentionally unresolved waveform ref."""
    pulse = PulseSpec(
        ch=0, nqz=1, freq=100.0, gain=gain, waveform="Q1.waveforms.not_materialized"
    )
    return LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="2.0",
            components={"Q1": ComponentLibrary(pulses={"pi": pulse})},
        )
    )


class NullableModuleForm(RebuildingForm):
    """Keep the public alias metadata inside a nullable field annotation."""

    branch_models = {("drive",): (ModuleRef, SyntheticModuleInline)}

    drive: SyntheticModuleInput | None = Field(
        default_factory=lambda: ModuleRef(name="Q1.pulses.pi")
    )


@pytest.mark.parametrize("operation", ["defaults", "write", "result"])
def test_nullable_module_alias_requires_used_library(operation: str) -> None:
    env = FormEnv()
    reference: dict[str, JsonValue] = {"kind": "ref", "name": "Q1.pulses.pi"}
    if operation == "defaults":
        with pytest.raises(PydanticCustomError, match="drive.name"):
            NullableModuleForm.default_input(env)
    elif operation == "write":
        state = FormState(FormInput(NullableModuleForm, {"drive": None}))
        before = state.input.data
        base = state.base.data
        notifications: list[dict[str, JsonValue]] = []
        state.subscribe(lambda: notifications.append(state.input.data))
        with pytest.raises(PydanticCustomError, match="drive.name"):
            state.write({("drive",): reference}, env)
        assert state.input.data == before
        assert state.base.data == base
        assert notifications == []
    else:
        state = FormState(FormInput(NullableModuleForm, {"drive": reference}))
        errors = state.result(env)
        assert isinstance(errors, FormErrors)
        assert ("drive", "name") in errors.messages
    null = FormState(FormInput(NullableModuleForm, {"drive": None})).result(env)
    assert isinstance(null, NullableModuleForm)
    assert null.drive is None


def test_public_module_alias_projects_reference_choices() -> None:
    class ModuleAliasForm(FormModel):
        """Parent options reach the exported module union reference branch."""

        drive: SyntheticModuleInput = Field(
            default_factory=lambda: ModuleRef(name="Q1.pulses.pi")
        )

        @classmethod
        def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
            """Allow the fixed reference name at its model-field path."""
            return {("drive", "name"): ("Q1.pulses.pi",)}

    env = FormEnv(library=library())
    valid: dict[str, JsonValue] = {"drive": {"kind": "ref", "name": "Q1.pulses.pi"}}
    assert isinstance(
        FormState(FormInput(ModuleAliasForm, valid)).result(env), ModuleAliasForm
    )
    invalid: dict[str, JsonValue] = {
        "drive": {"kind": "ref", "name": "Q1.pulses.absent"}
    }
    errors = FormState(FormInput(ModuleAliasForm, invalid)).result(env)
    assert isinstance(errors, FormErrors)
    assert ("drive", "name") in errors.messages


def test_same_batch_explicit_inline_content_follows_branch_selection() -> None:
    env = FormEnv(library=library())
    state = FormState(ModuleForm.default_input(env))
    state.write(
        {
            ("drive", "module", "gain"): 0.9,
            ("drive", "module", "waveform"): "Q1.waveforms.not_materialized",
            ("drive", "module", "type"): "pulse",
            ("drive", "module", "ch"): 0,
            ("drive", "module", "nqz"): 1,
            ("drive", "module", "freq"): 100.0,
            ("drive", "kind"): "inline",
        },
        env,
    )
    result = state.result(env)
    assert isinstance(result, ModuleForm)
    assert isinstance(result.drive, SyntheticModuleInline)
    assert isinstance(result.drive.module, KeyPulseSpec)
    assert result.drive.module.gain == 0.9


@pytest.mark.parametrize("source", [None, "missing_key", "wrong_family"])
def test_used_reference_source_failure_keeps_its_consuming_path(
    source: str | None,
) -> None:
    name = {
        None: "Q1.pulses.pi",
        "missing_key": "Q1.pulses.absent",
        "wrong_family": "Q1.waveforms.envelope",
    }[source]
    env = FormEnv(library=None if source is None else library())
    state = FormState(FormInput(ModuleForm, {"drive": {"kind": "ref", "name": name}}))
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert any(path[:1] == ("drive",) for path in result.messages)
    assert () not in result.messages


def test_default_input_fails_for_used_missing_library_not_unused_capability() -> None:
    env = FormEnv()
    with pytest.raises(PydanticCustomError, match="drive"):
        ModuleForm.default_input(env)


def test_captured_devices_choices_missing_source_and_no_mutation() -> None:
    env = FormEnv(devices={"flux": FakeDeviceInfo(address="memory")})
    before = env.devices["flux"].model_copy(deep=True)
    state = FormState(DeviceChoiceForm.default_input(env))
    assert isinstance(state.result(env), DeviceChoiceForm)
    state.write({("device",): "absent"}, env)
    rejected = state.result(env)
    assert isinstance(rejected, FormErrors)
    assert ("device",) in rejected.messages
    assert env.devices["flux"] == before

    missing_env = FormEnv()
    result = state.result(missing_env)
    assert isinstance(result, FormErrors)
    assert ("device",) in result.messages
    current = state.input.data
    with pytest.raises(PydanticCustomError, match="device"):
        state.write({("device",): "flux"}, missing_env)
    assert state.input.data == current
    with pytest.raises(PydanticCustomError, match="device"):
        DeviceChoiceForm.default_input(missing_env)


class CustomDriverInfo(BaseDeviceInfo):
    """A custom driver's detached info, outside the native DeviceInfo union."""

    type: str = "custom/attenuator"
    attenuation_db: float = 0.0


class SourceFreeForm(FormModel):
    """Evaluate without reading any captured device field."""

    averages: int = 1


def test_custom_driver_info_is_a_captured_device() -> None:
    custom = CustomDriverInfo(address="lab:attenuator", attenuation_db=3.0)
    env = FormEnv(devices={"flux": FakeDeviceInfo(address="none"), "atten": custom})

    state = FormState(DeviceChoiceForm.default_input(env))
    state.write({("device",): "atten"}, env)
    chosen = state.result(env)
    assert isinstance(chosen, DeviceChoiceForm) and chosen.device == "atten"

    unused = FormState(SourceFreeForm.default_input(env))
    assert unused.result(env) == SourceFreeForm(averages=1)
    assert env.devices["atten"] is custom
