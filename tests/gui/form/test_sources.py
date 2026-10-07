"""Captured source failures and module-reference switching at public seams."""

from collections.abc import Mapping
from typing import Annotated

import pytest
from jsonschema import Draft202012Validator
from pydantic import Field, JsonValue
from pydantic_core import PydanticCustomError
from zcu_tools.cfg_model import PulseSpec
from zcu_tools.device import FakeDeviceInfo
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    ModuleInline,
    ModuleInput,
    ModuleRef,
    field_error,
)
from zcu_tools.resources.context import MetaDict
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)


class ModuleForm(FormModel):
    """A generic library reference, unrelated to a particular workflow role."""

    drive: Annotated[ModuleRef | ModuleInline, Field(discriminator="kind")] = Field(
        default_factory=lambda: ModuleRef(name="Q1.pulses.pi")
    )


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
            format_version="1.0",
            components={"Q1": ComponentLibrary(pulses={"pi": pulse})},
        )
    )


class NullableModuleForm(FormModel):
    """Keep the public alias metadata inside a nullable field annotation."""

    drive: ModuleInput | None = Field(
        default_factory=lambda: ModuleRef(name="Q1.pulses.pi")
    )


@pytest.mark.parametrize("operation", ["defaults", "write", "result"])
def test_nullable_module_alias_requires_used_library(operation: str) -> None:
    env = FormEnv(md=MetaDict())
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


def test_nullable_module_alias_uses_captured_library_for_active_branch_switch() -> None:
    source = library()
    env = FormEnv(md=MetaDict(), library=source)
    before = source.document()
    state = FormState(NullableModuleForm.default_input(env))
    state.write({("drive", "kind"): "inline"}, env)
    result = state.result(env)
    assert isinstance(result, NullableModuleForm)
    assert isinstance(result.drive, ModuleInline)
    assert result.drive.module == source.module_spec("Q1.pulses.pi")
    assert source.document() == before


def test_public_module_alias_projects_reference_choices() -> None:
    class ModuleAliasForm(FormModel):
        """Parent options reach the exported module union reference branch."""

        drive: ModuleInput = Field(
            default_factory=lambda: ModuleRef(name="Q1.pulses.pi")
        )

        @classmethod
        def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
            """Allow the fixed reference name at its model-field path."""
            return {("drive", "name"): ("Q1.pulses.pi",)}

    env = FormEnv(md=MetaDict(), library=library())
    validator = Draft202012Validator(ModuleAliasForm.input_schema(env))
    valid: dict[str, JsonValue] = {"drive": {"kind": "ref", "name": "Q1.pulses.pi"}}
    assert validator.is_valid(valid)
    assert isinstance(
        FormState(FormInput(ModuleAliasForm, valid)).result(env), ModuleAliasForm
    )
    invalid: dict[str, JsonValue] = {
        "drive": {"kind": "ref", "name": "Q1.pulses.absent"}
    }
    errors = FormState(FormInput(ModuleAliasForm, invalid)).result(env)
    assert isinstance(errors, FormErrors)
    assert ("drive", "name") in errors.messages
    assert not validator.is_valid(invalid)


def test_ref_to_inline_seeds_a_detached_spec_without_materializing_children() -> None:
    source = library()
    env = FormEnv(md=MetaDict(), library=source)
    before = source.document()
    state = FormState(ModuleForm.default_input(env))
    reference = state.result(env)
    assert isinstance(reference, ModuleForm)
    assert isinstance(reference.drive, ModuleRef)
    assert reference.drive.name == "Q1.pulses.pi"

    state.write({("drive", "kind"): "inline"}, env)
    inline = state.result(env)
    assert isinstance(inline, ModuleForm)
    assert isinstance(inline.drive, ModuleInline)
    assert inline.drive.module == source.module_spec("Q1.pulses.pi")
    assert state.input.data["drive"] == {
        "kind": "inline",
        "module": source.module_spec("Q1.pulses.pi").model_dump(mode="json"),
    }

    state.write({("drive", "module", "gain"): 0.5}, env)
    changed_env = FormEnv(md=MetaDict(), library=library(0.75))
    detached = state.result(changed_env)
    assert isinstance(detached, ModuleForm)
    assert isinstance(detached.drive, ModuleInline)
    assert isinstance(detached.drive.module, PulseSpec)
    assert detached.drive.module.gain == 0.5
    assert detached.drive.module.waveform == "Q1.waveforms.not_materialized"
    assert source.document() == before


def test_same_batch_inline_content_overrides_the_library_seed() -> None:
    env = FormEnv(md=MetaDict(), library=library())
    state = FormState(ModuleForm.default_input(env))
    state.write({("drive", "module", "gain"): 0.9, ("drive", "kind"): "inline"}, env)
    result = state.result(env)
    assert isinstance(result, ModuleForm)
    assert isinstance(result.drive, ModuleInline)
    assert isinstance(result.drive.module, PulseSpec)
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
    env = FormEnv(md=MetaDict(), library=None if source is None else library())
    state = FormState(FormInput(ModuleForm, {"drive": {"kind": "ref", "name": name}}))
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert any(path[:1] == ("drive",) for path in result.messages)
    assert () not in result.messages

    before = state.input.data
    with pytest.raises(PydanticCustomError, match="drive"):
        state.write({("drive", "kind"): "inline"}, env)
    assert state.input.data == before


def test_default_input_fails_for_used_missing_library_not_unused_capability() -> None:
    env = FormEnv(md=MetaDict())
    with pytest.raises(PydanticCustomError, match="drive"):
        ModuleForm.default_input(env)


def test_captured_devices_choices_missing_source_and_no_mutation() -> None:
    env = FormEnv(md=MetaDict(), devices={"flux": FakeDeviceInfo(address="memory")})
    before = env.devices["flux"].model_copy(deep=True)
    state = FormState(DeviceChoiceForm.default_input(env))
    assert isinstance(state.result(env), DeviceChoiceForm)
    state.write({("device",): "absent"}, env)
    rejected = state.result(env)
    assert isinstance(rejected, FormErrors)
    assert ("device",) in rejected.messages
    assert env.devices["flux"] == before

    missing_env = FormEnv(md=MetaDict())
    result = state.result(missing_env)
    assert isinstance(result, FormErrors)
    assert ("device",) in result.messages
    current = state.input.data
    with pytest.raises(PydanticCustomError, match="device"):
        state.write({("device",): "flux"}, missing_env)
    assert state.input.data == current
    with pytest.raises(PydanticCustomError, match="device"):
        DeviceChoiceForm.default_input(missing_env)
