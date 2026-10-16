"""Handwritten forms use generated module fields and a source-free Spec boundary."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import JsonValue, ValidationError, create_model
from pydantic_core import PydanticCustomError
from zcu_tools.cfg_model import (
    BathResetSpec,
    ConfigBase,
    ConstWaveformCfg,
    ModuleSpec,
    NoneResetCfg,
    PulseReadoutSpec,
    PulseResetSpec,
    PulseSpec,
    TwoPulseResetSpec,
)
from zcu_tools.gui import form_adapter
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    Option,
)
from zcu_tools.gui.form_adapter import ModuleInline, ModuleInput
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)


class SelectionForm(FormModel):
    """Nominal view of the generated field in a minimal handwritten consumer."""

    selection: ModuleInput | None = None

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        del cls, env
        return {("selection", "name"): ("Q.pulses.first",)}


def make_field(
    family: object = ModuleSpec,
    *,
    nullable: bool = False,
    ref_family: object | None = None,
) -> form_adapter.ModuleInputField:
    field_type = getattr(form_adapter, "ModuleInputField", None)
    assert field_type is not None, "Public module input field generation is missing"
    return field_type(family, nullable=nullable, ref_family=ref_family)


def selection_model(field: form_adapter.ModuleInputField) -> type[SelectionForm]:
    class Consumer(SelectionForm):
        @classmethod
        def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
            del cls
            return {("selection",): field.options(env)}

    return create_model(
        "GeneratedSelection", __base__=Consumer, selection=(field.annotation, ...)
    )


def selection_result(
    state: FormState[SelectionForm], env: FormEnv
) -> ModuleInput | None:
    result = state.result(env)
    if isinstance(result, FormErrors):
        pytest.fail(str(result.messages))
    return result.selection


def persistent_spec(value: str | ModuleSpec | None) -> ModuleSpec:
    if value is None or isinstance(value, str):
        pytest.fail(f"Expected a persistent Spec, got {value!r}")
    return value


def library_env() -> FormEnv:
    pulse = PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.2, waveform="Q.waveforms.const")
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q": ComponentLibrary(
                        pulses={"first": pulse, "second": pulse},
                        reset={"none": NoneResetCfg()},
                    )
                },
            )
        )
    )


@pytest.mark.parametrize(
    ("content", "spec_type", "child", "expected"),
    [
        (
            {
                "type": "pulse",
                "ch": 0,
                "nqz": 1,
                "freq": 100.0,
                "gain": 0.2,
                "waveform": {"kind": "ref", "name": "Q.waveforms.const"},
            },
            PulseSpec,
            "waveform",
            "Q.waveforms.const",
        ),
        (
            {
                "type": "readout/pulse",
                "gen_ch": 1,
                "nqz": 1,
                "freq": 100.0,
                "gain": 0.2,
                "ro_ch": 0,
                "ro_length": 1.0,
                "waveform": {"kind": "ref", "name": "Q.waveforms.const"},
            },
            PulseReadoutSpec,
            "waveform",
            "Q.waveforms.const",
        ),
        ({"type": "reset/none"}, NoneResetCfg, "type", "reset/none"),
        (
            {
                "type": "reset/pulse",
                "pulse_cfg": {"kind": "ref", "name": "Q.pulses.first"},
            },
            PulseResetSpec,
            "pulse_cfg",
            "Q.pulses.first",
        ),
        (
            {
                "type": "reset/two_pulse",
                "pulse1_cfg": {"kind": "ref", "name": "Q.pulses.first"},
                "pulse2_cfg": {"kind": "ref", "name": "Q.pulses.second"},
            },
            TwoPulseResetSpec,
            "pulse2_cfg",
            "Q.pulses.second",
        ),
        (
            {
                "type": "reset/bath",
                "cavity_tone_cfg": {"kind": "ref", "name": "Q.pulses.first"},
                "qubit_tone_cfg": {"kind": "ref", "name": "Q.pulses.second"},
                "pi2_cfg": {"kind": "ref", "name": "Q.pulses.first"},
            },
            BathResetSpec,
            "qubit_tone_cfg",
            "Q.pulses.second",
        ),
    ],
)
def test_full_family_inline_projects_detached_specs_without_sources(
    content: dict[str, JsonValue],
    spec_type: type[ConfigBase],
    child: str,
    expected: JsonValue,
) -> None:
    field = make_field(ref_family=PulseSpec)
    model = selection_model(field)
    env = FormEnv()
    state = FormState(
        FormInput(model, {"selection": {"kind": "inline", "module": content}})
    )
    before = state.input.data
    spec = persistent_spec(field.to_spec(selection_result(state, env)))
    assert type(spec) is spec_type
    assert spec.model_dump(mode="json")[child] == expected
    assert state.input.data == before
    if isinstance(spec, PulseSpec):
        spec.gain = 0.5
        assert (
            persistent_spec(field.to_spec(selection_result(state, env))).model_dump()[
                "gain"
            ]
            == 0.2
        )


def test_ref_family_only_limits_captured_keys_and_preserves_local_choices() -> None:
    field = make_field(ref_family=PulseSpec)
    model = selection_model(field)
    env = library_env()
    options = field.options(env)
    assert [
        o.label
        for o in options
        if isinstance(o.value, dict) and o.value.get("kind") == "ref"
    ] == ["Q.pulses.first", "Q.pulses.second"]
    unrestricted_inline_labels = [
        o.label for o in make_field().options(env) if o.label.startswith("Inline")
    ]
    assert unrestricted_inline_labels
    assert [
        o.label for o in options if o.label.startswith("Inline")
    ] == unrestricted_inline_labels
    state = FormState(
        FormInput(model, {"selection": {"kind": "ref", "name": "Q.pulses.second"}})
    )
    assert field.to_spec(selection_result(state, env)) == "Q.pulses.second"
    # The generated ref's most-local choices win over the parent's narrower list.
    state.write({("selection",): {"kind": "ref", "name": "Q.reset.none"}}, env)
    errors = state.result(env)
    assert isinstance(errors, FormErrors)
    assert ("selection", "name") in errors.messages
    assert (
        "choices" in errors.messages[("selection", "name")].lower()
        or "listed" in errors.messages[("selection", "name")].lower()
    )
    before = state.input.data
    base = state.base.data
    notifications: list[str] = []
    state.subscribe(lambda: notifications.append("changed"))
    with pytest.raises(PydanticCustomError, match=r"selection\.name"):
        state.write({("selection",): {"kind": "ref", "name": "Q.pulses.missing"}}, env)
    assert state.input.data == before
    assert state.base.data == base
    assert notifications == []
    missing_state = FormState(
        FormInput(model, {"selection": {"kind": "ref", "name": "Q.pulses.missing"}})
    )
    missing = missing_state.result(env)
    assert isinstance(missing, FormErrors)
    assert ("selection", "name") in missing.messages


@pytest.mark.parametrize("nullable", [False, True])
def test_nullability_and_foreign_value_fail_at_the_public_boundary(
    nullable: bool,
) -> None:
    field = make_field(nullable=nullable)
    model = selection_model(field)
    env = FormEnv()
    state = FormState(FormInput(model, {"selection": None}))
    if nullable:
        assert selection_result(state, env) is None
        assert field.to_spec(None) is None
        assert field.options(env)[0].value is None
    else:
        assert isinstance(state.result(env), FormErrors)
        with pytest.raises(TypeError, match="None|nullable"):
            field.to_spec(None)
    foreign = make_field(nullable=nullable)
    foreign_model = selection_model(foreign)
    foreign_state = FormState(
        FormInput(
            foreign_model,
            {"selection": {"kind": "inline", "module": {"type": "reset/none"}}},
        )
    )
    with pytest.raises(TypeError, match="field|generated"):
        field.to_spec(selection_result(foreign_state, env))


def test_spec_conversion_revalidates_without_mutating_ordinary_result() -> None:
    field = make_field(PulseSpec)
    model = selection_model(field)
    state = FormState(
        FormInput(
            model,
            {
                "selection": {
                    "kind": "inline",
                    "module": {
                        "type": "pulse",
                        "ch": 0,
                        "nqz": 1,
                        "freq": 100.0,
                        "gain": 0.2,
                        "waveform": {"kind": "ref", "name": "Q.waveforms.const"},
                    },
                }
            },
        )
    )
    value = selection_result(state, FormEnv())
    assert isinstance(value, ModuleInline)
    invalid = value.model_copy(
        update={"module": value.module.model_copy(update={"nqz": 3})}
    )
    before = invalid.model_dump(mode="json")
    with pytest.raises(ValidationError):
        field.to_spec(invalid)
    assert invalid.model_dump(mode="json") == before


class UnreadableLibrary(LibrarySnapshot):
    def module_spec(self, path: str) -> ModuleSpec:
        del path
        raise OSError("Cannot read captured module")


def test_copy_command_is_atomic_and_uses_existing_supported_spec_projection() -> None:
    field = make_field(ref_family=PulseSpec)
    model = selection_model(field)
    env = library_env()
    state = FormState(
        FormInput(model, {"selection": {"kind": "ref", "name": "Q.pulses.first"}})
    )
    notifications: list[str] = []
    state.subscribe(lambda: notifications.append("changed"))
    before = state.input.data
    assert env.library is not None
    broken = FormEnv(library=UnreadableLibrary(env.library.document()))
    with pytest.raises(OSError, match="Cannot read captured module"):
        state.run_command(("selection",), "to_inline", broken)
    assert state.input.data == before
    assert notifications == []
    state.run_command(("selection",), "to_inline", env)
    assert notifications == ["changed"]
    spec = field.to_spec(selection_result(state, FormEnv()))
    assert isinstance(spec, PulseSpec)
    assert spec.waveform == "Q.waveforms.const"
    assert spec.gain == 0.2


@pytest.mark.parametrize("family", [(), str, ConstWaveformCfg, PulseSpec | None])
def test_invalid_family_declarations_fail_when_building_a_field(family: object) -> None:
    field_type = type(make_field())
    with pytest.raises((TypeError, ValueError), match="family|Spec|nullable"):
        field_type(family)


def test_ref_family_must_be_a_nonempty_subset_of_inline_family() -> None:
    field_type = type(make_field())
    with pytest.raises((TypeError, ValueError), match="family|Spec"):
        field_type(PulseSpec, ref_family=PulseResetSpec)
    with pytest.raises((TypeError, ValueError), match="family|Spec"):
        field_type(PulseSpec, ref_family=())


def test_flat_readout_ref_conversion_keeps_captured_channels_and_waveform_key() -> None:
    source = PulseReadoutSpec(
        gen_ch=2,
        nqz=1,
        freq=100.0,
        gain=0.2,
        waveform="Q.waveforms.const",
        ro_ch=3,
        ro_length=1.0,
        desc="captured",
    )
    env = FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={"Q": ComponentLibrary(readout={"probe": source})},
            )
        )
    )
    field = make_field(PulseReadoutSpec)
    model = selection_model(field)
    state = FormState(
        FormInput(
            model,
            {
                "selection": {"kind": "ref", "name": "Q.readout.probe"},
            },
        )
    )
    assert field.to_spec(selection_result(state, env)) == "Q.readout.probe"

    state.run_command(("selection",), "to_inline", env)
    state.write({("selection", "module", "ro_ch"): 4}, env)
    spec = persistent_spec(field.to_spec(selection_result(state, FormEnv())))

    assert isinstance(spec, PulseReadoutSpec)
    assert (spec.gen_ch, spec.ro_ch, spec.freq) == (2, 4, 100.0)
    assert spec.waveform == "Q.waveforms.const"
    assert spec.desc is None
    assert source.ro_ch == 3
