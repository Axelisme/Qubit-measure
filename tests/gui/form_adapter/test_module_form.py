"""Persistent module Specs generate editable forms that convert back to Specs."""

from collections.abc import Mapping

import pytest
from pydantic import JsonValue, create_model
from zcu_tools.cfg_model import (
    ConfigBase,
    DirectReadoutCfg,
    ModuleSpec,
    PulseReadoutSpec,
    PulseSpec,
    WaveformSpec,
)
from zcu_tools.cfg_model import PulseCfg as MaterializedPulseCfg
from zcu_tools.gui import form_adapter
from zcu_tools.gui.form import FormEnv, FormErrors, FormModel, FormState, ModuleInline
from zcu_tools.program.v2.modules import PulseCfg as NativePulseCfg

UNSUPPORTED_PULSE_FIELDS = (
    "desc",
    "mux_freqs",
    "mux_gains",
    "mux_phases",
    "mask",
    "outsel",
    "ro_ch",
)

PULSE_LEAVES: dict[tuple[str | int, ...], JsonValue] = {
    ("ch",): 2,
    ("nqz",): 1,
    ("freq",): 100.0,
    ("gain",): 0.25,
}


class SyntheticCfg(ConfigBase):
    """An ordinary cfg is not a persistent module or waveform Spec."""

    value: float = 1.0


def result_spec[S: ConfigBase](
    form: form_adapter.Form[S], state: FormState[FormModel], env: FormEnv
) -> S:
    result = state.result(env)
    if isinstance(result, FormErrors):
        pytest.fail(str(result.messages))
    cfg = form.to_cfg(result, env)
    if isinstance(cfg, FormErrors):
        pytest.fail(str(cfg.messages))
    return cfg


@pytest.mark.parametrize(
    "spec_type",
    [ModuleSpec, WaveformSpec, MaterializedPulseCfg, NativePulseCfg, SyntheticCfg],
    ids=["module-union", "waveform-union", "materialized", "native", "synthetic"],
)
def test_module_form_rejects_non_persistent_spec_roots(
    spec_type: type[ConfigBase],
) -> None:
    with pytest.raises(TypeError):
        form_adapter.module_form(spec_type)


def test_spec_form_evaluates_input_and_keeps_child_library_key() -> None:
    form = form_adapter.module_form(PulseSpec)
    env = FormEnv()
    state = FormState(form.default_input(env))
    state.write(
        {**PULSE_LEAVES, ("gain",): {"expr": "0.5 / 2"}, ("waveform",): "Q.flat"}, env
    )

    cfg = result_spec(form, state, env)

    assert type(cfg) is PulseSpec
    assert cfg.gain == 0.25
    assert cfg.waveform == "Q.flat"


@pytest.mark.parametrize("name", UNSUPPORTED_PULSE_FIELDS)
def test_unsupported_pulse_fields_cannot_be_written_and_use_spec_defaults(
    name: str,
) -> None:
    form = form_adapter.module_form(PulseSpec)
    env = FormEnv()
    state = FormState(form.default_input(env))
    properties = form.model.input_schema(env)["properties"]

    with pytest.raises(ValueError, match="Unknown field path"):
        state.write({(name,): 1}, env)
    state.write({**PULSE_LEAVES, ("waveform",): "Q.flat"}, env)
    cfg = result_spec(form, state, env)

    assert isinstance(properties, dict) and name not in properties
    assert getattr(cfg, name) == PulseSpec.model_fields[name].default


def test_required_direct_readout_channel_stays_settable() -> None:
    form = form_adapter.module_form(PulseReadoutSpec)
    env = FormEnv()
    state = FormState(form.default_input(env))

    state.write({("pulse_cfg",): {}}, env)
    with pytest.raises(ValueError, match="Unknown field path"):
        state.write({("pulse_cfg", "ro_ch"): 4}, env)
    state.write(
        {
            ("pulse_cfg",): "Q.probe",
            ("ro_cfg", "ro_ch"): 3,
            ("ro_cfg", "ro_length"): 1.5,
            ("ro_cfg", "ro_freq"): 6000.0,
        },
        env,
    )
    cfg = result_spec(form, state, env)

    assert cfg.pulse_cfg == "Q.probe"
    assert cfg.ro_cfg.ro_ch == 3


def library_entry() -> PulseSpec:
    """A stored entry sets unsupported fields that the Form cannot edit."""
    return PulseSpec.model_validate(
        {
            "ch": 2,
            "nqz": 1,
            "freq": 100.0,
            "gain": 0.25,
            "desc": "stored entry",
            "mask": [0],
            "outsel": 1,
            "ro_ch": 5,
            "waveform": "Q.flat",
        }
    )


def writeback_proposal() -> PulseReadoutSpec:
    """A proposal keeps its child key and sets unsupported nested pulse fields."""
    return PulseReadoutSpec.model_validate(
        {
            "desc": "proposal",
            "pulse_cfg": {
                "ch": 0,
                "nqz": 2,
                "freq": 6000.0,
                "gain": 0.1,
                "mux_freqs": [6000.0],
                "ro_ch": 3,
                "waveform": "Q.square",
            },
            "ro_cfg": {"ro_ch": 3, "ro_length": 1.0, "ro_freq": 6000.0},
        }
    )


@pytest.mark.parametrize(
    ("spec", "unsupported"),
    [
        (library_entry(), {("desc",), ("mask",), ("outsel",), ("ro_ch",)}),
        (
            writeback_proposal(),
            {("desc",), ("pulse_cfg", "mux_freqs"), ("pulse_cfg", "ro_ch")},
        ),
    ],
    ids=["library-entry", "writeback-proposal"],
)
def test_existing_spec_round_trips_except_unsupported_values(
    spec: PulseSpec | PulseReadoutSpec, unsupported: set[tuple[str, ...]]
) -> None:
    form = form_adapter.module_form(type(spec))
    env = FormEnv()
    original = spec.model_dump(mode="json")

    state = FormState(form.input_from_spec(spec, env))
    cfg = result_spec(form, state, env)

    assert spec.model_dump(mode="json") == original
    assert not any(path[-1] == "mixer_freq" for path in leaves(state.input.data))
    changed = {
        path
        for path in leaves(original) | leaves(cfg.model_dump(mode="json"))
        if lookup(original, path) != lookup(cfg.model_dump(mode="json"), path)
    }
    assert changed == unsupported


def leaves(data: Mapping[str, JsonValue]) -> set[tuple[str, ...]]:
    paths: set[tuple[str, ...]] = set()
    for name, value in data.items():
        if isinstance(value, dict):
            paths |= {(name, *child) for child in leaves(value)}
        else:
            paths.add((name,))
    return paths


def lookup(data: JsonValue, path: tuple[str, ...]) -> JsonValue:
    for part in path:
        data = data.get(part) if isinstance(data, dict) else None
    return data


class DerivedPulse(PulseSpec):
    """A subclass instance is not the Form's concrete persistent Spec."""


def test_input_from_spec_rejects_another_spec_type_or_core_form() -> None:
    env = FormEnv()
    derived = DerivedPulse.model_validate(library_entry().model_dump(mode="json"))
    direct = DirectReadoutCfg(ro_ch=1, ro_length=1.0, ro_freq=10.0)

    with pytest.raises(TypeError):
        form_adapter.module_form(PulseSpec).input_from_spec(derived, env)
    with pytest.raises(TypeError):
        form_adapter.Form(DirectReadoutCfg, {}).input_from_spec(direct, env)


class HostInline(ModuleInline):
    """A host offers its own channel choices for the nested generated Spec."""

    @classmethod
    def choices(
        cls, env: FormEnv
    ) -> Mapping[tuple[str | int, ...], tuple[JsonValue, ...]]:
        del env
        return {("module", "ch"): (2, 3)}


def test_generated_model_nests_in_module_inline_with_host_choices() -> None:
    form = form_adapter.module_form(PulseSpec)
    host = create_model("PulseInline", __base__=HostInline, module=(form.model, ...))
    env = FormEnv()
    state = FormState(host.default_input(env))
    leaves_batch = {("module", *path): value for path, value in PULSE_LEAVES.items()}

    state.write({**leaves_batch, ("module", "ch"): 7, ("module", "waveform"): "w"}, env)
    rejected = state.result(env)
    state.write({("module", "ch"): 3}, env)
    accepted = state.result(env)

    assert isinstance(rejected, FormErrors)
    assert set(rejected.messages) == {("module", "ch")}
    assert isinstance(accepted, HostInline) and isinstance(accepted.module, FormModel)
    cfg = form.to_cfg(accepted.module, env)
    assert isinstance(cfg, PulseSpec) and cfg.ch == 3 and cfg.waveform == "w"
