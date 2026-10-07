"""Pure typed rewriting of persistent module and waveform child references."""

from collections.abc import Mapping

from .materialization import module_spec_from_raw, waveform_spec_from_raw
from .module import (
    BathResetSpec,
    ModuleSpec,
    PulseReadoutSpec,
    PulseResetSpec,
    PulseSpec,
    TwoPulseResetSpec,
)
from .waveform import FlatTopWaveformSpec, WaveformSpec


def remap_waveform_references(
    spec: WaveformSpec, *, waveforms: Mapping[str, str]
) -> WaveformSpec:
    """Copy a persistent waveform Spec and rewrite its string child references.

    spec must satisfy waveform_spec_from_raw. waveforms maps literal legacy
    waveform keys to explicit destination keys; no canonical-key fallback occurs.
    Only FlatTop's string ramp is a reference. Inline ramps and Arb asset keys
    retain their values. Input models and mappings are never mutated.
    Missing reference mappings raise KeyError. Invalid persistent inputs or
    rewritten values raise ValueError; no resolver, assets or I/O are used.
    """
    candidate = waveform_spec_from_raw(spec)
    if isinstance(candidate, FlatTopWaveformSpec) and isinstance(
        candidate.raise_waveform, str
    ):
        candidate.raise_waveform = waveforms[candidate.raise_waveform]
    return waveform_spec_from_raw(candidate)


def _remap_pulse(spec: PulseSpec, waveforms: Mapping[str, str]) -> PulseSpec:
    candidate = spec.model_copy(deep=True)
    if isinstance(candidate.waveform, str):
        candidate.waveform = waveforms[candidate.waveform]
    else:
        candidate.waveform = remap_waveform_references(
            candidate.waveform, waveforms=waveforms
        )
    return candidate


def _pulse_child(
    child: str | PulseSpec,
    modules: Mapping[str, str],
    waveforms: Mapping[str, str],
) -> str | PulseSpec:
    return modules[child] if isinstance(child, str) else _remap_pulse(child, waveforms)


def remap_module_references(
    spec: ModuleSpec,
    *,
    modules: Mapping[str, str],
    waveforms: Mapping[str, str],
) -> ModuleSpec:
    """Copy a persistent module Spec and rewrite typed string child references.

    spec must satisfy module_spec_from_raw. modules and waveforms map literal
    legacy keys within their respective source partitions to explicit new keys.
    Pulse waveform references use waveforms; readout/reset pulse references use
    modules. Inline children follow the same typed schema recursively. Scalar
    fields, description strings and Arb asset keys remain unchanged.
    No canonical-key fallback, evaluation, lookup or I/O occurs.
    Input models and mappings remain independent of the result. Missing mappings
    raise KeyError; invalid persistent inputs or rewritten values raise ValueError.
    """
    candidate = module_spec_from_raw(spec)
    if isinstance(candidate, PulseSpec):
        candidate = _remap_pulse(candidate, waveforms)
    elif isinstance(candidate, (PulseReadoutSpec, PulseResetSpec)):
        candidate.pulse_cfg = _pulse_child(candidate.pulse_cfg, modules, waveforms)
    elif isinstance(candidate, TwoPulseResetSpec):
        candidate.pulse1_cfg = _pulse_child(candidate.pulse1_cfg, modules, waveforms)
        candidate.pulse2_cfg = _pulse_child(candidate.pulse2_cfg, modules, waveforms)
    elif isinstance(candidate, BathResetSpec):
        candidate.cavity_tone_cfg = _pulse_child(
            candidate.cavity_tone_cfg, modules, waveforms
        )
        candidate.qubit_tone_cfg = _pulse_child(
            candidate.qubit_tone_cfg, modules, waveforms
        )
        candidate.pi2_cfg = _pulse_child(candidate.pi2_cfg, modules, waveforms)
    return module_spec_from_raw(candidate)
