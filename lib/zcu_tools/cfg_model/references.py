"""Pure typed rewriting of persistent module and waveform child references."""

from collections.abc import Mapping

from .dependencies import ModuleResolver
from .materialization import module_spec_from_raw, waveform_spec_from_raw
from .module import (
    BathResetSpec,
    ModuleSpec,
    PulseReadoutSpec,
    PulseResetSpec,
    PulseSpec,
    TwoPulseResetSpec,
)
from .waveform import ArbWaveformCfg, FlatTopWaveformSpec, WaveformSpec


def module_arb_keys(spec: ModuleSpec, resolver: ModuleResolver) -> frozenset[str]:
    """Find arb asset keys in a persistent module and its resolvable typed children.

    resolver supplies the library whose child Specs would be used. Missing or
    invalid library references do not contribute keys; this is not full module
    materialization or validation. No asset is read and no model is changed.
    """
    if isinstance(spec, (PulseSpec, PulseReadoutSpec)):
        return _waveform_arb_keys(spec.waveform, resolver)

    match spec:
        case PulseResetSpec(pulse_cfg=pulse):
            children = (pulse,)
        case TwoPulseResetSpec(pulse1_cfg=first, pulse2_cfg=second):
            children = (first, second)
        case BathResetSpec(cavity_tone_cfg=cavity, qubit_tone_cfg=qubit, pi2_cfg=pi2):
            children = (cavity, qubit, pi2)
        case _:
            return frozenset()
    keys: set[str] = set()
    for child in children:
        if isinstance(child, str):
            try:
                child = resolver.module_spec(child)
            except (KeyError, ValueError):
                continue
        if isinstance(child, PulseSpec):
            keys.update(module_arb_keys(child, resolver))
    return frozenset(keys)


def _waveform_arb_keys(
    waveform: str | WaveformSpec, resolver: ModuleResolver
) -> frozenset[str]:
    try:
        if isinstance(waveform, str):
            waveform = resolver.waveform_spec(waveform)
        if isinstance(waveform, FlatTopWaveformSpec):
            ramp = waveform.raise_waveform
            waveform = resolver.waveform_spec(ramp) if isinstance(ramp, str) else ramp
    except (KeyError, ValueError):
        return frozenset()
    return (
        frozenset({waveform.data})
        if isinstance(waveform, ArbWaveformCfg)
        else frozenset()
    )


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


def _remap_module_waveform[T: PulseSpec | PulseReadoutSpec](
    spec: T, waveforms: Mapping[str, str]
) -> T:
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
    return (
        modules[child]
        if isinstance(child, str)
        else _remap_module_waveform(child, waveforms)
    )


def remap_module_references(
    spec: ModuleSpec,
    *,
    modules: Mapping[str, str],
    waveforms: Mapping[str, str],
) -> ModuleSpec:
    """Copy a persistent module Spec and rewrite typed string child references.

    spec must satisfy module_spec_from_raw. modules and waveforms map literal
    legacy keys within their respective source partitions to explicit new keys.
    Pulse/readout waveform references use waveforms; reset pulse references use
    modules. Inline children follow the same typed schema recursively. Scalar
    fields, description strings and Arb asset keys remain unchanged.
    No canonical-key fallback, evaluation, lookup or I/O occurs.
    Input models and mappings remain independent of the result. Missing mappings
    raise KeyError; invalid persistent inputs or rewritten values raise ValueError.
    """
    candidate = module_spec_from_raw(spec)
    if isinstance(candidate, (PulseSpec, PulseReadoutSpec)):
        candidate = _remap_module_waveform(candidate, waveforms)
    elif isinstance(candidate, PulseResetSpec):
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
