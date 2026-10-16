"""Rabi starting values from the fixed, validated persistent library capture."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

from zcu_tools.cfg_model import ArbWaveformCfg, PulseSpec, Slot, Template, WaveformSpec

if TYPE_CHECKING:
    from zcu_tools.gui.session.parameters import ParameterContext


def len_rabi_initial_values(
    parameters: ParameterContext | None,
) -> tuple[Slot | Template, float | None]:
    """Return the Length Rabi pulse source and sweep stop in microseconds.

    parameters is the fixed Open/Reset capture, or None. A missing/null x180
    slot selects a fresh qub_probe Template with editable gain 1.0 and stop
    1.0 us. An existing x180 selects its Slot without changing its gain.
    Read pi_len from its persistent waveform, following a waveform library key
    without materializing the pulse. A finite positive non-bool pi_len gives
    stop=4*pi_len; an invalid stored pi_len gives None for a required blank.
    Arb has no persistent scalar length and uses stop=1.0 us without assets.

    Reads validated Specs, not pre-validation raw source data. Scalar waveform
    length is required and non-null at this boundary. Alias, malformed
    or dangling references, source and library errors propagate unchanged.
    A non-pulse x180 raises TypeError. Nothing is refreshed or written back.
    """
    if parameters is None:
        return Template("qub_probe", set={"gain": 1.0}), 1.0
    pulse = _x180_spec(parameters)
    if pulse is None:
        return Template("qub_probe", set={"gain": 1.0}), 1.0
    waveform = _waveform_spec(parameters, pulse)
    if isinstance(waveform, ArbWaveformCfg):
        return Slot("$qubit.module.x180"), 1.0
    pi_len = _valid_pi_len(waveform)
    stop = None if pi_len is None else 4 * pi_len
    return Slot("$qubit.module.x180"), stop


def amp_rabi_initial_pulse(parameters: ParameterContext | None) -> Slot | Template:
    """Return an editable Amplitude Rabi pulse from the fixed Open/Reset capture.

    Missing capture or missing/null x180 selects qub_probe with gain 1.0 and
    explicit waveform length 1.0 us. Arb x180 keeps its original Slot and data,
    without copying, reading assets or overriding its asset-derived length.
    For scalar waveforms, copy captured x180 and override only waveform.length
    with 1.5*pi_len*pi_gain in us. pi_len must be finite
    and positive; pi_gain must be finite and in (0, 1]. Neither accepts bool.
    Invalid stored values leave the required length blank, not a fallback.
    Only nonfinite gain also leaves gain blank so the copied pulse is editable
    as JSON input; both required fields must then be filled before submission.
    Finite invalid gain is preserved unchanged.

    Only validated Specs are read, following waveform keys without reading
    assets. Pre-validation rejected/coerced raw values are outside this
    boundary. Alias, source and library errors propagate; non-pulse x180
    raises TypeError. No source is refreshed, mutated or written back.
    """
    if parameters is None:
        return Template("qub_probe", set={"gain": 1.0, "waveform.length": 1.0})
    pulse = _x180_spec(parameters)
    if pulse is None:
        return Template("qub_probe", set={"gain": 1.0, "waveform.length": 1.0})
    waveform = _waveform_spec(parameters, pulse)
    if isinstance(waveform, ArbWaveformCfg):
        return Slot("$qubit.module.x180")
    pi_len = _valid_pi_len(waveform)
    pi_gain = pulse.gain
    if not isfinite(pi_gain):
        return Slot(
            "$qubit.module.x180", copy=True, set={"gain": None, "waveform.length": None}
        )
    if (
        pi_len is None
        or isinstance(pi_gain, bool)
        or not isinstance(pi_gain, (int, float))
        or not 0 < pi_gain <= 1
    ):
        length = None
    else:
        length = 1.5 * pi_len * pi_gain
    return Slot("$qubit.module.x180", copy=True, set={"waveform.length": length})


def _waveform_spec(parameters: ParameterContext, pulse: PulseSpec) -> WaveformSpec:
    return (
        parameters.library.waveform_spec(pulse.waveform)
        if isinstance(pulse.waveform, str)
        else pulse.waveform
    )


def _valid_pi_len(waveform: WaveformSpec) -> float | None:
    if isinstance(waveform, ArbWaveformCfg):
        raise TypeError("Arb waveform has no persistent scalar length")
    pi_len = waveform.length
    if (
        isinstance(pi_len, bool)
        or not isinstance(pi_len, (int, float))
        or not isfinite(pi_len)
        or pi_len <= 0
    ):
        return None
    return pi_len


def _x180_spec(parameters: ParameterContext) -> PulseSpec | None:
    source = parameters.lookup("$qubit.module.x180")
    if not source.present or source.value is None:
        return None
    key = parameters.library_key("$qubit.module.x180")
    pulse = parameters.library.module_spec(key)
    if not isinstance(pulse, PulseSpec):
        raise TypeError("Rabi x180 must reference a PulseSpec")
    return pulse
