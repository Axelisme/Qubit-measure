"""Rabi starting values from the fixed, validated persistent library capture."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

from zcu_tools.cfg_model import PulseSpec, Slot, Template

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

    Reads validated Specs, whose waveform length is required and non-null,
    not pre-validation raw source data. Alias, malformed
    or dangling references, source and library errors propagate unchanged.
    A non-pulse x180 raises TypeError. Nothing is refreshed or written back.
    """
    if parameters is None:
        return Template("qub_probe", set={"gain": 1.0}), 1.0
    pulse = _x180_spec(parameters)
    if pulse is None:
        return Template("qub_probe", set={"gain": 1.0}), 1.0
    waveform = (
        parameters.library.waveform_spec(pulse.waveform)
        if isinstance(pulse.waveform, str)
        else pulse.waveform
    )
    pi_len = waveform.length
    if (
        isinstance(pi_len, bool)
        or not isinstance(pi_len, (int, float))
        or not isfinite(pi_len)
        or pi_len <= 0
    ):
        stop = None
    else:
        stop = 4 * pi_len
    return Slot("$qubit.module.x180"), stop


def _x180_spec(parameters: ParameterContext) -> PulseSpec | None:
    source = parameters.lookup("$qubit.module.x180")
    if not source.present or source.value is None:
        return None
    key = parameters.library_key("$qubit.module.x180")
    pulse = parameters.library.module_spec(key)
    if not isinstance(pulse, PulseSpec):
        raise TypeError("Rabi x180 must reference a PulseSpec")
    return pulse
