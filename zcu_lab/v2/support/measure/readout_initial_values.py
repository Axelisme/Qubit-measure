"""Shared starting values for readout length sweeps."""

from dataclasses import dataclass
from math import isfinite

from zcu_tools.cfg_model import PulseReadoutCfg, Slot, Template
from zcu_tools.gui.form_adapter import FormContext

from zcu_lab.templates import TEMPLATES


@dataclass(frozen=True)
class ReadoutLengthMarginSeed:
    """Starting length difference and source for an editable sweep field."""

    value: float | None
    """Pulse duration minus ADC window in µs, or None if unresolved."""
    source: str
    """Description of the captured slot or template and its source fields."""


def pulse_readout_length_margin(
    ctx: FormContext, readout: Slot | Template
) -> ReadoutLengthMarginSeed:
    """Read pulse duration minus ADC window from the selected readout source.

    ctx is the fixed Open/Reset capture. Pass the original Slot or lab Template
    before overriding lengths for the sweep. A Slot requires a capture and
    resolves its key there. The selected Template uses the lab catalog's
    authored scalar seeds and its own overrides. Either source returns None
    for unresolved or non-finite lengths, without substituting another source.
    Slot key, template ID and capture failures propagate. No sources are edited.
    """
    if isinstance(readout, Slot):
        if ctx.parameters is None:
            raise ValueError("Readout Slot requires captured parameters")
        library = ctx.parameters.library
        key = ctx.parameters.library_key(readout.path)
        source = f"Captured {readout.path}: waveform.length − ro_length"
        try:
            module = library.get_module(key)
            if not isinstance(module, PulseReadoutCfg):
                return ReadoutLengthMarginSeed(None, source)
            pulse, window = module.waveform.length, module.ro_length
        except ValueError:
            # Unresolvable waveform/cfg leaves the margin blank, not a template value.
            return ReadoutLengthMarginSeed(None, source)
    else:
        source = f"{readout.id} template: waveform.length − ro_length"
        seeds = TEMPLATES[readout.id].seeds
        pulse = readout.set.get("waveform.length", seeds.get("waveform.length"))
        window = readout.set.get("ro_length", seeds.get("ro_length"))

    if (
        isinstance(pulse, (int, float))
        and not isinstance(pulse, bool)
        and isinstance(window, (int, float))
        and not isinstance(window, bool)
    ):
        margin = float(pulse) - float(window)
        if isfinite(margin):
            return ReadoutLengthMarginSeed(margin, source)
    return ReadoutLengthMarginSeed(None, source)
