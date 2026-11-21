"""Shared starting values for readout length sweeps."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from zcu_tools.cfg_model import FieldMeta, PulseReadoutCfg, Slot, Template, TemplateDef
from zcu_tools.gui.form_adapter import FormContext

from zcu_lab.v2.support.measure.seed_metadata import seed_metadata


@dataclass(frozen=True)
class ReadoutLengthMarginSeed:
    """Starting length difference and source for an editable sweep field."""

    value: float | None
    """Pulse duration minus ADC window in µs, or None if unresolved."""
    metadata: FieldMeta
    """Source description; only a supplied fixed template value is uncalibrated."""


def pulse_readout_length_margin(
    ctx: FormContext, readout: Slot | Template, *, templates: Mapping[str, TemplateDef]
) -> ReadoutLengthMarginSeed:
    """Read pulse duration minus ADC window from the selected readout source.

    Pass the original Slot or Template before hiding lengths for the sweep.
    A Slot reads the fixed Open/Reset capture in ctx. A Template uses the
    supplied registry catalog's scalar seeds, with Template.set taking priority.
    Unknown template IDs raise ValueError, and Slot key and capture failures
    propagate. Unresolved or non-finite lengths leave the value blank instead
    of falling back to another source. Only a supplied template margin is
    marked uncalibrated. No sources are edited.
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
                return ReadoutLengthMarginSeed(None, seed_metadata(source))
            pulse, window = module.waveform.length, module.ro_length
        except ValueError:
            # Unresolvable waveform/cfg leaves the margin blank, not a template value.
            return ReadoutLengthMarginSeed(None, seed_metadata(source))
    else:
        source = f"{readout.id} template: waveform.length − ro_length"
        try:
            seeds = templates[readout.id].seeds
        except KeyError as exc:
            raise ValueError(f"Unknown template {readout.id!r}") from exc
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
            return ReadoutLengthMarginSeed(
                margin, seed_metadata(source, fallback=isinstance(readout, Template))
            )
    return ReadoutLengthMarginSeed(None, seed_metadata(source))
