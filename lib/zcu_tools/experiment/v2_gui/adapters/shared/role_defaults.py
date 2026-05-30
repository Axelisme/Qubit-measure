"""Role-named default value helpers (thin wrappers over the generic factories).

Each wrapper encodes one notebook role's consistent fallback strategy in its
name, so adapters read like the notebook (default_pi / default_pi2 / ...) instead
of passing preferred_names lists. They delegate to the two-layer
make_*_default / make_*_ref_default helpers — no duplicated construction.

By contract these produce *value* defaults only and **never lock a field**;
locking (e.g. a probe's freq fixed to 0.0 because a sweep axis drives it) is the
adapter's job in cfg_spec() via spec.lock_literal(). See ADR-0009.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Literal, Optional, overload

from zcu_tools.gui.adapter import ModuleRefValue, WaveformRefValue

from .module_ref_defaults import (
    make_pulse_ref_default,
    make_readout_ref_default,
    make_reset_ref_default,
)
from .module_value_defaults import (
    make_pulse_default,
    make_pulse_readout_default,
    make_res_pulse_default,
    make_waveform_default,
)

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext

# Role preferred-name chains (library lookup order before fallback).
_PI_NAMES = ["pi_amp", "pi_len"]
_PI2_NAMES = ["pi2_amp", "pi2_len", "pi_amp", "pi_len"]


def default_pi(ctx: ExpContext) -> ModuleRefValue:
    """π pulse: prefer library pi_amp → pi_len, else blank pulse default."""
    return make_pulse_ref_default(ctx, _PI_NAMES)


def default_pi2(ctx: ExpContext) -> ModuleRefValue:
    """π/2 pulse: prefer pi2_amp → pi2_len, then degrade to pi, else blank."""
    return make_pulse_ref_default(ctx, _PI2_NAMES)


def default_qub_probe(ctx: ExpContext) -> ModuleRefValue:
    """Qubit probe pulse: a fresh blank qubit pulse (qub_ch / q_f). freq usually
    locked by the adapter because a sweep axis drives it. Does not reuse a
    calibrated pi pulse."""
    return make_pulse_default(ctx)


def default_res_probe(ctx: ExpContext) -> ModuleRefValue:
    """Resonator probe pulse: a fresh blank resonator-side pulse (res_ch / r_f),
    **no ro_cfg** (e.g. CKP res_pulse, AC-Stark stark_pulse1)."""
    return make_res_pulse_default(ctx)


def default_probe_readout(ctx: ExpContext) -> ModuleRefValue:
    """Inline blank readout (pulse + ro_cfg), not a library reference. Used by
    spectroscopy where the readout is not yet calibrated."""
    return make_pulse_readout_default(ctx)


@overload
def default_readout(
    ctx: ExpContext, *, optional: Literal[False] = ...
) -> ModuleRefValue: ...


@overload
def default_readout(
    ctx: ExpContext, *, optional: Literal[True]
) -> Optional[ModuleRefValue]: ...


def default_readout(
    ctx: ExpContext, *, optional: bool = False
) -> Optional[ModuleRefValue]:
    """Calibrated readout: reference a library readout (readout_dpm / readout_rf),
    else fall back to a blank pulse-readout. Used by post-calibration experiments
    (t1 / t2 / rabi)."""
    return make_readout_ref_default(ctx, optional=optional)


def default_waveform(ctx: ExpContext) -> WaveformRefValue:
    """Qubit-pulse waveform: reference a library waveform (qub_flat / qub_cos),
    else a blank cosine. Goes into a pulse's ``waveform`` sub-field."""
    return make_waveform_default(ctx)


@overload
def default_reset(
    ctx: ExpContext, *, optional: Literal[False] = ...
) -> ModuleRefValue: ...


@overload
def default_reset(
    ctx: ExpContext, *, optional: Literal[True]
) -> Optional[ModuleRefValue]: ...


def default_reset(
    ctx: ExpContext, *, optional: bool = False
) -> Optional[ModuleRefValue]:
    """Reset module: prefer library reset entries, else blank (often optional)."""
    return make_reset_ref_default(ctx, optional=optional)
