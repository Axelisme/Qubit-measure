"""Dispersive readout model — physical quantities to complex IQ signal.

This module converts a qubit's excited-state population ``P_e`` (plus flux and
readout probe frequency) into a complex IQ readout signal, the way an
accumulated (averaged) measurement reads out a fluxonium dispersively coupled
to a hanger resonator.

The accumulated signal is the population-weighted mixture of the two
state-conditioned resonator responses (see task_plans/mocksim/task_plan.md, P1-4):

    signal = S21(f_ro; rf_g) + P_e * [S21(f_ro; rf_e) - S21(f_ro; rf_g)]

This is the physically correct form for an *averaged* readout: the resonator
sits at ``rf_g`` when the qubit is in |g> and at ``rf_e`` when in |e>, and the
average over many shots with excited-state probability ``P_e`` is the linear
blend above. (Per-shot Bernoulli sampling is a Phase 2 concern; this module is
purely the deterministic physics layer.)

Responsibility boundary: this file only maps *physical quantities -> IQ signal*.
It does not touch sweeps, timelines, acc_buf assembly, or noise — those belong
to the lowering / engine layers. All resonator / dispersive / hanger physics is
delegated to the existing building blocks; no physics is re-implemented here.

The only deviation from the repo-wide Fast-Fail principle is the
``DressedLabelingError`` fallback in :func:`resonator_freqs` — see Q3 in
task_plan.md: a real measurement never raises at a physics edge, so the model
degrades to "no dispersive shift" deterministically and logs a warning instead
of crashing.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.fluxonium.dispersive import (
    DressedLabelingError,
    calculate_dispersive_vs_flux_fast,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor
from zcu_tools.utils.fitting.resonance.hanger import HangerModel

from .params import SimParams

# Decorative hanger parameters with no physical counterpart in SimParams. A flat
# unit response keeps the model focused on the resonance line shape itself:
#   phi    : 0.0  -> symmetric (ideal) line, no impedance-mismatch rotation.
#   a0     : 1.0  -> unit off-resonant transmission.
#   edelay : 0.0  -> no cable delay phase ramp.
_DEFAULT_PHI: float = 0.0
_DEFAULT_A0: complex = 1.0 + 0.0j
_DEFAULT_EDELAY: float = 0.0


def value_to_flux(sim: SimParams, device_value: float) -> float:
    """Convert a flux-device setting value to reduced flux (in units of Phi_0).

    Delegates the flux alignment to ``FluxoniumPredictor.value_to_flux`` so the
    exact same affine map (flux_bias / flux_half / flux_period) used everywhere
    else in the codebase is applied here too — no alignment formula is rewritten.
    """
    predictor = FluxoniumPredictor(
        params=(sim.EJ, sim.EC, sim.EL),
        flux_half=sim.flux_half,
        flux_period=sim.flux_period,
        flux_bias=sim.flux_bias,
    )
    return predictor.value_to_flux(device_value)


def resonator_freqs(sim: SimParams, flux: float) -> tuple[float, float]:
    """Return ``(rf_g, rf_e)`` resonator frequencies (GHz) at the given flux.

    ``rf_g`` / ``rf_e`` are the dressed resonator frequencies when the qubit is
    in its ground / excited state, computed by
    ``calculate_dispersive_vs_flux_fast`` (the scqubits-free fast path).

    Q3 fallback (task_plan.md): the fast labeling can raise
    ``DressedLabelingError`` at flux points where the dressed-state labeling is
    ambiguous (coupling too strong / levels too dense). A real measurement never
    raises at such a physics edge, so instead of propagating the error this
    degrades deterministically to "no dispersive shift" (``rf_g = rf_e =
    bare_rf``) and emits a warning naming the flux value. The degradation is
    deterministic and logged — never a silent random value.
    """
    try:
        rf_g_arr, rf_e_arr = calculate_dispersive_vs_flux_fast(
            (sim.EJ, sim.EC, sim.EL),
            np.array([flux], dtype=np.float64),
            sim.bare_rf,
            sim.g,
        )
    except DressedLabelingError:
        warnings.warn(
            f"dispersive labeling ambiguous at flux={flux:.4f}; "
            f"falling back to no dispersive shift (rf_g = rf_e = "
            f"bare_rf = {sim.bare_rf} GHz)",
            stacklevel=2,
        )
        return (sim.bare_rf, sim.bare_rf)

    return (float(rf_g_arr[0]), float(rf_e_arr[0]))


def s21(
    sim: SimParams, freqs: NDArray[np.float64], rf: float
) -> NDArray[np.complex128]:
    """Complex S21 of the hanger resonator at frequencies ``freqs`` (GHz).

    ``rf`` is the resonance frequency (rf_g or rf_e). The line shape is provided
    by ``HangerModel.calc_signals``; decorative parameters (phi / a0 / edelay)
    use the flat defaults documented at module top.

    Qc is derived from SimParams.Ql and SimParams.Qi via the hanger relation
    ``1/Qc = 1/Ql - 1/Qi``.  The dip depth equals ``1 - Ql/Qi``; a large Qi
    (Qi >> Ql) gives a deep dip approaching the coupling-limited ideal, while
    Qi close to Ql yields a shallow dip.  SimParams.Qi > Ql is enforced at
    construction time, guaranteeing Qc > 0.
    """
    Qc = 1.0 / (1.0 / sim.Ql - 1.0 / sim.Qi)
    return HangerModel.calc_signals(
        freqs=np.asarray(freqs, dtype=np.float64),
        freq=rf,
        Ql=sim.Ql,
        Qc=Qc,
        phi=_DEFAULT_PHI,
        a0=_DEFAULT_A0,
        edelay=_DEFAULT_EDELAY,
    )


def mixed_signal(
    sim: SimParams,
    freqs: NDArray[np.float64],
    flux: float,
    p_e: float,
) -> NDArray[np.complex128]:
    """Population-weighted accumulated readout signal at ``freqs`` (GHz).

    Assembles ``S21(rf_g) + p_e * [S21(rf_e) - S21(rf_g)]`` where ``rf_g`` /
    ``rf_e`` come from :func:`resonator_freqs` at the given ``flux``.

    ``freqs`` are the readout probe frequencies — an array swept by onetone, or
    a length-1 array holding the single fixed ``f_ro`` for twotone / time-domain
    experiments. ``p_e`` is the excited-state population, which must lie in
    ``[0, 1]`` (Fast-Fail).
    """
    if not 0.0 <= p_e <= 1.0:
        raise ValueError(f"p_e must be in [0, 1], got {p_e}")

    rf_g, rf_e = resonator_freqs(sim, flux)
    s_g = s21(sim, freqs, rf_g)
    s_e = s21(sim, freqs, rf_e)
    return s_g + p_e * (s_e - s_g)
