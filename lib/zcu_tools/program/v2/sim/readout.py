"""Dispersive readout model — physical quantities to complex IQ signal.

This module converts physical quantities (flux + readout probe frequency, plus
either an excited-state population ``P_e`` or a chosen state) into a complex IQ
readout signal, the way a measurement reads out a fluxonium dispersively coupled
to a hanger resonator.

The two state-conditioned responses are the per-shot Bernoulli outcomes the
engine selects between: ``s21(f_ro; rf_g)`` when the qubit is in |g> and
``s21(f_ro; rf_e)`` when in |e>.  Their population-weighted mixture (see
.agent_state/plans/mocksim/task_plan.md, P1-4) is the *averaged* readout

    signal = S21(f_ro; rf_g) + P_e * [S21(f_ro; rf_e) - S21(f_ro; rf_g)]

which :func:`mixed_signal` provides for the accumulated (reps-averaged) path.
The per-shot Bernoulli sampling itself lives in the engine (it draws state ~
Bernoulli(P_e) per shot and picks ``s21`` of the chosen state); this module is
the deterministic readout layer — it returns the per-state ``s21`` blobs, their
blend, and pure readout integration / noise-scale helpers, never random draws.

Responsibility boundary: this file owns deterministic S21 plus pure readout
integration / noise-scale helpers. It does not touch sweeps, timelines,
``acc_buf`` assembly, random noise draws, or the per-shot Bernoulli draw — those
belong to the lowering / engine layers. All resonator / dispersive / hanger
physics is delegated to the existing building blocks; no physics is
re-implemented here.

The only deviation from the repo-wide Fast-Fail principle is the
``DressedLabelingError`` fallback in :func:`resonator_freqs` — see Q3 in
task_plan.md: a real measurement never raises at a physics edge, so the model
degrades to "no dispersive shift" deterministically and logs a warning instead
of crashing.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.simulate.fluxonium.dispersive import (
    DressedLabelingError,
    calculate_dispersive_vs_flux_fast,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor
from zcu_tools.utils.fitting.resonance.hanger import HangerModel

from .params import SimParams
from .waveforms import envelope_at

# Decorative hanger parameters with no physical counterpart in SimParams. A flat
# unit response keeps the model focused on the resonance line shape itself:
#   phi    : 0.0  -> symmetric (ideal) line, no impedance-mismatch rotation.
#   a0     : 1.0  -> unit off-resonant transmission.
#   edelay : 0.0  -> no cable delay phase ramp.
_DEFAULT_PHI: float = 0.0
_DEFAULT_A0: complex = 1.0 + 0.0j
_DEFAULT_EDELAY: float = 0.0

# Conservative cQED dispersive operating region.  Literature commonly treats
# nbar/ncrit ~ 0.1 as the small-parameter guardrail; below it the mock keeps the
# existing linear readout response.
_DISPERSIVE_SAFE_PHOTON_RATIO: float = 0.1


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


@lru_cache(maxsize=256)
def _cached_resonator_freqs(
    EJ: float,
    EC: float,
    EL: float,
    bare_rf: float,
    g: float,
    flux: float,
) -> tuple[float, float]:
    """Cached dressed resonator prediction for one physical operating point."""

    try:
        rf_g_arr, rf_e_arr = calculate_dispersive_vs_flux_fast(
            (EJ, EC, EL),
            np.array([flux], dtype=np.float64),
            bare_rf,
            g,
        )
    except DressedLabelingError:
        warnings.warn(
            f"dispersive labeling ambiguous at flux={flux:.4f}; "
            f"falling back to no dispersive shift (rf_g = rf_e = "
            f"bare_rf = {bare_rf} GHz)",
            stacklevel=3,
        )
        return (bare_rf, bare_rf)

    return (float(rf_g_arr[0]), float(rf_e_arr[0]))


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
    return _cached_resonator_freqs(
        sim.EJ,
        sim.EC,
        sim.EL,
        sim.bare_rf,
        sim.g,
        float(flux),
    )


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


def critical_photon_number(
    f_qubit_ghz: float,
    resonator_ghz: float,
    g_ghz: float,
) -> float:
    """Return the dispersive critical photon number ``Delta^2 / (4 g^2)``.

    The ratio is unitless, so GHz inputs are fine as long as ``Delta`` and ``g``
    use the same frequency basis.  The expression is only meaningful in the
    dispersive regime; non-positive coupling or zero detuning fast-fail because
    there is no finite safe photon budget to infer from the approximation.
    """

    f_qubit = float(f_qubit_ghz)
    resonator = float(resonator_ghz)
    coupling = float(g_ghz)
    if not all(math.isfinite(v) for v in (f_qubit, resonator, coupling)):
        raise ValueError(
            "f_qubit_ghz, resonator_ghz, and g_ghz must be finite "
            f"(got {f_qubit_ghz!r}, {resonator_ghz!r}, {g_ghz!r})"
        )
    if coupling <= 0.0:
        raise ValueError(f"g_ghz must be > 0.0, got {g_ghz!r}")

    detuning = abs(f_qubit - resonator)
    if detuning <= 0.0:
        raise ValueError(
            "critical photon number is undefined at zero qubit-resonator detuning"
        )
    return (detuning * detuning) / (4.0 * coupling * coupling)


def readout_photon_ratio(
    gain: float | None,
    n_crit: float,
    photons_per_gain2: float,
) -> float:
    """Return ``n_bar / n_crit`` for a PulseReadout gain.

    ``DirectReadout`` has no explicit drive gain (``gain is None``), so it stays
    on the safe linear path and contributes no nonlinear readout penalty.  A
    PulseReadout must pass an explicit gain² -> photon calibration.
    """

    if gain is None:
        return 0.0

    amplitude = readout_drive_amplitude(gain)
    critical = float(n_crit)
    if not math.isfinite(critical) or critical <= 0.0:
        raise ValueError(f"n_crit must be finite and > 0.0, got {n_crit!r}")

    if photons_per_gain2 is None:
        raise ValueError("photons_per_gain2 must be provided for PulseReadout")
    photon_scale = float(photons_per_gain2)
    if not math.isfinite(photon_scale) or photon_scale <= 0.0:
        raise ValueError(
            f"photons_per_gain2 must be finite and > 0.0 (got {photons_per_gain2!r})"
        )

    return (photon_scale * amplitude * amplitude) / critical


def readout_drive_amplitude(
    gain: float | None,
    *,
    n_crit: float | None = None,
    photons_per_gain2: float = 100.0,
) -> float:
    """Return the readout drive amplitude after dispersive-limit compression."""

    if gain is None:
        return 1.0
    amplitude = float(gain)
    if not math.isfinite(amplitude):
        raise ValueError(f"readout gain must be finite, got {gain!r}")
    if n_crit is None:
        return amplitude

    ratio = _readout_nonlinear_ratio(gain, n_crit, photons_per_gain2)
    return amplitude / math.sqrt(1.0 + ratio)


def readout_state_visibility(
    gain: float | None,
    n_crit: float,
    photons_per_gain2: float,
) -> float:
    """Return the remaining |g>/|e> readout contrast under high photon number."""

    ratio = _readout_nonlinear_ratio(gain, n_crit, photons_per_gain2)
    return 1.0 / (1.0 + ratio)


def _readout_nonlinear_ratio(
    gain: float | None,
    n_crit: float,
    photons_per_gain2: float,
) -> float:
    """Return the excess photon ratio above the conservative dispersive guardrail."""

    ratio = readout_photon_ratio(gain, n_crit, photons_per_gain2)
    return max(0.0, ratio - _DISPERSIVE_SAFE_PHOTON_RATIO)


def apply_readout_visibility(
    s_g: NDArray[np.complex128],
    s_e: NDArray[np.complex128],
    visibility: float,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Compress state-conditioned blob separation around its midpoint."""

    vis = float(visibility)
    if not math.isfinite(vis) or not 0.0 <= vis <= 1.0:
        raise ValueError(f"visibility must be finite and in [0, 1], got {visibility!r}")
    center = 0.5 * (s_g + s_e)
    return center + vis * (s_g - center), center + vis * (s_e - center)


def blend_state_responses(
    s_g: NDArray[np.complex128],
    s_e: NDArray[np.complex128],
    p_e: float,
) -> NDArray[np.complex128]:
    """Return the validated population-weighted blend of two state responses."""

    if not 0.0 <= p_e <= 1.0:
        raise ValueError(f"p_e must be in [0, 1], got {p_e}")
    return s_g + p_e * (s_e - s_g)


def _validate_sample_times(sample_times_us: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize and validate a readout integration sample axis."""

    ts = np.asarray(sample_times_us, dtype=np.float64)
    if ts.ndim != 1:
        raise ValueError(f"sample_times_us must be a 1-D array, got shape {ts.shape}")
    if ts.size == 0:
        raise ValueError("sample_times_us must not be empty")
    if not np.all(np.isfinite(ts)):
        raise ValueError("sample_times_us must contain only finite values")
    return ts


def readout_envelope_samples(
    readout_cfg: PulseCfg | None,
    pulse_length_us: float | None,
    sample_times_us: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the readout envelope sampled on the ADC integration axis.

    ``DirectReadout`` has no generator envelope and therefore contributes one
    unit per integrated sample.  ``PulseReadout`` returns the resolved generator
    envelope over the same compiled ADC sample axis.  The returned values are
    dimensionless and peak-normalized.
    """

    ts = _validate_sample_times(sample_times_us)
    if readout_cfg is None:
        if pulse_length_us is not None:
            raise ValueError("pulse_length_us must be None when readout_cfg is None")
        return np.ones(ts.shape, dtype=np.float64)

    if pulse_length_us is None:
        raise ValueError("pulse_length_us is required when readout_cfg is provided")
    pulse_length = float(pulse_length_us)
    if not math.isfinite(pulse_length) or pulse_length <= 0.0:
        raise ValueError(
            f"pulse_length_us must be finite and positive, got {pulse_length_us!r}"
        )

    return envelope_at(readout_cfg, ts, pulse_length)


def effective_signal_samples(
    readout_cfg: PulseCfg | None,
    pulse_length_us: float | None,
    sample_times_us: NDArray[np.float64],
) -> float:
    """Return the envelope-weighted signal area in compiled ADC samples.

    This helper returns a sample count, not a time integral; QICK's accumulated
    path normalizes by the compiled readout sample count.
    """

    envelope = readout_envelope_samples(readout_cfg, pulse_length_us, sample_times_us)
    return float(np.sum(envelope))


def effective_noise_samples(
    readout_cfg: PulseCfg | None,
    pulse_length_us: float | None,
    sample_times_us: NDArray[np.float64],
) -> float:
    """Return the envelope-weighted Gaussian-noise scale in ADC samples.

    This is the square-root of the sampled envelope power.  It is used for the
    readout-drive-proportional noise source: constant full-window readout gives
    ``sqrt(n_samples)``, while shaped or partially clipped readout pulses only
    contribute where their envelope overlaps the ADC window.
    """

    envelope = readout_envelope_samples(readout_cfg, pulse_length_us, sample_times_us)
    return math.sqrt(float(np.sum(envelope * envelope)))


def noise_std_sample_scale(n_samples: int) -> float:
    """Return the Gaussian raw-noise std scale for an integrated sample count."""

    n = int(n_samples)
    if n != n_samples or n <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}")
    return math.sqrt(float(n))


def mixed_signal(
    sim: SimParams,
    freqs: NDArray[np.float64],
    rf_g: float,
    rf_e: float,
    p_e: float,
) -> NDArray[np.complex128]:
    """Population-weighted accumulated readout signal at ``freqs`` (GHz).

    Assembles ``S21(rf_g) + p_e * [S21(rf_e) - S21(rf_g)]`` from the
    state-conditioned dressed resonator frequencies ``rf_g`` / ``rf_e``.

    The dressed frequencies are passed in (not computed here): they depend only
    on flux + EJ/EC/EL/g/bare_rf — all fixed across a sweep once the operating
    flux is pinned (R-3) — so the engine computes them ONCE via
    :func:`resonator_freqs` and feeds them to every sweep point's blend.  This
    function is therefore the pure, eigh-free S21 mixing layer (no fluxonium
    diagonalisation), which is what keeps the per-point hot path cheap.

    ``freqs`` are the readout probe frequencies — an array swept by onetone, or
    a length-1 array holding the single fixed ``f_ro`` for twotone / time-domain
    experiments. ``p_e`` is the excited-state population, which must lie in
    ``[0, 1]`` (Fast-Fail).
    """
    if not 0.0 <= p_e <= 1.0:
        raise ValueError(f"p_e must be in [0, 1], got {p_e}")

    s_g = s21(sim, freqs, rf_g)
    s_e = s21(sim, freqs, rf_e)
    return s_g + p_e * (s_e - s_g)


def decimated_trace(
    sim: SimParams,
    ts: NDArray[np.float64],
    readout_cfg: PulseCfg,
    pulse_length: float,
    f_ro_ghz: float,
    rf_g: float,
    rf_e: float,
    p_e: float,
    *,
    pulse_pre_delay_us: float = 0.0,
    state_visibility: float = 1.0,
) -> NDArray[np.complex128]:
    """Time-domain down-converted readout trace (model A) at ADC samples ``ts``.

    Model A (.agent_state/plans/mocksim/findings.md, "decimated 支援評估"):

        trace(t) = readout_envelope(t - timeFly - pulse_pre_delay)
                 * steady_mixed_S21(f_ro; rf_g, rf_e, p_e)

    i.e. the down-converted readout pulse envelope scaled by the *steady-state*
    population-weighted resonator response at the single probe frequency
    ``f_ro``.  There is no resonator ring-up transient (that is model B); the
    response is the same complex IQ point :func:`mixed_signal` produces for an
    accumulated readout, applied per sample.

    Time of flight.  A PulseReadout schedules its generator pulse at
    ``module_t + pulse_pre_delay``; that signal reaches the ADC ``sim.timeFly``
    later (the propagation delay), so the envelope is sampled at
    ``ts - sim.timeFly - pulse_pre_delay``.  With zero pre-delay, the trace is ~0
    for ``ts < sim.timeFly`` and the readout pulse appears in
    ``[timeFly, timeFly + pulse_length)``.  ``ts`` is the program-time axis
    (``cycles2us(get_time_axis) + trig_offset``); the trig_offset is already
    baked into ``ts`` by the engine, so this layer applies no trig_offset shift.

    Parameters
    ----------
    sim
        Physical parameters: the hanger Ql/Qi (via :func:`s21`) and ``timeFly``
        (the readout propagation delay that positions the envelope).
    ts
        Program-time axis in µs (``cycles2us(get_time_axis) + trig_offset``),
        same axis lookback plots.
    readout_cfg
        The readout pulse cfg whose waveform defines the envelope shape (for a
        PulseReadout this is ``ro_module.cfg.pulse_cfg``).
    pulse_length
        The resolved generator pulse/envelope length (µs) at this sweep point;
        the envelope is zero outside ``[0, pulse_length)``. This is deliberately
        separate from the ADC/readout integration window length (`ro_length`),
        which only determines the sampled ``ts`` axis upstream.
    f_ro_ghz
        The readout probe frequency (GHz); the steady S21 is evaluated here.
    rf_g, rf_e
        State-conditioned dressed resonator frequencies (GHz), computed ONCE by
        the engine (no dispersive eigh is done in this layer — same boundary as
        :func:`mixed_signal`).
    p_e
        Excited-state population in ``[0, 1]`` (Fast-Fail via ``mixed_signal``).
    pulse_pre_delay_us
        Resolved generator pulse pre-delay in µs.  Defaults to 0 for legacy tests
        and readout configs that do not delay the generator pulse.
    state_visibility
        Remaining |g>/|e> readout contrast after nonlinear readout backaction.
        Defaults to 1.0 for the linear model.

    Returns
    -------
    NDArray[np.complex128]
        One complex IQ value per entry of ``ts`` (same length).

    Responsibility boundary: physical -> time-domain IQ only.  No sweep / acc_buf
    / noise assembly (noise is added by the engine) and no dispersive computation
    (rf_g/rf_e arrive pre-computed).
    """

    ts = np.asarray(ts, dtype=np.float64)
    if ts.ndim != 1:
        raise ValueError(f"ts must be a 1-D array, got shape {ts.shape}")

    # Steady-state mixed S21 at the single probe frequency.  Apply the same
    # high-photon visibility compression as accumulated readout before mixing.
    freqs = np.array([f_ro_ghz], dtype=np.float64)
    s_g = s21(sim, freqs, rf_g)
    s_e = s21(sim, freqs, rf_e)
    s_g, s_e = apply_readout_visibility(s_g, s_e, state_visibility)
    steady = blend_state_responses(s_g, s_e, p_e)[0]

    pulse_pre_delay = float(pulse_pre_delay_us)
    if not math.isfinite(pulse_pre_delay) or pulse_pre_delay < 0.0:
        raise ValueError(
            f"pulse_pre_delay_us must be finite and non-negative, got {pulse_pre_delay_us!r}"
        )

    amp = envelope_at(readout_cfg, ts - sim.timeFly - pulse_pre_delay, pulse_length)
    return (amp * steady).astype(np.complex128)
