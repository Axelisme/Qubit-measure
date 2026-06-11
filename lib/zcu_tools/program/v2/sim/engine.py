"""SimEngine — assemble lowering + bloch + readout into QICK raw acc_buf data.

This is the top of the physics stack: it consumes a *compiled* ``MyProgramV2``
(for its loop structure / readout channels) plus :class:`SimParams`, walks every
sweep point of the program, and produces the integer I/Q "raw" buffer that the
real QICK accumulated-readout path expects.  The engine deliberately owns only
the *glue* responsibilities:

  - flux operating point (fixed at reduced flux Phi/Phi0 = 1.0; R-3),
  - qubit transition frequency ``f_qubit`` (fluxonium prediction at that flux),
  - driving lowering -> bloch -> excited population -> dispersive IQ per point,
  - laying the per-point IQ out into the ``(*loop_dims, nreads, 2)`` int64 buffer
    in QICK's flat time order, and
  - the per-shot Gaussian noise model (snr / reps / rounds / seed).

It does NOT re-implement lowering (module tree -> Bloch timeline), readout (IQ
physics) or bloch (TLS propagation); those are delegated to the sibling modules.

acc_buf layout (the load-bearing invariant)
-------------------------------------------
A compiled ``AveragerProgramV2`` has ``loop_dims = [reps, sweep0, sweep1, ...]``
(reps outermost, ``avg_level == 0``).  The accumulated path fills, for each
readout channel, an array of shape ``(*loop_dims, nreads, 2)`` of int64, in flat
C-order time sequence.  The engine computes one *deterministic* complex IQ value
per (sweep-point, read) — identical across reps for an averaged readout (per-shot
Bernoulli sampling is a Phase-2 concern) — then broadcasts across the reps axis
and adds independent per-shot noise.

Noise model
-----------
``SimParams.snr`` is the signal-to-noise ratio of a *single* repetition.  Each
element of the raw buffer (one rep at one sweep point) gets additive Gaussian
noise with standard deviation ``_FULL_SCALE / snr`` on each I/Q quadrature.
Averaging over the ``reps`` axis (and, across rounds, over ``rounds``) then
improves the effective SNR by ``sqrt(reps * rounds)`` — exactly as on hardware,
because the round loop reruns the program and software-averages.  ``seed`` makes
the noise reproducible; each round draws fresh noise so re-running the round loop
is statistically meaningful (Q1).
"""

from __future__ import annotations

import itertools
import logging
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

from . import bloch
from .lowering import LoweredPoint, lower_point
from .params import SimParams
from .readout import mixed_signal, resonator_freqs

logger = logging.getLogger(__name__)

# Operating flux is fixed at reduced flux Phi/Phi0 = 1.0 (R-3): the simulation is a
# device-pipeline validator, so it pins one operating point instead of deriving it
# from the experiment cfg's ``dev`` map.  The engine works in true (absolute,
# non-folded) frequencies throughout; the mock gen f_dds is high enough (12288 MHz)
# that the f01 the prediction lands on at this operating flux sits well below f_dds,
# so the analyzer reports it un-folded.  Folding is a ``f mod f_dds`` analyzer-axis
# effect only and is physically harmless to the Bloch dynamics; see sim/README
# Nyquist note.  Changing the operating point is a one-line edit here.
_SIM_OPERATING_FLUX = 1.0

# MHz -> GHz for the predictor output (predict_freq returns MHz; lowering and the
# readout model both work in GHz).
_MHZ_TO_GHZ = 1e-3

# Full-scale raw amplitude per readout sample.  The dispersive IQ signal from
# ``mixed_signal`` is order-unity, so it is scaled by this and by the readout
# window length (real hardware integrates ``length`` samples) to land in a
# sensible int range.  The absolute value is arbitrary — only its ratio to the
# noise (via snr) carries physical meaning.
_FULL_SCALE = 1.0e4

# Number of Gauss-Legendre nodes for the Lorentzian quasi-static detune ensemble
# (Phase-2 dephasing model).  The substitution ``delta = Gamma * tan(theta)``
# maps the Lorentzian weight to a *uniform* weight on ``theta in (-pi/2, pi/2)``,
# so a fixed Gauss-Legendre rule on that interval integrates the ensemble average
# deterministically.  41 nodes reproduce the analytic FID ``exp(-Gamma|t|)`` to
# within a few % over the observable decay window ``Gamma*t in [0, 2]`` (the
# oscillatory integrand converges slowly only deep in the decayed tail, which is
# below the noise floor and not load-bearing); it is a constant (not a SimParams
# field) because it is a numerics knob, not physics.  The end-to-end inject->recover
# integration tests (against the real analyzers) are the true correctness gate.
_ENSEMBLE_NODES = 41


def _lorentzian_quadrature(
    n_nodes: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(theta_nodes, weights)`` for the Lorentzian detune ensemble average.

    With ``delta = Gamma * tan(theta)`` the Lorentzian (HWHM ``Gamma``) average of
    any ``f(delta)`` becomes ``(1/pi) * integral over theta in (-pi/2, pi/2) of
    f(Gamma*tan(theta)) dtheta`` (the Jacobian cancels the Lorentzian density
    exactly).  Gauss-Legendre on ``[-1, 1]`` is rescaled to ``theta`` here; the
    returned ``weights`` already fold in the ``(pi/2) / pi = 1/2`` interval
    rescaling and normalisation so they sum to 1 — i.e. a node's complex signal is
    multiplied by ``weights[i]`` and summed to get the ensemble mean.
    """

    x, w = leggauss(n_nodes)  # nodes/weights on [-1, 1], sum(w) == 2
    half = math.pi / 2.0
    theta = half * x  # map [-1, 1] -> (-pi/2, pi/2)
    weights = (half * w) / math.pi  # (interval scale) * (Lorentzian 1/pi), sums to 1
    return theta, weights


class SimEngine:
    """Turn a compiled program + SimParams into QICK raw accumulated I/Q data.

    The engine is constructed from an already-compiled ``MyProgramV2`` so it can
    read the loop structure (``loop_dims`` / ``avg_level``), the readout channels
    (``ro_chs``), and the semantic module tree (``modules`` / ``sweep_dict``).
    """

    def __init__(self, program, sim: SimParams) -> None:
        from zcu_tools.program.v2.modular import ModularProgramV2

        if not isinstance(program, ModularProgramV2):
            raise TypeError(
                "SimEngine requires a ModularProgramV2 (it reads the semantic "
                f"module tree); got {type(program).__name__}"
            )
        if program.loop_dims is None or program.avg_level is None:
            raise RuntimeError(
                "SimEngine requires a compiled program (loop_dims / avg_level "
                "are set by compile()); call program.compile() first"
            )

        self.program = program
        self.sim = sim

        # Deterministic raw-IQ grid (reps-broadcast, no noise), built lazily on
        # the first compute_round and reused across rounds; None until then.
        self._det_full: NDArray[np.float64] | None = None

        # One predictor instance reused across all sweep points; building it is
        # expensive (lazy scqubits import + Fluxonium construction).
        self._predictor = FluxoniumPredictor(
            params=(sim.EJ, sim.EC, sim.EL),
            flux_half=sim.flux_half,
            flux_period=sim.flux_period,
            flux_bias=sim.flux_bias,
        )
        self._rng = np.random.default_rng(sim.seed)

        # Lorentzian quasi-static detune ensemble (Phase-2 dephasing model).
        # Gamma is the Lorentzian HWHM in rad/µs (numerically == the
        # inhomogeneous rate in 1/µs).  Gamma == 0 (T2_star == T2) means no
        # inhomogeneous broadening: a single node at delta == 0 reproduces the
        # Phase-1 single-eval timeline exactly (bit-identical signal, RNG
        # untouched).  Only when Gamma > 0 do we spend the quadrature.
        self._gamma = sim.inhomogeneous_rate
        if self._gamma > 0.0:
            theta, weights = _lorentzian_quadrature(_ENSEMBLE_NODES)
            self._detune_nodes = self._gamma * np.tan(theta)  # delta_i in rad/µs
            self._detune_weights = weights
        else:
            self._detune_nodes = np.zeros(1, dtype=np.float64)
            self._detune_weights = np.ones(1, dtype=np.float64)

    # ----------------------------------------------------------- sweep points
    def _sweep_axes(self) -> list[tuple[str, int]]:
        """Return the sweep axes as ``[(name, count), ...]`` (program order).

        Mirrors ``loop_dims`` minus the leading reps axis, so the cartesian
        product of the counts indexes the per-point IQ array's sweep dimensions.
        """

        sweep = self.program.sweep_dict or []
        from zcu_tools.program.v2.sweep import SweepCfg

        return [
            (name, spec.expts if isinstance(spec, SweepCfg) else int(spec))
            for name, spec in sweep
        ]

    def _signal_grid(
        self, f_qubit_ghz: float, rf_g: float, rf_e: float
    ) -> NDArray[np.complex128]:
        """Compute the deterministic per-(sweep-point, read) complex IQ grid.

        Returns an array of shape ``(*sweep_dims, nreads)`` of complex IQ values
        (one readout channel; multi-channel sim is out of scope — the engine
        asserts a single readout channel).  Each entry is the population-weighted
        dispersive readout signal at that sweep point.

        ``rf_g`` / ``rf_e`` are the dressed resonator frequencies at the (fixed)
        operating flux; the caller computes them once and passes them in, so the
        per-point loop never re-runs the fluxonium eigensolve (see
        :meth:`_ensure_signal`).
        """

        axes = self._sweep_axes()
        sweep_dims = tuple(count for _, count in axes)
        nreads = self._nreads()

        grid = np.empty((*sweep_dims, nreads), dtype=np.complex128)

        index_ranges = [range(count) for _, count in axes]
        for multi_index in itertools.product(*index_ranges):
            point = {name: idx for (name, _), idx in zip(axes, multi_index)}
            iq = self._point_signal(point, f_qubit_ghz, rf_g, rf_e)
            grid[(*multi_index, slice(None))] = iq

        return grid

    def _point_signal(
        self, point: dict[str, int], f_qubit_ghz: float, rf_g: float, rf_e: float
    ) -> complex:
        """Deterministic complex IQ at one sweep point (single unified path).

        Every sweep point follows the *same* physics: lower the module tree,
        evolve the Bloch timeline to an excited population ``P_e``, and read out
        the dispersive signal at this point's readout probe frequency
        ``f_ro``.  No experiment type (onetone / twotone / qubit-pulse + swept
        ``f_ro``) is special-cased — they differ only in what the timeline and
        the per-point ``f_ro`` happen to be:

          - onetone has no qubit pulse, so the Bloch vector relaxes to ~thermal
            and ``P_e ≈ thermal_pop`` while the swept ``f_ro`` traces the S21
            dip near ``rf_g``;
          - twotone / time-domain drive the qubit (``P_e`` set by the pulse) and
            read out at a fixed ``f_ro``;
          - a qubit pulse *with* a swept ``f_ro`` excites ``P_e`` AND sweeps the
            probe, so the dip rides ``rf_e`` (π pulse) or ``rf_g`` (no pulse).

        ``f_ro`` is taken per point from ``ReadoutPlan.f_ro_ghz``, which
        ``lower_point`` already resolves to the swept-or-fixed value at this
        sweep index — so a swept readout frequency flows through naturally
        without the engine branching on it.

        Under the Lorentzian quasi-static detune model the observed signal is the
        *ensemble average* over the static detune δ.  The dispersive readout
        ``signal = S21(rf_g) + P_e·[S21(rf_e) − S21(rf_g)]`` is linear in
        ``P_e``, so averaging ``P_e`` over the ensemble and feeding the mean into
        ``mixed_signal`` once is exactly equal to (and cheaper than) averaging the
        complex signal — we average ``P_e``.  An echo π flip refocuses each δ; a
        Ramsey free evolution does not, so the extra ``exp(−Γt)`` decay (→ T2*)
        emerges from this average without the engine identifying the sequence.
        The ``f_ro`` is δ-independent, so it is read once from the δ=0 lowering.

        ``rf_g`` / ``rf_e`` are the flux-constant dressed resonator frequencies
        the caller computed once; they are fed straight into ``mixed_signal``, so
        no fluxonium eigensolve runs per point.
        """

        # δ=0 lowering supplies this point's f_ro (δ never affects readout) and
        # serves as the single-node ensemble when Gamma == 0.
        zero_lowered = self._lower(point, f_qubit_ghz, 0.0)

        # No-drive short-circuit: the quasi-static detune δ enters the Bloch
        # generator only through the x/y rotation (gen[0,1] / gen[1,0]).  With no
        # drive segment (every omega == 0) the z-row decouples entirely from δ
        # (gen[2, :] is just [0, 0, -gamma1, z_eq*gamma1]), so the excited
        # population P_e = (1+z)/2 is δ-independent — every ensemble node yields
        # the identical P_e.  The ensemble average then equals a single eval
        # exactly (this is the *value* of the average, not an approximation), so a
        # driveless timeline (e.g. a pure onetone readout sweep) skips the 41-node
        # quadrature.  This does NOT special-case experiment type or restore a
        # per-experiment split (R-1): it is the mathematical identity "mean of
        # identical values == the value", gated only on whether any drive exists.
        has_drive = any(seg.omega != 0.0 for seg in zero_lowered.segments)

        v0 = bloch.ground_state(self.sim.thermal_pop)
        if has_drive:
            # Ensemble-average P_e over the Lorentzian detune nodes.  The Gamma == 0
            # ensemble is a single node at δ = 0, so this reduces to one Bloch eval
            # (zero regression vs the pre-ensemble single-eval path).
            p_e_mean = 0.0
            for delta, weight in zip(self._detune_nodes, self._detune_weights):
                lowered = (
                    zero_lowered
                    if delta == 0.0
                    else self._lower(point, f_qubit_ghz, float(delta))
                )
                v_final = bloch.evolve(v0, lowered.segments)
                p_e = bloch.excited_population(v_final)
                # Bloch keeps z in [-1, 1] (CPTP), but matrix-exponential round-off
                # can nudge p_e a hair outside [0, 1]; clamp each node before
                # averaging (mixed_signal Fast-fails on out-of-range p_e).
                p_e_mean += float(weight) * float(np.clip(p_e, 0.0, 1.0))
        else:
            v_final = bloch.evolve(v0, zero_lowered.segments)
            p_e_mean = float(np.clip(bloch.excited_population(v_final), 0.0, 1.0))

        freqs = np.array([zero_lowered.readout.f_ro_ghz], dtype=np.float64)
        signal = mixed_signal(self.sim, freqs, rf_g, rf_e, p_e_mean)
        return complex(signal[0])

    def _lower(
        self, point: dict[str, int], f_qubit_ghz: float, detune_offset: float
    ) -> LoweredPoint:
        """Lower the module tree at one sweep point with a static frame shift.

        ``detune_offset`` (rad/µs) is one Lorentzian ensemble node's quasi-static
        detune; lowering applies it as a global frame shift on every segment.
        """

        return lower_point(
            self.program.modules,
            self.program.sweep_dict,
            self.sim,
            f_qubit_ghz,
            point,
            self.program.soccfg.cycles2us,
            detune_offset=detune_offset,
        )

    def _nreads(self) -> int:
        """Reads per shot for the single readout channel."""

        ro_chs = self.program.ro_chs
        if len(ro_chs) != 1:
            raise NotImplementedError(
                f"SimEngine supports exactly one readout channel; program has "
                f"{len(ro_chs)}"
            )
        (ro,) = ro_chs.values()
        return int(ro["trigs"])

    # ----------------------------------------------------------- raw assembly
    def _ensure_signal(self) -> NDArray[np.float64]:
        """Build (once, cached) the deterministic raw I/Q broadcast over reps.

        The flux, f_qubit and per-point dispersive signal grid are independent of
        the round index — only the additive per-shot noise differs round to round.
        So the expensive part (lowering + Bloch + readout for every sweep point)
        is computed on the *first* round poll and cached; later rounds reuse it
        and only redraw noise.  This is what makes ``compute_round`` lazy: a run
        that early-stops never computes a round it does not poll.

        Returns the deterministic ``(reps, *sweep, nreads, 2)`` float array
        (no noise yet), at the integration full scale.
        """

        if self._det_full is not None:
            return self._det_full

        # Operating point is fixed at reduced flux = 1.0 (R-3).  predict_freq
        # consumes a *device value*, so map the fixed reduced flux back through the
        # predictor's affine alignment (flux_to_value) rather than rewriting it.
        flux = _SIM_OPERATING_FLUX
        device_value = self._predictor.flux_to_value(_SIM_OPERATING_FLUX)
        f_qubit_mhz = float(self._predictor.predict_freq(device_value))
        f_qubit_ghz = f_qubit_mhz * _MHZ_TO_GHZ

        # Dressed resonator frequencies are flux-constant (R-3 pins the operating
        # flux), so the fluxonium eigensolve behind ``resonator_freqs`` runs ONCE
        # here instead of once per sweep point inside the old ``mixed_signal``.
        # CACHE DEPENDENCY: this is valid only because ``_SIM_OPERATING_FLUX`` is a
        # single fixed flux for the whole sweep; if the operating flux ever became
        # per-point, rf_g/rf_e would have to move back into the per-point loop.
        rf_g, rf_e = resonator_freqs(self.sim, flux)

        logger.debug(
            "SimEngine: flux=%.4f, f_qubit=%.4f GHz, rf_g=%.4f GHz, rf_e=%.4f GHz",
            flux,
            f_qubit_ghz,
            rf_g,
            rf_e,
        )

        signal_grid = self._signal_grid(  # (*sweep, nreads) complex
            f_qubit_ghz, rf_g, rf_e
        )

        loop_dims = self.program.loop_dims
        assert loop_dims is not None  # guaranteed by __init__; reasserted for typing
        reps = loop_dims[0]
        sweep_dims = signal_grid.shape[:-1]
        nreads = signal_grid.shape[-1]

        # Deterministic raw I/Q (no reps axis yet): scale the order-unity IQ to
        # the integration full scale.
        det = _FULL_SCALE * signal_grid  # (*sweep, nreads) complex
        det_iq = np.stack([det.real, det.imag], axis=-1)  # (*sweep, nreads, 2)

        # Broadcast over reps -> (reps, *sweep, nreads, 2).  np.array materializes
        # the broadcast so each round can add a fresh independent noise draw.
        self._det_full = np.array(
            np.broadcast_to(det_iq, (reps, *sweep_dims, nreads, 2))
        )
        return self._det_full

    def compute_round(self, round_idx: int) -> list[NDArray[np.int64]]:
        """Compute one round's raw acc_buf lazily (called by the mock soc's poll).

        ``round_idx`` is informational (rounds redraw independent noise from the
        engine's RNG, in poll order); the deterministic signal grid is built once
        on the first call and cached (:meth:`_ensure_signal`), so each round only
        adds a fresh per-shot Gaussian noise draw.  Returns the one-channel list
        ``[acc_buf]`` with ``acc_buf`` of shape ``(*loop_dims, nreads, 2)`` int64
        — exactly what the mock soc serves through ``poll_data`` for that round.
        Software-averaging the rounds improves SNR because each draw is independent.
        """

        logger.debug("SimEngine.compute_round: round_idx=%d", round_idx)

        det_full = self._ensure_signal()
        noise_std = _FULL_SCALE / self.sim.snr
        noise = self._rng.normal(0.0, noise_std, size=det_full.shape)
        acc = np.rint(det_full + noise).astype(np.int64)
        return [acc]
