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
C-order time sequence.  The engine computes two *deterministic* complex blobs per
(sweep-point, read) — the |g>- and |e>-conditioned readout ``s_g`` / ``s_e`` —
plus the excited population ``p_e``, then draws a per-shot Bernoulli(``p_e``)
across the reps axis to select between the blobs and adds independent per-shot
noise.  This single unified path serves both the accumulated readout (its
reps-mean is ``(1−p_e)·s_g + p_e·s_e``, the averaged dispersive signal) and
singleshot ``get_raw`` (which sees the two Gaussian blobs) — no singleshot-mode
detection.

Noise model
-----------
``SimParams.snr`` is the per-sample Gaussian readout scale before integration.
Each raw buffer element is an integrated ADC sum: the deterministic signal scales
with the effective signal samples, and the Gaussian noise standard deviation
scales with ``sqrt(compiled_readout_samples)``. Averaging over the ``reps`` axis
(and, across rounds, over ``rounds``) then improves the effective SNR by
``sqrt(reps * rounds)`` — exactly as on hardware, because the round loop reruns
the program and software-averages.  ``seed`` makes the noise reproducible; each
round draws fresh noise so re-running the round loop is statistically meaningful
(Q1).
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from zcu_tools.program.v2.modules.readout import (
    AbsReadout,
    DirectReadout,
    PulseReadout,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

from . import bloch
from .lowering import LoweredPoint, _resolve_scalar, lower_point
from .params import SimParams
from .readout import (
    decimated_trace,
    effective_signal_samples,
    noise_std_sample_scale,
    readout_drive_amplitude,
    resonator_freqs,
    s21,
)

logger = logging.getLogger(__name__)

# Default operating flux: reduced flux Phi/Phi0 = 1.0 (R-3).  This is the operating
# point when no device is bound; the FLUX-AWARE-MOCK path (SimParams.flux_device,
# see _reduced_operating_flux) overrides it by reading a live FakeDevice value.  The
# simulation is a device-pipeline validator, so by default it pins one operating
# point instead of deriving it from the experiment cfg's ``dev`` map.  The engine
# works in true (absolute, non-folded) frequencies throughout; the mock gen f_dds is
# high enough (12288 MHz) that the f01 the prediction lands on at this operating flux
# sits well below f_dds, so the analyzer reports it un-folded.  Folding is a
# ``f mod f_dds`` analyzer-axis effect only and is physically harmless to the Bloch
# dynamics; see sim/README Nyquist note.
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


class SimCancelledError(RuntimeError):
    """Raised when a mock simulation stops cooperatively via ``stop_checkers``."""


@lru_cache(maxsize=256)
def _cached_predict_freq_ghz(
    EJ: float,
    EC: float,
    EL: float,
    flux_half: float,
    flux_period: float,
    flux_bias: float,
    reduced_flux: float,
) -> float:
    """Cached qubit transition frequency for one physical operating point."""

    predictor = FluxoniumPredictor(
        params=(EJ, EC, EL),
        flux_half=flux_half,
        flux_period=flux_period,
        flux_bias=flux_bias,
    )
    device_value = predictor.flux_to_value(reduced_flux)
    f_qubit_mhz = float(predictor.predict_freq(device_value))
    return f_qubit_mhz * _MHZ_TO_GHZ


class SimEngine:
    """Turn a compiled program + SimParams into QICK raw accumulated I/Q data.

    The engine is constructed from an already-compiled ``MyProgramV2`` so it can
    read the loop structure (``loop_dims`` / ``avg_level``), the readout channels
    (``ro_chs``), and the semantic module tree (``modules`` / ``sweep_dict``).
    """

    def __init__(
        self,
        program,
        sim: SimParams,
        stop_checkers: list[Callable[[], bool]] | None = None,
    ) -> None:
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
        self._stop_checkers = tuple(stop_checkers or ())

        # Deterministic per-(sweep-point, read) blob grids plus integration
        # scales, each of shape ``(*sweep, nreads)``, built lazily on the first
        # compute_round and reused across rounds; None until then. Bernoulli
        # sampling and noise are NOT cached here — they are redrawn per round.
        self._det_grids: (
            tuple[
                NDArray[np.complex128],
                NDArray[np.complex128],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
            ]
            | None
        ) = None

        # Flux-constant operating point (f_qubit, rf_g, rf_e), computed once and
        # shared by the accumulated and decimated paths; None until first use.
        self._operating: tuple[float, float, float] | None = None

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

    def _raise_if_cancelled(self) -> None:
        """Fail fast when any acquire-level stop checker requests cancellation."""

        for checker in self._stop_checkers:
            if checker():
                raise SimCancelledError(
                    "mock simulation cancelled because a stop_checker returned True"
                )

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
    ) -> tuple[
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Compute the deterministic per-(sweep-point, read) blob grids.

        Returns ``(s_g_grid, s_e_grid, p_e_grid, signal_scale_grid,
        noise_std_scale_grid)``, each of shape
        ``(*sweep_dims, nreads)`` (one readout channel; multi-channel sim is out
        of scope — the engine asserts a single readout channel):

          - ``s_g_grid`` / ``s_e_grid`` are the |g>- / |e>-conditioned complex
            readout blobs (``S21(f_ro; rf_g)`` / ``S21(f_ro; rf_e)``) — the two
            per-shot Bernoulli outcomes,
          - ``p_e_grid`` is the (ensemble-averaged) excited population at each
            point, the Bernoulli probability that a shot lands on the ``s_e``
            blob.
          - ``signal_scale_grid`` / ``noise_std_scale_grid`` carry the raw
            integration factors for deterministic signal and Gaussian noise.

        ``compute_round`` draws per-shot Bernoulli(``p_e``) and selects between
        ``s_g`` / ``s_e``; the reps-mean is ``(1−p_e)·s_g + p_e·s_e ==
        mixed_signal``, so the accumulated readout is unchanged while singleshot
        ``get_raw`` sees two distinct Gaussian blobs (single unified path, no
        singleshot-mode detection).

        ``rf_g`` / ``rf_e`` are the dressed resonator frequencies at the (fixed)
        operating flux; the caller computes them once and passes them in, so the
        per-point loop never re-runs the fluxonium eigensolve (see
        :meth:`_ensure_signal`).
        """

        axes = self._sweep_axes()
        sweep_dims = tuple(count for _, count in axes)
        nreads = self._nreads()
        n_samples, sample_times_us = self._readout_sample_times_us()

        s_g_grid = np.empty((*sweep_dims, nreads), dtype=np.complex128)
        s_e_grid = np.empty((*sweep_dims, nreads), dtype=np.complex128)
        p_e_grid = np.empty((*sweep_dims, nreads), dtype=np.float64)
        signal_scale_grid = np.empty((*sweep_dims, nreads), dtype=np.float64)
        noise_std_scale_grid = np.empty((*sweep_dims, nreads), dtype=np.float64)

        index_ranges = [range(count) for _, count in axes]
        for multi_index in itertools.product(*index_ranges):
            self._raise_if_cancelled()
            point = {name: idx for (name, _), idx in zip(axes, multi_index)}
            s_g, s_e, p_e, signal_scale, noise_std_scale = self._point_signal(
                point, f_qubit_ghz, rf_g, rf_e, n_samples, sample_times_us
            )
            idx = (*multi_index, slice(None))
            s_g_grid[idx] = s_g
            s_e_grid[idx] = s_e
            p_e_grid[idx] = p_e
            signal_scale_grid[idx] = signal_scale
            noise_std_scale_grid[idx] = noise_std_scale

        return (
            s_g_grid,
            s_e_grid,
            p_e_grid,
            signal_scale_grid,
            noise_std_scale_grid,
        )

    def _point_signal(
        self,
        point: dict[str, int],
        f_qubit_ghz: float,
        rf_g: float,
        rf_e: float,
        n_samples: int,
        sample_times_us: NDArray[np.float64],
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float, float, float]:
        """Deterministic readout blobs at one sweep point (single unified path).

        Returns ``(s_g, s_e, p_e, signal_scale, noise_std_scale)`` where
        ``s_g`` / ``s_e`` are the |g>- / |e>-conditioned complex readout blobs at
        this point's probe frequency (``S21(f_ro; rf_g)`` / ``S21(f_ro; rf_e)``),
        ``p_e`` is the (ensemble-averaged) excited population, and the two scale
        values convert the order-unity blobs/noise into integrated raw ADC sums.

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

        Under the Lorentzian quasi-static detune model the observed population is
        the *ensemble average* over the static detune δ.  Because the readout is
        linear in ``P_e``, averaging ``P_e`` over the ensemble (then drawing the
        per-shot Bernoulli with that mean ``p_e``) is exactly equal to averaging
        the dispersive signal — so the engine averages ``P_e``.  An echo π flip
        refocuses each δ; a Ramsey free evolution does not, so the extra
        ``exp(−Γt)`` decay (→ T2*) emerges from this average without the engine
        identifying the sequence.  The ``f_ro`` is δ-independent, so it is read
        once from the δ=0 lowering.

        ``rf_g`` / ``rf_e`` are the flux-constant dressed resonator frequencies
        the caller computed once; they are fed straight into ``s21``, so no
        fluxonium eigensolve runs per point.
        """

        # δ=0 lowering supplies this point's f_ro (δ never affects readout) and
        # serves as the single-node ensemble when Gamma == 0.
        self._raise_if_cancelled()
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
                self._raise_if_cancelled()
                lowered = (
                    zero_lowered
                    if delta == 0.0
                    else self._lower(point, f_qubit_ghz, float(delta))
                )
                v_final = bloch.evolve(v0, lowered.segments)
                p_e = bloch.excited_population(v_final)
                # Bloch keeps z in [-1, 1] (CPTP), but matrix-exponential round-off
                # can nudge p_e a hair outside [0, 1]; clamp each node before
                # averaging (the Bernoulli probability must stay in [0, 1]).
                p_e_mean += float(weight) * float(np.clip(p_e, 0.0, 1.0))
        else:
            v_final = bloch.evolve(v0, zero_lowered.segments)
            p_e_mean = float(np.clip(bloch.excited_population(v_final), 0.0, 1.0))

        # The two per-shot blobs: the |g>- and |e>-conditioned dispersive readout
        # at this point's probe frequency.  ``s21`` is the pure (eigh-free) hanger
        # response; rf_g / rf_e arrive pre-computed so no fluxonium solve runs here.
        freqs = np.array([zero_lowered.readout.f_ro_ghz], dtype=np.float64)
        s_g = s21(self.sim, freqs, rf_g)
        s_e = s21(self.sim, freqs, rf_e)
        signal_scale = readout_drive_amplitude(
            zero_lowered.readout.readout_gain
        ) * effective_signal_samples(
            zero_lowered.readout.pulse_cfg,
            zero_lowered.readout.pulse_length_us,
            sample_times_us,
        )
        noise_std_scale = noise_std_sample_scale(n_samples)
        return s_g, s_e, p_e_mean, signal_scale, noise_std_scale

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

    def _readout_sample_times_us(self) -> tuple[int, NDArray[np.float64]]:
        """Compiled integration sample count and sample times for one readout."""

        ro_chs = self.program.ro_chs
        if len(ro_chs) != 1:
            raise NotImplementedError(
                f"SimEngine supports exactly one readout channel; program has "
                f"{len(ro_chs)}"
            )
        ((ro_ch, ro),) = ro_chs.items()
        n_samples = int(ro["length"])
        if n_samples <= 0:
            raise ValueError(
                f"compiled readout length must be positive, got {n_samples}"
            )
        ts = self.program.soccfg.cycles2us(np.arange(n_samples), ro_ch=ro_ch)
        return n_samples, np.asarray(ts, dtype=np.float64)

    # ------------------------------------------------------- operating point
    def _reduced_operating_flux(self) -> float:
        """Resolve the reduced operating flux Phi/Phi0 for this acquire.

        FLUX-AWARE-MOCK: the operating flux is normally pinned at reduced flux =
        1.0 (R-3), but ``SimParams.flux_device`` opts into reading it live from a
        connected device.  This is a deliberate cross-layer reach: the engine
        lives in ``program/v2/sim/`` yet, when bound, peers into the
        ``GlobalDeviceManager`` registry (``device/``) to read the *current*
        device value, because the mock soc must mirror the real rig where the
        software flux sweep sets a YOKO/FakeDevice value per acquire and the qubit
        frequency follows it.  The read is intentionally lazy and happens once per
        acquire: a fresh SimEngine is built on every ``MyProgramV2.acquire`` (see
        base._attach_sim_engine), so reading here is equivalent to "read the live
        flux just before each acquisition".  Within a single acquire the flux is
        constant (the runner does software-per-acquire: set device value, then run
        one acquire), which is exactly the assumption the flux-constant caches in
        :meth:`_operating_signal` rely on.

        Only a ``FakeDevice`` is supported as the source: the simulation models a
        dev-only mock rig, and a FakeDevice exposes a plain in-memory ``value`` the
        engine can read without any instrument I/O.  A missing device or a
        non-FakeDevice is a wiring mistake, so fail-fast here (the binding itself,
        via ``set_flux_device``, is permitted before the device is registered;
        the resolution is what enforces the contract).
        """

        flux_device = self.sim.flux_device
        if flux_device is None:
            # Zero-regression path: no binding -> the historical fixed operating
            # point (reduced flux = 1.0, R-3).
            return _SIM_OPERATING_FLUX

        # Lazy local import (FLUX-AWARE-MOCK): keep the device dependency off the
        # sim package's import graph.  ``device/`` never imports ``program/v2/sim``,
        # so there is no import cycle; importing inside the function also avoids
        # paying the device import cost on the (default) fixed-flux path.
        from zcu_tools.device import FakeDevice, GlobalDeviceManager

        dev = GlobalDeviceManager.get_device(flux_device)
        if not isinstance(dev, FakeDevice):
            raise TypeError(
                f"SimEngine flux_device {flux_device!r} must be a FakeDevice "
                f"(the mock simulation only reads a FakeDevice's in-memory value); "
                f"got {type(dev).__name__}"
            )

        # Map the live device value to reduced flux through THIS SimParams' affine
        # (flux_half / flux_period / flux_bias), the same alignment predict_freq
        # uses internally, so the operating point stays self-consistent.
        device_value = dev.get_value()
        return self._predictor.value_to_flux(device_value)

    def _operating_signal(self) -> tuple[float, float, float]:
        """Return (cached) ``(f_qubit_ghz, rf_g, rf_e)`` at the operating flux.

        The operating flux comes from :meth:`_reduced_operating_flux`
        (FLUX-AWARE-MOCK: fixed reduced flux = 1.0 by default, or read live from a
        bound FakeDevice).  ``predict_freq`` consumes a *device value*, so map the
        resolved reduced flux back through the predictor's affine alignment
        (``flux_to_value``) rather than rewriting it.  The dressed resonator
        frequencies are flux-constant *within one acquire* (the flux is read once
        here and held for the whole sweep), so the fluxonium eigensolve behind
        ``resonator_freqs`` runs ONCE here and is reused by every sweep point
        (accumulated) and by the decimated trace.  The cache is per-engine and an
        engine is rebuilt every acquire, so a changed device value between acquires
        is picked up by the next engine (no stale cross-acquire flux).
        """

        if self._operating is not None:
            return self._operating

        self._raise_if_cancelled()
        reduced_flux = self._reduced_operating_flux()
        f_qubit_ghz = _cached_predict_freq_ghz(
            self.sim.EJ,
            self.sim.EC,
            self.sim.EL,
            self.sim.flux_half,
            self.sim.flux_period,
            self.sim.flux_bias,
            reduced_flux,
        )
        self._raise_if_cancelled()
        rf_g, rf_e = resonator_freqs(self.sim, reduced_flux)
        self._raise_if_cancelled()

        logger.debug(
            "SimEngine: flux=%.4f, f_qubit=%.4f GHz, rf_g=%.4f GHz, rf_e=%.4f GHz",
            reduced_flux,
            f_qubit_ghz,
            rf_g,
            rf_e,
        )

        self._operating = (f_qubit_ghz, rf_g, rf_e)
        return self._operating

    # ----------------------------------------------------------- raw assembly
    def _ensure_signal(
        self,
    ) -> tuple[
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Build (once, cached) deterministic blob grids and integration scales.

        The flux, f_qubit and per-point blob grids are independent of the round
        index — only the per-shot Bernoulli draw and the additive noise differ
        round to round.  So the expensive part (lowering + Bloch + readout for
        every sweep point) is computed on the *first* round poll and cached;
        later rounds reuse it and only redraw the Bernoulli state + noise.  This
        is what makes ``compute_round`` lazy: a run that early-stops never
        computes a round it does not poll.

        Returns ``(s_g_grid, s_e_grid, p_e_grid, signal_scale_grid,
        noise_std_scale_grid)``, each of shape ``(*sweep, nreads)``. Bernoulli
        sampling and random noise are applied in :meth:`compute_round`, not
        cached here.
        """

        self._raise_if_cancelled()
        if self._det_grids is not None:
            return self._det_grids

        f_qubit_ghz, rf_g, rf_e = self._operating_signal()
        self._det_grids = self._signal_grid(f_qubit_ghz, rf_g, rf_e)
        return self._det_grids

    def compute_round(self, round_idx: int) -> list[NDArray[np.int64]]:
        """Compute one round's raw acc_buf lazily (called by the mock soc's poll).

        ``round_idx`` is informational (rounds redraw an independent Bernoulli
        state + noise from the engine's RNG, in poll order); the deterministic
        ``(s_g, s_e, p_e)`` grids are built once on the first call and cached
        (:meth:`_ensure_signal`).

        Per-shot Bernoulli (the unified singleshot path).  For each element of
        the ``(reps, *sweep, nreads)`` buffer a shot is drawn ``state ~
        Bernoulli(p_e)`` and its complex blob is ``s_e`` where ``state == 1`` else
        ``s_g``.  Averaging over reps recovers ``(1−p_e)·s_g + p_e·s_e ==
        mixed_signal`` in expectation, so the accumulated readout is unchanged
        (zero regression); a singleshot ``get_raw`` instead sees two Gaussian
        blobs centred on ``_FULL_SCALE·s_g`` / ``_FULL_SCALE·s_e``.  A fresh
        Bernoulli state is drawn each round (matching the fresh-noise-per-round
        model), so software-averaging the rounds is statistically meaningful.

        Returns the one-channel list ``[acc_buf]`` with ``acc_buf`` of shape
        ``(*loop_dims, nreads, 2)`` int64 — exactly what the mock soc serves
        through ``poll_data`` for that round.
        """

        logger.debug("SimEngine.compute_round: round_idx=%d", round_idx)

        self._raise_if_cancelled()
        (
            s_g_grid,
            s_e_grid,
            p_e_grid,
            signal_scale_grid,
            noise_std_scale_grid,
        ) = self._ensure_signal()
        self._raise_if_cancelled()

        loop_dims = self.program.loop_dims
        assert loop_dims is not None  # guaranteed by __init__; reasserted for typing
        reps = loop_dims[0]
        full_shape = (reps, *s_g_grid.shape)  # (reps, *sweep, nreads)

        # Per-shot Bernoulli: 1 -> excited blob (s_e), 0 -> ground blob (s_g).
        # ``binomial(1, p_e_grid)`` broadcasts p_e over the reps axis, so each
        # shot at a point draws independently with that point's excited
        # probability.
        state = self._rng.binomial(1, p_e_grid, size=full_shape).astype(bool)
        blob = np.where(state, s_e_grid, s_g_grid)  # (reps, *sweep, nreads) complex

        det = _FULL_SCALE * signal_scale_grid * blob
        det_iq = np.stack([det.real, det.imag], axis=-1)  # (..., 2)

        noise_std = (_FULL_SCALE / self.sim.snr) * noise_std_scale_grid
        noise = self._rng.normal(0.0, noise_std[..., None], size=det_iq.shape)
        acc = np.rint(det_iq + noise).astype(np.int64)
        return [acc]

    # ------------------------------------------------------ decimated assembly
    def _readout_module(self) -> AbsReadout:
        """The single readout module in the program (the decimated source).

        Decimated/lookback timelines have exactly one readout; more than one (or
        none) is unsupported, matching the accumulated single-channel invariant.
        """

        readouts = [m for m in self.program.modules if isinstance(m, AbsReadout)]
        if len(readouts) != 1:
            raise NotImplementedError(
                f"SimEngine decimated path requires exactly one readout module; "
                f"the program has {len(readouts)}"
            )
        return readouts[0]

    def compute_decimated(self) -> list[NDArray[np.int64]]:
        """Compute one round's decimated time-domain trace (model A, lookback).

        The lookback timeline (reset + optional init pulse + readout) has no sweep
        (or a single point), so the engine lowers that one point, evolves the Bloch
        vector to the excited population ``P_e`` *before* readout, and renders the
        time-domain readout trace via :func:`decimated_trace` (model A: the readout
        envelope scaled by the steady mixed S21, shifted by ``sim.timeFly``).

        The trace is laid out as QICK's ``get_decimated`` expects for a single-rep
        single-trigger readout: a real/imag stacked array of shape ``(n_samples,
        2)``.  Fresh per-round Gaussian noise (same scale as the accumulated path)
        is added so software-averaging the rounds improves SNR; ``_summarize_decimated``
        means them.
        """

        self._raise_if_cancelled()
        f_qubit_ghz, rf_g, rf_e = self._operating_signal()
        self._raise_if_cancelled()

        # Single-point lowering: lookback declares no sweep, so the sweep
        # multi-index is empty and the Bloch timeline / readout plan are resolved
        # once.  (delta=0: decimated is not a coherence experiment, so the
        # quasi-static detune ensemble does not apply here.)
        point: dict[str, int] = {}
        lowered = self._lower(point, f_qubit_ghz, 0.0)
        self._raise_if_cancelled()
        v_final = bloch.evolve(
            bloch.ground_state(self.sim.thermal_pop), lowered.segments
        )
        p_e = float(np.clip(bloch.excited_population(v_final), 0.0, 1.0))

        # Readout window geometry from the single readout module + compiled ro_chs.
        readout = self._readout_module()
        if isinstance(readout, PulseReadout):
            ro_pulse_cfg = readout.cfg.pulse_cfg
            ro_cfg = readout.cfg.ro_cfg
        elif isinstance(readout, DirectReadout):
            raise NotImplementedError(
                "SimEngine decimated path requires a PulseReadout (its pulse_cfg "
                "defines the readout envelope shape); a DirectReadout has no pulse "
                "envelope to render in the time domain"
            )
        else:
            raise NotImplementedError(
                f"unsupported readout module {type(readout).__name__} for decimated"
            )

        pulse_length = _resolve_scalar(ro_pulse_cfg.waveform.length, {}, point)
        trig_offset = _resolve_scalar(ro_cfg.trig_offset, {}, point)

        # Program-time axis: cycles2us(get_time_axis) + trig_offset.  The decimated
        # sample count is the compiled readout window length (ro['length']); the
        # channel is the single readout channel.  This mirrors qick's
        # get_time_axis (cycles2us(ro_ch, arange(n))) so the engine's trace lands
        # on exactly the axis lookback plots.
        ((ro_ch, ro),) = self.program.ro_chs.items()
        n_samples = int(ro["length"])
        ts = (
            self.program.soccfg.cycles2us(np.arange(n_samples), ro_ch=ro_ch)
            + trig_offset
        )

        trace = decimated_trace(
            self.sim,
            ts,
            ro_pulse_cfg,
            pulse_length,
            lowered.readout.f_ro_ghz,
            rf_g,
            rf_e,
            p_e,
        )
        trace = readout_drive_amplitude(lowered.readout.readout_gain) * trace

        det = _FULL_SCALE * np.stack([trace.real, trace.imag], axis=-1)  # (n, 2)
        noise_std = _FULL_SCALE / self.sim.snr
        noise = self._rng.normal(0.0, noise_std, size=det.shape)
        return [np.rint(det + noise).astype(np.int64)]
