"""t2ramsey — Ramsey-fringe Builder: decay_cos → fit_decay_fringe → t2r.

Translates the notebook's T2RamseyTask cfg_maker. Synthesises a decaying cosine
fringe vs delay time (t2 planted from the smoothed previous t2r * 1.1, fringe
frequency fixed at 0.3 1/us), fits it with the real ``fit_decay_fringe``, fills
its sweep Result row in place, and returns the raw t2r and the measured detune.

- requires the ``pi2_pulse`` module (lenrabi produces it) — the Ramsey sequence
  needs a pi/2 pulse; it is a required module dep with a placeholder default for
  the prototype.
- reads ``t1`` (smooth="ewma") and ``t2r`` (smooth="ewma") as optional deps:
  ``t2r`` seeds the planted t2 so the sweep tracks a plausible decoherence time;
  ``t1`` is available for cfg sanity checks (not used directly in the prototype).
- the ``opt_readout`` module is optional (ro_optimize produces it → ml preset →
  default).
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any, Mapping, Optional

from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.plotters import Decay1DPlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.nodes.synth import (
    accumulate_rounds,
    decay_cos,
    flux_drift,
    flux_snr,
    is_good_fit,
    parse_linear_axis,
    resolve_acquire_delay,
    resolve_rounds,
    signal_to_real,
)
from zcu_tools.utils.fitting import fit_decay_fringe

logger = logging.getLogger(__name__)

_DEFAULT_T1 = 10.0  # us — smoothed t1 fallback
_DEFAULT_T2R = 5.0  # us — smoothed t2r fallback
_FRINGE_FREQ = 0.3  # 1/us — fixed planted fringe (detune) frequency
_DEFAULT_SWEEP = (0.0, 25.0, 121)  # delay-time axis (us): start, stop, npts


def _default_t1() -> float:
    return _DEFAULT_T1


def _default_t2r() -> float:
    return _DEFAULT_T2R


def _placeholder_pi2_pulse() -> Any:
    # prototype placeholder — lenrabi produces the real (placeholder) module
    return {"type": "pi2", "length": 0.05}


def _default_readout() -> Optional[Any]:
    return None


class T2RamseyNode(Node):
    """One flux point's t2ramsey: synth decay-cos fringe → fit_decay_fringe → fill row → Patch."""

    def __init__(self, env: RunEnv) -> None:
        self._env = env

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env
        _ = snapshot["t1"]  # optional smoothed t1 (available for cfg sanity)
        _ = float(
            snapshot["t2r"]
        )  # smoothed (declared smooth="ewma") — dependency contract
        _ = snapshot.module("pi2_pulse")  # required module — prototype does not use
        _ = snapshot.module("opt_readout")  # optional — prototype does not use

        result: Sweep1DResult = env.result
        times = result.x

        # t2r drifts parabolically with flux: ~5 us at sweet spot, up to ~20 us at
        # the edges. SNR varies sinusoidally to 0 at its troughs (dead points).
        true_t2 = flux_drift(env.flux, baseline=5.0, amplitude=15.0)
        snr = flux_snr(env.flux)

        idx = env.flux_idx
        result.flux[idx] = env.flux

        base = env.flux_idx * 1000

        def make_round(k: int) -> NDArray[np.complex128]:
            return decay_cos(times, true_t2, _FRINGE_FREQ, snr=snr, seed=base + k)

        def on_round(avg: NDArray[np.complex128], _k: int) -> None:
            np.copyto(result.signal[idx], signal_to_real(avg))
            if env.round_hook is not None:
                env.round_hook(idx)

        averaged = accumulate_rounds(
            make_round,
            resolve_rounds(env.params),
            on_round,
            delay=resolve_acquire_delay(env.params),
        )
        real = signal_to_real(averaged)

        t2f, _, detune, _, fit_curve, _ = fit_decay_fringe(times, real)

        if not is_good_fit(real, fit_curve):
            logger.debug(
                "t2ramsey fit @flux%d: poor fit (SNR-trough?) — discarded", idx
            )
            if env.round_hook is not None:
                env.round_hook(idx)  # raw row already shown; fit fields stay nan
            return Patch()  # partial: omit t2r/t2r_detune → downstream fallback

        result.fit_value[idx] = float(t2f)
        np.copyto(result.fit_curve[idx], np.asarray(fit_curve, dtype=np.float64))
        if env.round_hook is not None:
            env.round_hook(idx)

        logger.debug(
            "t2ramsey fit @flux%d: t2r=%.3f us detune=%.4f (plant t2=%.3f)",
            idx,
            float(t2f),
            float(detune),
            true_t2,
        )

        patch = Patch()
        patch.set("t2r", float(t2f))
        patch.set("t2r_detune", float(detune))
        return patch


class T2RamseyBuilder(Builder):
    """The t2ramsey provider — decay-cosine synth, real fit_decay_fringe, accumulating
    colormap.  Reports the raw Ramsey t2r and the measured detuning detune.
    """

    name = "t2ramsey"
    provides = ("t2r", "t2r_detune")
    optional = (
        Dependency("t1", smooth="ewma", default=_default_t1),
        Dependency("t2r", smooth="ewma", default=_default_t2r),
    )
    requires_modules = (ModuleDep("pi2_pulse", default=_placeholder_pi2_pulse),)
    optional_modules = (ModuleDep("opt_readout", default=_default_readout),)
    base_params = (
        "sweep_range",
        "num_expts",
        "detune_ratio",
        "reps",
        "rounds",
        "earlystop_snr",
        "acquire_delay",
    )

    def make_init_result(self, params: Mapping[str, Any], flux: Any) -> Sweep1DResult:
        times = parse_linear_axis(params.get("sweep_range"), _DEFAULT_SWEEP)
        return Sweep1DResult.allocate(flux, times, x_label="delay time (us)")

    def make_plotter(self, figure: Any) -> Decay1DPlotter:
        return Decay1DPlotter(
            figure, title="t2ramsey", value_label="T2Ramsey (us)", x_label="Time (us)"
        )

    def build_node(self, env: RunEnv) -> T2RamseyNode:
        return T2RamseyNode(env)
