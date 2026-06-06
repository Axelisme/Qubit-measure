"""t1 — the simplest 1D Builder: exp decay → fit_decay → t1.

Translates the notebook's T1Task cfg_maker. Synthesises an exponential decay vs
relax time (time constant planted from the smoothed previous t1), fits it with
the real ``fit_decay``, fills its sweep Result row in place, and returns the raw
``t1`` Patch.

- needs the ``pi_pulse`` module (lenrabi produces it) — without a pi-pulse there
  is no excited state to relax. In the prototype it carries a placeholder
  default, so it never actually skips; Phase B drops the default (real lenrabi
  output) to restore true skip-if-absent.
- reads ``t1`` declared ``smooth="ewma"`` (the notebook's smooth_t1) for the
  relax_delay guess + the planted decay constant; reports raw ``t1`` back.
- the ``opt_readout`` module is optional (ro_optimize produces it → ml preset →
  default).

The smoothing is consumer-declared and same-key: this Builder reads ``t1``
smoothed and provides raw ``t1`` — the orchestrator's SmoothingService projects
the smoothed estimate under the same key for the next point's readers (lenrabi /
ro_optimize / t2*), so no separate ``smooth_t1`` key exists.
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
    exp_decay,
    flux_drift,
    flux_snr,
    is_good_fit,
    parse_linear_axis,
    resolve_acquire_delay,
    resolve_rounds,
    signal_to_real,
)
from zcu_tools.utils.fitting import fit_decay

logger = logging.getLogger(__name__)

_DEFAULT_T1 = 10.0  # us — md.t1 stand-in (the smoothed-t1 fallback)
_DEFAULT_SWEEP = (0.5, 60.0, 101)  # relax-time axis (us): start, stop, npts


def _default_t1() -> float:
    return _DEFAULT_T1


def _placeholder_pi_pulse() -> Any:
    # prototype placeholder — lenrabi produces the real (placeholder) module
    return {"type": "pi", "length": 0.1}


def _default_readout() -> Optional[Any]:
    return None


class T1Node(Node):
    """One flux point's t1: synth exp decay → fit_decay → fill row → Patch."""

    def __init__(self, env: RunEnv) -> None:
        self._env = env

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env
        _ = snapshot["t1"]  # smoothed (declared smooth="ewma") — dependency contract
        _ = snapshot.module("pi_pulse"), snapshot.module("opt_readout")

        result: Sweep1DResult = env.result
        times = result.x

        # t1 drifts parabolically with flux: ~10 us at the sweet spot, up to ~50 us
        # at the edges. SNR varies sinusoidally to 0 at its troughs (dead points).
        # normalised sweep position (0→1): the drift/SNR shapes live on
        # [0,1], while real flux values are tiny — use the position so the
        # whole sweep spans one drift parabola and a few SNR cycles.
        pos = env.flux_idx / max(1, result.n_flux - 1)
        true_t1 = flux_drift(pos, baseline=10.0, amplitude=40.0)
        snr = flux_snr(pos)

        idx = env.flux_idx
        result.flux[idx] = env.flux

        base = env.flux_idx * 1000

        def make_round(k: int) -> NDArray[np.complex128]:
            return exp_decay(times, true_t1, snr=snr, seed=base + k)

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

        t1, _t1err, fit_curve, _ = fit_decay(times, real)

        if not is_good_fit(real, fit_curve):
            logger.debug("t1 fit @flux%d: poor fit (SNR-trough?) — discarded", idx)
            if env.round_hook is not None:
                env.round_hook(idx)  # raw row already shown; fit fields stay nan
            return Patch()  # partial: omit t1 → downstream fallback

        result.fit_value[idx] = float(t1)
        np.copyto(result.fit_curve[idx], np.asarray(fit_curve, dtype=np.float64))
        if env.round_hook is not None:
            env.round_hook(idx)

        logger.debug(
            "t1 fit @flux%d: t1=%.3f us (plant=%.3f)",
            idx,
            float(t1),
            true_t1,
        )

        patch = Patch()
        patch.set("t1", float(t1))
        return patch


class T1Builder(Builder):
    """The t1 provider — exp-decay synth, real fit_decay, accumulating colormap."""

    name = "t1"
    provides = ("t1",)
    optional = (Dependency("t1", smooth="ewma", default=_default_t1),)
    requires_modules = (ModuleDep("pi_pulse", default=_placeholder_pi_pulse),)
    optional_modules = (ModuleDep("opt_readout", default=_default_readout),)
    base_params = (
        "sweep_range",
        "num_expts",
        "reps",
        "rounds",
        "earlystop_snr",
        "acquire_delay",
    )

    def make_init_result(self, params: Mapping[str, Any], flux: Any) -> Sweep1DResult:
        times = parse_linear_axis(params.get("sweep_range"), _DEFAULT_SWEEP)
        return Sweep1DResult.allocate(flux, times, x_label="relax time (us)")

    def make_plotter(self, figure: Any) -> Decay1DPlotter:
        return Decay1DPlotter(
            figure, title="t1", value_label="T1 (us)", x_label="Time (us)"
        )

    def build_node(self, env: RunEnv) -> T1Node:
        return T1Node(env)
