"""ro_optimize — 2D readout-optimisation Builder: argmax over freq × gain.

Synthesises a Gaussian peak landscape (``gaussian_peak_2d``) over a freq × gain
grid, finds the optimum via ``argmax`` (no fit — the peak location IS the result),
fills its Sweep2DResult row in place, and returns a Patch with ``best_ro_freq``
and ``best_ro_gain``, plus the ``opt_readout`` module constructed from them.

- requires the ``pi_pulse`` module (a pi-pulse is needed to prepare the excited
  state before measuring readout fidelity); placeholder default for the prototype.
- reads optional ``best_ro_freq`` and ``best_ro_gain`` (smoothed prev-point values
  used to plant the Gaussian centre so the optimum tracks across flux points), with
  sensible MHz defaults when absent.
- reads optional ``t1`` (smoothed prev-point T1) for the prototype only — unused
  in computation but declared to exercise the dependency mechanism.
- the ``readout`` module is optional (a base readout template); unused in the
  prototype body but declared to mirror the real experiment's dependency.

No fit step: the 2D landscape is computed in one shot (one effective "round"), so
``round_hook`` is called exactly once after filling the row.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any, Mapping, Optional

from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.plotters import Sweep2DPlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep2DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.nodes.synth import (
    accumulate_rounds,
    gaussian_peak_2d,
    parse_linear_axis,
    resolve_acquire_delay,
    resolve_rounds,
)

logger = logging.getLogger(__name__)

# Default axis specs: (start, stop, npts)
_DEFAULT_FREQ: tuple[float, float, int] = (4998.0, 5002.0, 21)
_DEFAULT_GAIN: tuple[float, float, int] = (0.3, 0.7, 21)

_DEFAULT_CENTER_FREQ = 5000.0  # MHz — baseline readout resonance
_DEFAULT_CENTER_GAIN = 0.5


def _default_t1() -> float:
    return 10.0


def _default_best_freq() -> float:
    return _DEFAULT_CENTER_FREQ


def _default_best_gain() -> float:
    return _DEFAULT_CENTER_GAIN


def _placeholder_pi_pulse() -> Any:
    # prototype placeholder — lenrabi produces the real module
    return {"type": "pi", "length": 0.1}


def _default_readout() -> Optional[Any]:
    return None


class RoOptimizeNode(Node):
    """One flux point's ro_optimize: synth 2D Gaussian → argmax → fill row → Patch.

    No fit step: ``gaussian_peak_2d`` is the landscape; ``argmax`` along each axis
    recovers (best_freq, best_gain). Fills ``result.signal[idx]``,
    ``result.best_freq[idx]``, ``result.best_gain[idx]``, and ``result.flux[idx]``
    in place, then calls ``round_hook`` once. Produces the ``opt_readout`` module
    from the argmax result.
    """

    def __init__(self, env: RunEnv) -> None:
        self._env = env

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env

        # read optional previous best to plant the Gaussian centre (tracks flux)
        prev_best_freq = float(snapshot["best_ro_freq"])
        prev_best_gain = float(snapshot["best_ro_gain"])
        _ = snapshot["t1"]  # declared optional; unused in computation
        _ = snapshot.module("pi_pulse")  # required — consumed by real hardware
        _ = snapshot.module("readout")  # optional — base readout template

        result: Sweep2DResult = env.result
        freqs = result.freq
        gains = result.gain

        # plant the peak slightly offset from the previous best so it drifts,
        # clamped to the gain grid so the argmax stays recoverable
        plant_freq = prev_best_freq + 0.2
        plant_gain = float(np.clip(prev_best_gain, gains[0], gains[-1]))

        idx = env.flux_idx
        result.flux[idx] = env.flux

        base = env.flux_idx * 1000

        def make_round(k: int) -> NDArray[np.float64]:
            return gaussian_peak_2d(freqs, gains, plant_freq, plant_gain, seed=base + k)

        def on_round(avg: NDArray[np.float64], _k: int) -> None:
            np.copyto(result.signal[idx], avg)
            if env.round_hook is not None:
                env.round_hook(idx)

        landscape = accumulate_rounds(
            make_round,
            resolve_rounds(env.params),
            on_round,
            delay=resolve_acquire_delay(env.params),
        )

        # argmax: project onto each axis and take the index of the max
        best_fi = int(np.argmax(landscape.max(axis=1)))
        best_gi = int(np.argmax(landscape.max(axis=0)))
        best_freq = float(freqs[best_fi])
        best_gain = float(gains[best_gi])

        result.best_freq[idx] = best_freq
        result.best_gain[idx] = best_gain

        logger.debug(
            "ro_optimize @flux%d: best_freq=%.3f best_gain=%.3f (plant freq=%.3f gain=%.3f)",
            idx,
            best_freq,
            best_gain,
            plant_freq,
            plant_gain,
        )

        patch = Patch()
        patch.set("best_ro_freq", best_freq)
        patch.set("best_ro_gain", best_gain)
        patch.set_module(
            "opt_readout", {"type": "readout", "freq": best_freq, "gain": best_gain}
        )
        return patch


class RoOptimizeBuilder(Builder):
    """The ro_optimize provider — 2D Gaussian synth, argmax (no fit), overwrite plot.

    Sweeps a freq × gain grid per flux point, synthesises a readout-fidelity
    landscape, and finds the optimum via argmax. No fitting: the Gaussian peak
    location IS the best readout point. Produces ``best_ro_freq``, ``best_ro_gain``,
    and the ``opt_readout`` module for downstream consumers (e.g. t1, mist).
    """

    name = "ro_optimize"
    provides = ("best_ro_freq", "best_ro_gain")
    provides_modules = ("opt_readout",)
    optional = (
        Dependency("t1", smooth="ewma", default=_default_t1),
        Dependency("best_ro_freq", default=_default_best_freq),
        Dependency("best_ro_gain", default=_default_best_gain),
    )
    requires_modules = (ModuleDep("pi_pulse", default=_placeholder_pi_pulse),)
    optional_modules = (ModuleDep("readout", default=_default_readout),)
    base_params = ("freq_expts", "gain_expts", "reps", "rounds", "acquire_delay")

    def make_init_result(self, params: Mapping[str, Any], n_flux: int) -> Sweep2DResult:
        freq_expts: int = int(params.get("freq_expts") or _DEFAULT_FREQ[2])
        gain_expts: int = int(params.get("gain_expts") or _DEFAULT_GAIN[2])

        freqs = parse_linear_axis(
            params.get("freq_range"),
            (_DEFAULT_FREQ[0], _DEFAULT_FREQ[1], freq_expts),
        )
        gains = parse_linear_axis(
            params.get("gain_range"),
            (_DEFAULT_GAIN[0], _DEFAULT_GAIN[1], gain_expts),
        )
        return Sweep2DResult.allocate(n_flux, freqs, gains)

    def make_plotter(self, figure: Any) -> Sweep2DPlotter:
        return Sweep2DPlotter(figure, title="ro_optimize")

    def build_node(self, env: RunEnv) -> RoOptimizeNode:
        return RoOptimizeNode(env)
