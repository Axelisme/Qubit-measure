"""mist — 1D gain-sweep Builder: variance curve readout (no fit).

Synthesises a state-disturbance curve (``variance_curve``) over a gain axis,
reads the variance directly — there is no fit step — and fills its Sweep1DResult
row in place. ``fit_value`` and ``fit_curve`` remain nan (allocated as nan by
``Sweep1DResult.allocate``); the ``ColormapLinePlotter`` shows the flux × gain
colormap with the latest flux rows as traces (no fit marker).

- requires the ``pi_pulse`` module (pi-pulse prepares the excited state whose
  disturbance the variance measures); placeholder default for the prototype.
- the ``opt_readout`` module is optional (ro_optimize produces it); unused in the
  prototype body but declared to mirror the real experiment's dependency.

No fit step: the variance curve is a monotone logistic ramp; MIST reads the
variance magnitude directly (the real pipeline reads IQ scatter). The row is
considered complete after one compute step, so ``round_hook`` is called once.
Provides ``success=1.0`` (float, consistent with the info-value domain) to signal
that the MIST pass completed without a hardware error.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any, Mapping, Optional

from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.plotters import ColormapLinePlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import ModuleDep
from zcu_tools.gui.app.autofluxdep.nodes.synth import (
    accumulate_rounds,
    parse_linear_axis,
    resolve_acquire_delay,
    resolve_rounds,
    variance_curve,
)

logger = logging.getLogger(__name__)

_DEFAULT_GAIN_SWEEP: tuple[float, float, int] = (0.0, 1.0, 51)
_ONSET_GAIN = 0.5  # fixed onset for the prototype variance curve


def _placeholder_pi_pulse() -> Any:
    # prototype placeholder — lenrabi produces the real module
    return {"type": "pi", "length": 0.1}


def _default_opt_readout() -> Optional[Any]:
    return None


class MistNode(Node):
    """One flux point's MIST: synth variance curve → fill row → Patch.

    No fit step: ``variance_curve`` returns a real (n_gain,) magnitude directly.
    ``fit_value[idx]`` and ``fit_curve[idx]`` are left as nan (already the
    allocate default), so the ``ColormapLinePlotter`` shows only the colormap
    (no fit marker). Calls ``round_hook`` once per round while filling the row.
    """

    def __init__(self, env: RunEnv) -> None:
        self._env = env

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env

        _ = snapshot.module("pi_pulse")  # required — consumed by real hardware
        _ = snapshot.module("opt_readout")  # optional — optimised readout preset

        result: Sweep1DResult = env.result
        gains = result.x

        idx = env.flux_idx
        result.flux[idx] = env.flux
        # fit_value[idx] and fit_curve[idx] remain nan — mist has no fit scalar

        base = env.flux_idx * 1000

        def make_round(k: int) -> NDArray[np.float64]:
            return variance_curve(gains, _ONSET_GAIN, seed=base + k)

        def on_round(avg: NDArray[np.float64], _k: int) -> None:
            np.copyto(result.signal[idx], avg)
            if env.round_hook is not None:
                env.round_hook(idx)

        curve = accumulate_rounds(
            make_round,
            resolve_rounds(env.params),
            on_round,
            delay=resolve_acquire_delay(env.params),
        )

        logger.debug(
            "mist @flux%d: success, variance range [%.3f, %.3f]",
            idx,
            float(curve.min()),
            float(curve.max()),
        )

        patch = Patch()
        patch.set("success", 1.0)  # float: consistent with info-value domain
        return patch


class MistBuilder(Builder):
    """The MIST provider — variance-curve synth, no fit, accumulating colormap.

    Sweeps a gain axis per flux point, synthesises a state-disturbance curve, and
    records the variance directly (no fit). ``fit_value`` stays nan so the
    ``ColormapLinePlotter`` renders only the flux×gain colormap. Provides ``success``
    (float 1.0) to signal that the MIST pass completed; the ``opt_readout``
    module is consumed (from ro_optimize) to configure the readout during the real
    measurement.
    """

    name = "mist"
    provides = ("success",)
    provides_modules: tuple[str, ...] = ()
    requires_modules = (ModuleDep("pi_pulse", default=_placeholder_pi_pulse),)
    optional_modules = (ModuleDep("opt_readout", default=_default_opt_readout),)
    base_params = ("gain_sweep", "reps", "rounds", "acquire_delay")

    def make_init_result(self, params: Mapping[str, Any], flux: Any) -> Sweep1DResult:
        gains = parse_linear_axis(params.get("gain_sweep"), _DEFAULT_GAIN_SWEEP)
        return Sweep1DResult.allocate(flux, gains, x_label="gain")

    def make_plotter(self, figure: Any) -> ColormapLinePlotter:
        return ColormapLinePlotter(
            figure, title="mist", y_label="Readout Gain (a.u.)", num_lines=1
        )

    def build_node(self, env: RunEnv) -> MistNode:
        return MistNode(env)
