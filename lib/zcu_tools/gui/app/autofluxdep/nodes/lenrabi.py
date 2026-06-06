"""lenrabi — length-Rabi Builder: rabi_oscillation → fit_rabi → pi/pi2 lengths.

Translates the notebook's LenRabiTask cfg_maker. Synthesises a cosine Rabi
oscillation vs pulse length (rabi_freq planted near 0.5 1/us), fits it with the
real ``fit_rabi``, fills its sweep Result row in place, and returns the raw pi
and pi2 lengths plus the Rabi frequency.

- requires ``qubit_freq`` (a hard require via Dependency): the Rabi experiment
  drives the qubit on resonance, so no qubit frequency → no sensible cfg.
- reads ``smooth_pi_product`` declared ``smooth="step_weighted"`` (the notebook's
  smooth_m / smooth_step_weighted, used for cfg tuning); treated as optional with
  default 0.3 — if absent the prototype ignores it (prototype doesn't build a
  real cfg).
- the ``opt_readout`` module is optional (ro_optimize produces it → ml preset →
  default).
- provides the ``pi_pulse`` and ``pi2_pulse`` modules (placeholder dicts in the
  prototype; the real impl would fill proper PulseReadoutCfg objects).
"""

from __future__ import annotations

import logging

import numpy as np
from typing_extensions import Any, Mapping, Optional

from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.plotters import Sweep1DPlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.nodes.synth import (
    parse_linear_axis,
    rabi_oscillation,
    resolve_acquire_delay,
    signal_to_real,
    simulate_acquire_delay,
)
from zcu_tools.utils.fitting import fit_rabi

logger = logging.getLogger(__name__)

_DEFAULT_RABI_FREQ = 0.5  # 1/us — planted Rabi frequency for the prototype
_DEFAULT_SWEEP = (0.0, 6.0, 121)  # pulse-length axis (us): start, stop, npts


def _default_smooth_pi_product() -> float:
    return 0.3


def _default_readout() -> Optional[Any]:
    return None


class LenRabiNode(Node):
    """One flux point's lenrabi: synth Rabi oscillation → fit_rabi → fill row → Patch."""

    def __init__(self, env: RunEnv) -> None:
        self._env = env

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env
        _ = snapshot["qubit_freq"]  # required — drives on resonance
        _ = snapshot["smooth_pi_product"]  # optional smoothed product param
        _ = snapshot.module("opt_readout")  # optional — prototype does not use

        result: Sweep1DResult = env.result
        lengths = result.x

        # plant a Rabi frequency near the default; the sweep should recover it
        rabi_freq = _DEFAULT_RABI_FREQ
        signals = rabi_oscillation(lengths, rabi_freq, seed=env.flux_idx)
        real = signal_to_real(signals)

        idx = env.flux_idx
        result.flux[idx] = env.flux
        np.copyto(result.signal[idx], real)
        if env.round_hook is not None:
            env.round_hook(idx)

        # emulate the acquire's wall-clock cost so the liveplot advances visibly
        simulate_acquire_delay(resolve_acquire_delay(env.params))

        pi_x, _, pi2_x, _, freq, _, fit_curve, _ = fit_rabi(lengths, real)
        result.fit_value[idx] = float(pi_x)
        np.copyto(result.fit_curve[idx], np.asarray(fit_curve, dtype=np.float64))
        if env.round_hook is not None:
            env.round_hook(idx)

        logger.debug(
            "lenrabi fit @flux%d: rabi_freq=%.4f pi_len=%.3f pi2_len=%.3f",
            idx,
            float(freq),
            float(pi_x),
            float(pi2_x),
        )

        patch = Patch()
        patch.set("pi_length", float(pi_x))
        patch.set("pi2_length", float(pi2_x))
        patch.set("rabi_freq", float(freq))
        patch.set_module("pi_pulse", {"type": "pi", "length": float(pi_x)})
        patch.set_module("pi2_pulse", {"type": "pi2", "length": float(pi2_x)})
        return patch


class LenRabiBuilder(Builder):
    """The lenrabi provider — Rabi-oscillation synth, real fit_rabi, accumulating
    colormap.  Produces pi_pulse and pi2_pulse module placeholders in addition to
    the scalar pi_length / pi2_length / rabi_freq info values.
    """

    name = "lenrabi"
    provides = ("pi_length", "pi2_length", "rabi_freq")
    provides_modules = ("pi_pulse", "pi2_pulse")
    requires = (Dependency("qubit_freq"),)
    optional = (
        Dependency(
            "smooth_pi_product",
            smooth="step_weighted",
            default=_default_smooth_pi_product,
        ),
    )
    optional_modules = (ModuleDep("opt_readout", default=_default_readout),)
    base_params = (
        "sweep_range",
        "num_expts",
        "reps",
        "rounds",
        "earlystop_snr",
        "acquire_delay",
    )

    def make_init_result(self, params: Mapping[str, Any], n_flux: int) -> Sweep1DResult:
        lengths = parse_linear_axis(params.get("sweep_range"), _DEFAULT_SWEEP)
        return Sweep1DResult.allocate(n_flux, lengths, x_label="pulse length (us)")

    def make_plotter(self, figure: Any) -> Sweep1DPlotter:
        return Sweep1DPlotter(figure, title="lenrabi", value_label="pi_length (us)")

    def build_node(self, env: RunEnv) -> LenRabiNode:
        return LenRabiNode(env)
