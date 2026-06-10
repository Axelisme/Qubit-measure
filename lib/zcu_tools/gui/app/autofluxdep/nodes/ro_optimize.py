"""ro_optimize — 2D readout-optimisation Builder: argmax over freq × gain.

Synthesises a Gaussian peak landscape (``gaussian_peak_2d``) over a freq × gain
grid, finds the optimum via ``argmax`` (no fit — the peak location IS the result),
fills its Sweep2DResult row in place, and returns a Patch with ``best_ro_freq``
and ``best_ro_gain``, plus the ``opt_readout`` module constructed from them.

- requires the ``pi_pulse`` module (a pi-pulse is needed to prepare the excited
  state before measuring readout fidelity); placeholder default for the prototype.
- reads optional ``best_ro_freq`` and ``best_ro_gain`` (raw prev-point values —
  no smoothing flag: the tracking loop deliberately follows the actual last best
  to plant the Gaussian centre so the optimum tracks across flux points), with
  sensible MHz defaults when absent.
- reads optional ``t1`` (smoothed prev-point T1) for the relax_delay (3·T1) and
  to exercise the dependency mechanism.
- the ``readout`` module is optional (a base readout template); it is the readout
  the cfg sweeps over (its freq/gain are swept), mirroring the real experiment.

No fit step: the 2D landscape is computed in one shot (one effective "round"), so
``round_hook`` is called exactly once after filling the row.

Phase B — cfg-driven simulation. ``produce`` lowers the active context + this
point's snapshot into a runnable ``RoOptimizeCfgTemplate`` via the Builder's
``make_cfg`` WHEN the context is configured (a populated ml + the ``pi_pulse`` and
``readout`` modules on the snapshot), exercising the real ``ml.make_cfg`` lowering
pipeline. The cfg's ``freq_range`` / ``gain_range`` (centred on the previous best,
mirroring the notebook ``RO_OptTask`` ``cfg_maker``) supply the plant centre of
the synthetic landscape; the acquire itself is still SIMULATED (no hardware). With
the demo / empty-ml context the cfg is None and the centre is the previous best
directly — the existing pure snapshot-driven simulation, unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.plotters import Landscape2DPlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep2DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.nodes.synth import (
    accumulate_rounds,
    gaussian_peak_2d,
    parse_linear_axis,
    resolve_acquire_delay,
    resolve_rounds,
)
from zcu_tools.program.v2 import ProgramV2Cfg, PulseCfg, PulseReadoutCfg, ResetCfg

logger = logging.getLogger(__name__)

# Default axis specs: (start, stop, npts)
_DEFAULT_FREQ: tuple[float, float, int] = (4998.0, 5002.0, 21)
_DEFAULT_GAIN: tuple[float, float, int] = (0.3, 0.7, 21)

_DEFAULT_CENTER_FREQ = 5000.0  # MHz — baseline readout resonance
_DEFAULT_CENTER_GAIN = 0.5

# the cfg sweep-window half-widths (the "設定頭"): the notebook centres the
# freq_range on the previous best ± ``0.2 * md.rf_w`` and the gain_range on the
# previous best ± ``0.05``. The GUI exposes those half-widths as params; these are
# their defaults when unset.
_DEFAULT_FREQ_WINDOW = 1.0  # MHz half-width of the readout-freq sweep window
_DEFAULT_GAIN_WINDOW = 0.05  # half-width of the readout-gain sweep window
_DEFAULT_T1 = 10.0  # us — fallback T1 for the relax_delay (3·T1)


class RoOptimizeModuleCfg(ConfigBase):
    """The modules ro_optimize lowers — an optional reset + the pi-pulse + readout.

    Mirrors the lower-layer ``experiment/v2/autofluxdep`` ``RO_OptModuleCfg``: the
    ``pi_pulse`` prepares the excited state and the ``readout`` is the pulse whose
    ``freq`` / ``gain`` the sweep optimises over.
    """

    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    readout: PulseReadoutCfg


class RoOptimizeCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    """The base program cfg ro_optimize lowers a context into.

    ``ProgramV2Cfg`` (reps/rounds/relax) + the ``ExpCfgModel`` device fields, plus
    the ``modules`` (pi_pulse + readout) and the ``freq_range`` / ``gain_range``
    sweep windows — exactly the lower-layer ``RO_OptCfgTemplate``. The flux ``dev``
    entry and the concrete ``freq`` / ``gain`` ``SweepCfg`` are merged in by the
    lower-layer ``run`` (the GUI prototype reads the window centres straight off
    the template to plant the synthetic landscape); ``freq_range`` / ``gain_range``
    are stripped before the runnable ``RO_OptCfg`` is validated downstream.
    """

    modules: RoOptimizeModuleCfg
    freq_range: tuple[float, float]
    gain_range: tuple[float, float]


def _default_t1() -> float:
    return _DEFAULT_T1


def _default_best_freq() -> float:
    return _DEFAULT_CENTER_FREQ


def _default_best_gain() -> float:
    return _DEFAULT_CENTER_GAIN


def _placeholder_pi_pulse() -> Any:
    # prototype placeholder — lenrabi produces the real module
    return {"type": "pi", "length": 0.1}


def _default_readout() -> Any | None:
    return None


def _resolve_window(value: Any, default: float) -> float:
    """The half-width of a cfg sweep window from a param, or ``default`` if unset.

    The prototype's param fields are free text, so a missing / unparseable value
    degrades to the default rather than failing make_cfg."""
    if value is None or value == "":
        return default
    try:
        return abs(float(value))
    except (ValueError, TypeError):
        return default


class RoOptimizeNode(Node):
    """One flux point's ro_optimize: synth 2D Gaussian → argmax → fill row → Patch.

    No fit step: ``gaussian_peak_2d`` is the landscape; ``argmax`` along each axis
    recovers (best_freq, best_gain). Fills ``result.signal[idx]``,
    ``result.best_freq[idx]``, ``result.best_gain[idx]``, and ``result.flux[idx]``
    in place, then calls ``round_hook`` once. Produces the ``opt_readout`` module
    from the argmax result.

    When the active context is configured (a populated ml + the pi_pulse / readout
    modules) ``produce`` lowers it into ``RoOptimizeCfgTemplate`` via the Builder's
    ``make_cfg`` and takes the plant-centre freq / gain from the cfg's sweep-window
    midpoints; the acquire is SIMULATED either way (no hardware).
    """

    def __init__(self, env: RunEnv, builder: RoOptimizeBuilder) -> None:
        self._env = env
        self._builder = builder

    def _maybe_make_cfg(self, snapshot: Snapshot) -> RoOptimizeCfgTemplate | None:
        """Build the run cfg when the context is configured for it, else None.

        ``make_cfg`` needs a populated ml + the ``pi_pulse`` (required) and
        ``readout`` (optional, the swept pulse) modules; the default / demo
        context (empty ml, ``readout`` resolving to None) has neither, so produce
        keeps the pure snapshot-driven simulation there. No hardware is touched
        either way — Phase B simulates the acquire uniformly; routing through
        ``make_cfg`` (when configured) exercises the real cfg pipeline and makes
        the cfg the source of the plant-centre freq / gain.
        """
        env = self._env
        if (
            env.ml is None
            or snapshot.module("pi_pulse") is None
            or snapshot.module("readout") is None
        ):
            return None
        return self._builder.make_cfg(env, snapshot)

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env

        # read optional previous best to plant the Gaussian centre (tracks flux)
        prev_best_freq = float(snapshot["best_ro_freq"])
        prev_best_gain = float(snapshot["best_ro_gain"])
        _ = snapshot["t1"]  # declared optional; relax_delay = 3·T1 in make_cfg
        _ = snapshot.module("pi_pulse")  # required — consumed by real hardware
        _ = snapshot.module("readout")  # optional — the swept readout template

        # Build the run cfg from the active context (when configured) and take the
        # plant-centre freq / gain from its sweep windows; the acquire is SIMULATED
        # below. With the demo / empty-ml context the cfg is None and the centre is
        # the previous best directly (same value — make_cfg centres freq_range /
        # gain_range on the previous best, mirroring the notebook cfg_maker).
        cfg = self._maybe_make_cfg(snapshot)
        if cfg is not None:
            center_freq = 0.5 * (cfg.freq_range[0] + cfg.freq_range[1])
            center_gain = 0.5 * (cfg.gain_range[0] + cfg.gain_range[1])
        else:
            center_freq = prev_best_freq
            center_gain = prev_best_gain

        result: Sweep2DResult = env.result
        freqs = result.freq
        gains = result.gain

        # plant the peak slightly offset from the centre so it drifts, clamped to
        # BOTH grids so the planted centre stays inside the sweep window —
        # otherwise the +0.2/point freq drift walks off the freq grid after ~10
        # flux points and the argmax pins to the boundary.
        plant_freq = float(np.clip(center_freq + 0.2, freqs[0], freqs[-1]))
        plant_gain = float(np.clip(center_gain, gains[0], gains[-1]))

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
    base_params = (
        "freq_expts",
        "gain_expts",
        "reps",
        "rounds",
        "acquire_delay",
        # the cfg sweep-window half-widths the cfg builder lowers into
        # freq_range / gain_range (centred on the previous best). The freq /
        # gain centres come from the snapshot; the rest from these params.
        "freq_window",
        "gain_window",
    )

    def make_init_result(self, params: Mapping[str, Any], flux: Any) -> Sweep2DResult:
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
        return Sweep2DResult.allocate(flux, freqs, gains)

    def make_plotter(self, figure: Any) -> Landscape2DPlotter:
        return Landscape2DPlotter(figure, title="ro_optimize")

    def build_node(self, env: RunEnv) -> RoOptimizeNode:
        return RoOptimizeNode(env, self)

    def make_cfg(self, env: RunEnv, snapshot: Snapshot) -> RoOptimizeCfgTemplate:
        """Lower the active context + this point's snapshot into the base run cfg.

        Mirrors the notebook's ro_optimize ``cfg_maker`` (runs in ``produce``,
        where the snapshot is available): the ``pi_pulse`` and ``readout`` modules
        come whole from the snapshot (lenrabi produces the pi-pulse; the readout is
        the base template), the relax_delay is ``3·T1`` (the smoothed prev-point
        T1), and the ``freq_range`` / ``gain_range`` are the previous best ± the
        window half-widths (the "設定頭"). The flux ``dev`` entry and the concrete
        ``freq`` / ``gain`` sweeps are NOT here — the lower-layer ``run`` merges
        them.

        Raises if the ml is unavailable or the pi_pulse / readout modules are
        unset — a real run needs both concrete modules (Fast Fail), unlike the
        synthetic path which fabricates a landscape.
        """
        params = env.params
        ml = env.ml
        if ml is None:
            raise RuntimeError("ro_optimize.make_cfg needs an active ModuleLibrary")
        pi_pulse = snapshot.module("pi_pulse")
        readout = snapshot.module("readout")
        if pi_pulse is None or readout is None:
            raise RuntimeError(
                "ro_optimize.make_cfg needs the pi_pulse + readout modules "
                "(none produced or preset)"
            )
        prev_best_freq = float(snapshot["best_ro_freq"])
        prev_best_gain = float(snapshot["best_ro_gain"])
        t1 = float(snapshot["t1"])
        freq_window = _resolve_window(params.get("freq_window"), _DEFAULT_FREQ_WINDOW)
        gain_window = _resolve_window(params.get("gain_window"), _DEFAULT_GAIN_WINDOW)
        return ml.make_cfg(
            {
                "modules": {
                    "pi_pulse": pi_pulse,
                    "readout": readout,
                },
                "relax_delay": 3.0 * t1,
                "reps": int(params.get("reps", 1000)),
                "rounds": int(params.get("rounds", 10)),
                "freq_range": (
                    prev_best_freq - freq_window,
                    prev_best_freq + freq_window,
                ),
                "gain_range": (
                    max(0.0, prev_best_gain - gain_window),
                    min(1.0, prev_best_gain + gain_window),
                ),
            },
            RoOptimizeCfgTemplate,
        )
