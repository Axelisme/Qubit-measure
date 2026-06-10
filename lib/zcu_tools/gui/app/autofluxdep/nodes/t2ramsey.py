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

Phase B (cfg-builder): when the active context is configured for a real run (a
populated ml + the ``pi2_pulse`` drive module + the ``opt_readout`` readout module
both present on the snapshot), ``make_cfg`` lowers the context + this point's
snapshot into the base ``T2RamseyCfgTemplate`` — exercising the real
``ml.make_cfg`` pipeline — and ``produce`` takes the planted-t2 baseline from the
cfg's ``sweep_range`` (which encodes ``2.5 * prev_t2r``). The acquire is still
SIMULATED (no hardware); with the demo / empty-ml context the cfg is None and the
baseline falls back to ``_DEFAULT_T2R`` (the pure snapshot-driven path, unchanged).

Compare ``notebook_md/autofluxdep.md`` (the T2RamseyTask block):

    cfg_maker=lambda ctx, ml: (
        (info := ctx.env["info"])
        and (cur_t1 := info.get("smooth_t1", md.t1))                 # relax_delay
        and (prev_t2r := info.last.get("smooth_t2r", md.t2r))        # sweep_range
        and (cur_pi2_pulse := info.get("pi2_pulse"))                 # required module
        and (opt_readout := info.last.get("opt_readout", readout_cfg))  # optional
        and ml.make_cfg({"modules": {"pi2_pulse": cur_pi2_pulse,
                                     "readout": opt_readout},
                         "relax_delay": max(1.0, 3 * cur_t1),
                         "reps": 1000, "rounds": 10,
                         "sweep_range": (0, 2.5 * prev_t2r)}, ...) )
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any, Mapping, Optional

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.autofluxdep.t2ramsey import T2RamseyModuleCfg
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
from zcu_tools.program.v2 import ProgramV2Cfg
from zcu_tools.utils.fitting import fit_decay_fringe

logger = logging.getLogger(__name__)

_DEFAULT_T1 = 10.0  # us — smoothed t1 fallback
_DEFAULT_T2R = 5.0  # us — smoothed t2r fallback
_FRINGE_FREQ = 0.3  # 1/us — fixed planted fringe (detune) frequency
_DEFAULT_SWEEP = (0.0, 25.0, 121)  # delay-time axis (us): start, stop, npts
_SWEEP_T2R_FACTOR = 2.5  # notebook: sweep_range = (0, 2.5 * prev_t2r)

# the real cfg ``type`` tags make_cfg can lower: a configured run supplies a
# concrete pi/2 ``pulse`` + a ``readout/*`` module, whereas the prototype workflow
# flows placeholder modules ({"type": "pi2"} / {"type": "readout"}) that are NOT
# valid PulseCfg / ReadoutCfg — so the guard routes only the former through the cfg
# pipeline and keeps the placeholder context on the pure synthetic path.
_REAL_PULSE_TYPE = "pulse"
_REAL_READOUT_TYPES = ("readout/pulse", "readout/direct")


def _module_type(module: Any) -> Optional[str]:
    """The ``type`` tag of a snapshot module (raw dict or built cfg), or None."""
    if module is None:
        return None
    if isinstance(module, Mapping):
        value = module.get("type")
    else:
        value = getattr(module, "type", None)
    return value if isinstance(value, str) else None


class T2RamseyCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    """The base Ramsey cfg t2ramsey lowers a context into.

    ``ProgramV2Cfg`` (reps/rounds/relax) + the ``ExpCfgModel`` device/save fields,
    plus the Ramsey ``modules`` (``pi2_pulse`` + ``readout``, optional ``reset``)
    and a free ``sweep_range`` (the delay-time span) — mirroring the lower-layer
    ``experiment/v2/autofluxdep`` ``T2RamseyCfgTemplate``. The flux ``dev`` entry,
    the concrete ``length`` sweep, and ``activate_detune`` are merged in by the
    lower-layer ``run()`` (not here): this template is the cfg-maker output, and
    ``produce`` reads the planted-t2 baseline from ``sweep_range``.
    """

    modules: T2RamseyModuleCfg
    sweep_range: tuple[float, float]


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

    def __init__(self, env: RunEnv, builder: "T2RamseyBuilder") -> None:
        self._env = env
        self._builder = builder

    def _maybe_make_cfg(self, snapshot: Snapshot) -> Optional[T2RamseyCfgTemplate]:
        """Build the run cfg when the context is configured for it, else None.

        ``make_cfg`` needs a populated ml + a *concrete* ``pi2_pulse`` drive cfg +
        a *concrete* ``opt_readout`` readout cfg on the snapshot. The default /
        demo context has an empty ml; the prototype workflow flows placeholder
        modules ({"type": "pi2"} / {"type": "readout"}) that are not valid
        PulseCfg / ReadoutCfg — both keep the pure snapshot-driven simulation
        (t2ramsey has no raw drive 設定頭 param to gate on, unlike qubit_freq,
        because its pi/2 pulse arrives pre-built as a module, so the gate is the
        module being a real cfg type). No hardware is touched either way — Phase B
        simulates the acquire uniformly; routing through ``make_cfg`` (when
        configured) exercises the real cfg pipeline and makes the cfg the source
        of the planted-t2 baseline (its ``sweep_range``).
        """
        env = self._env
        if env.ml is None:
            return None
        if _module_type(snapshot.module("pi2_pulse")) != _REAL_PULSE_TYPE:
            return None
        if _module_type(snapshot.module("opt_readout")) not in _REAL_READOUT_TYPES:
            return None
        return self._builder.make_cfg(env, snapshot)

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env
        _ = snapshot["t1"]  # optional smoothed t1 (available for cfg sanity)
        prev_t2r = float(
            snapshot["t2r"]
        )  # smoothed (declared smooth="ewma") — dependency contract
        _ = snapshot.module("pi2_pulse")  # required module — lowered into the cfg
        _ = snapshot.module("opt_readout")  # optional — lowered into the cfg

        # Build the run cfg from the active context (when configured) and take the
        # planted-t2 baseline from its sweep_range (= 2.5 * prev_t2r); the acquire is
        # SIMULATED below. With the demo / empty-ml context the cfg is None and the
        # baseline is the default (5 us) — the pure snapshot-driven path, unchanged.
        cfg = self._maybe_make_cfg(snapshot)
        if cfg is not None:
            t2r_baseline = float(cfg.sweep_range[1]) / _SWEEP_T2R_FACTOR
        else:
            t2r_baseline = _DEFAULT_T2R
        del prev_t2r  # the cfg's sweep_range carries the same smoothed t2r

        result: Sweep1DResult = env.result
        times = result.x

        # t2r drifts parabolically with flux: ~t2r_baseline at the sweet spot, up to
        # ~+15 us at the edges. SNR varies sinusoidally to 0 at its troughs (dead
        # points). normalised sweep position (0→1): the drift/SNR shapes live on
        # [0,1], while real flux values are tiny — use the position so the
        # whole sweep spans one drift parabola and a few SNR cycles.
        pos = env.flux_idx / max(1, result.n_flux - 1)
        true_t2 = flux_drift(pos, baseline=t2r_baseline, amplitude=15.0)
        snr = flux_snr(pos)

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
        return T2RamseyNode(env, self)

    def make_cfg(self, env: RunEnv, snapshot: Snapshot) -> T2RamseyCfgTemplate:
        """Lower the active context + this point's snapshot into the base run cfg.

        Mirrors the notebook's t2ramsey ``cfg_maker`` (runs in ``produce``, where
        the snapshot is available): the ``pi2_pulse`` drive module and the
        ``readout`` module come from the snapshot (lenrabi / ro_optimize produce
        them, ml-preset / default otherwise), ``relax_delay`` is ``3 * t1`` (the
        smoothed t1 from the snapshot, floored at 1 us), the ``sweep_range`` spans
        ``2.5 * t2r`` (the smoothed t2r), and ``reps`` / ``rounds`` come from the
        node's params. The flux ``dev`` entry, the concrete ``length`` sweep, and
        ``activate_detune`` are NOT here — the lower-layer ``run()`` merges them.

        Raises if the ml / drive / readout modules are unavailable — a real run
        needs a concrete Ramsey sequence (Fast Fail), unlike the synthetic path
        which fabricates a signal.
        """
        params = env.params
        ml = env.ml
        if ml is None:
            raise RuntimeError("t2ramsey.make_cfg needs an active ModuleLibrary")
        pi2_pulse = snapshot.module("pi2_pulse")
        if pi2_pulse is None:
            raise RuntimeError(
                "t2ramsey.make_cfg needs a pi2_pulse module (none produced or preset)"
            )
        readout = snapshot.module("opt_readout")
        if readout is None:
            raise RuntimeError(
                "t2ramsey.make_cfg needs a readout module (none produced or preset)"
            )
        t1 = float(snapshot["t1"])
        t2r = float(snapshot["t2r"])
        return ml.make_cfg(
            {
                "modules": {
                    "pi2_pulse": pi2_pulse,
                    "readout": readout,
                },
                "relax_delay": max(1.0, 3.0 * t1),
                "reps": int(params.get("reps", 1000)),
                "rounds": int(params.get("rounds", 10)),
                "sweep_range": (0.0, _SWEEP_T2R_FACTOR * t2r),
            },
            T2RamseyCfgTemplate,
        )
