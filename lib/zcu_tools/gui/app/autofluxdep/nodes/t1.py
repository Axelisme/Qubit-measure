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

Phase B (cfg-builder): ``make_cfg`` lowers the active context + this point's
snapshot into a runnable ``T1CfgTemplate`` (no acquire — construction only),
mirroring the notebook's T1Task ``cfg_maker``:

    cfg_maker=lambda ctx, ml: (
        (info := ctx.env["info"])
        and (prev_t1 := info.last.get("smooth_t1", md.t1))          # smoothed t1
        and (cur_pi_pulse := info.get("pi_pulse"))                  # pi_pulse module
        and (opt_readout := info.last.get("opt_readout", readout_cfg))  # readout module
        and ml.make_cfg(
            {"modules": {"pi_pulse": cur_pi_pulse, "readout": opt_readout},
             "relax_delay": max(1.0, 3 * prev_t1),
             "reps": 1000, "rounds": 10,
             "sweep_range": (0.5, max(1.0, 5 * prev_t1))},
            zefd.T1CfgTemplate))

Unlike qubit_freq (which builds its own drive pulse from "設定頭" params), t1's
drive ``pi_pulse`` and ``readout`` are MODULES taken from the snapshot (lenrabi /
ro_optimize produce them, or an ml preset / default). The ``relax_delay`` and the
``sweep_range`` derive from the snapshot's smoothed ``t1``; ``reps`` / ``rounds``
come from the node's params. The flux ``dev`` entry + the ``length`` sweep are NOT
in the template — the lower-layer ``run`` merges them per point.

The acquire is always SIMULATED (no hardware): when the context is configured the
cfg-derived ``sweep_range`` sets the relax-time axis (so the cfg drives the
simulation); with the demo / empty-ml context the cfg is None and the axis is the
``sweep_range`` param directly — the existing pure-synthetic path, unchanged.
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
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseCfg,
    ReadoutCfg,
    ResetCfg,
)
from zcu_tools.utils.fitting import fit_decay

logger = logging.getLogger(__name__)

_DEFAULT_T1 = 10.0  # us — md.t1 stand-in (the smoothed-t1 fallback)
_DEFAULT_SWEEP = (0.5, 60.0, 101)  # relax-time axis (us): start, stop, npts


class T1ModuleCfg(ConfigBase):
    """t1's run modules — mirrors the lower-layer ``experiment/v2/autofluxdep``.

    ``pi_pulse`` excites the qubit, ``readout`` measures it after the relax delay;
    ``reset`` is optional (notebook omits it, so it stays None). Both ``pi_pulse``
    and ``readout`` are taken from the snapshot (lenrabi / ro_optimize produce
    them), not built from params.
    """

    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1CfgTemplate(ProgramV2Cfg, ExpCfgModel):
    """The base run cfg t1 lowers a context into.

    ``ProgramV2Cfg`` (reps/rounds/relax) + the ``ExpCfgModel`` device fields + the
    t1 modules and the ``sweep_range`` (start, stop) the relax-time axis spans —
    mirroring the lower-layer ``T1CfgTemplate``. The flux ``dev`` entry and the
    ``length`` sweep are merged in by the lower-layer ``run`` (and ``produce``
    derives the synthetic relax-time axis from ``sweep_range``).
    """

    modules: T1ModuleCfg
    sweep_range: tuple[float, float]


def _default_t1() -> float:
    return _DEFAULT_T1


def _placeholder_pi_pulse() -> Any:
    # prototype placeholder — lenrabi produces the real (placeholder) module
    return {"type": "pi", "length": 0.1}


def _default_readout() -> Any | None:
    return None


def _resolve_sweep_range(smoothed_t1: float) -> tuple[float, float]:
    """The relax-time axis span from the smoothed t1 (the notebook's formula).

    ``(0.5, max(1.0, 5 * smooth_t1))`` — the sweep runs from just above zero to a
    few times the expected T1 so the decay is well resolved. Shared by ``make_cfg``
    (the cfg's ``sweep_range``) and the cfg-driven axis in ``produce``.
    """
    return (0.5, max(1.0, 5.0 * smoothed_t1))


def _resolve_relax_delay(smoothed_t1: float) -> float:
    """The relax delay between shots from the smoothed t1 (the notebook's formula).

    ``max(1.0, 3 * smooth_t1)`` — wait a few T1 so the qubit fully decays before
    the next shot.
    """
    return max(1.0, 3.0 * smoothed_t1)


class T1Node(Node):
    """One flux point's t1: synth exp decay → fit_decay → fill row → Patch."""

    def __init__(self, env: RunEnv, builder: T1Builder) -> None:
        self._env = env
        self._builder = builder

    def _maybe_make_cfg(self, snapshot: Snapshot) -> T1CfgTemplate | None:
        """Build the run cfg when the context is configured for it, else None.

        ``make_cfg`` needs a real ``pi_pulse`` + ``readout`` module (validatable
        as ``PulseCfg`` / ``ReadoutCfg``); the default / demo context (empty ml,
        or only the prototype placeholder modules) has neither, so produce keeps
        the pure snapshot-driven simulation there. No hardware is touched either
        way — Phase B simulates the acquire uniformly; routing through
        ``make_cfg`` (when configured) exercises the real cfg pipeline and makes
        the cfg the source of the relax-time axis.

        The placeholder modules the prototype workflow flows (a ``{"type": "pi",
        ...}`` pi_pulse, a ``{"type": "readout", ...}`` opt_readout) do NOT lower
        into a valid ``T1CfgTemplate``, so the lowering itself is the
        configured/not-configured discriminator: if it raises, the context is not
        configured and produce falls back to the synthetic axis. ``make_cfg``
        called directly (a real configured context) still Fast-Fails on a missing
        module — the guard only converts that into "not configured" here.
        """
        env = self._env
        if (
            env.ml is None
            or not snapshot.has_module("pi_pulse")
            or snapshot.module("pi_pulse") is None
            or not snapshot.has_module("opt_readout")
            or snapshot.module("opt_readout") is None
        ):
            return None
        try:
            return self._builder.make_cfg(env, snapshot)
        except Exception:
            # the snapshot modules are the prototype placeholders (not real
            # PulseCfg / ReadoutCfg) — not a configured run; synthesize instead.
            logger.debug(
                "t1._maybe_make_cfg: context not lowerable (placeholder modules?)"
                " — falling back to synthetic axis",
                exc_info=True,
            )
            return None

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env
        _ = snapshot["t1"]  # smoothed (declared smooth="ewma") — dependency contract

        result: Sweep1DResult = env.result

        # Build the run cfg from the active context (when configured) and take the
        # relax-time axis from its sweep_range; the acquire is SIMULATED below.
        # With the demo / empty-ml context the cfg is None and the axis is the
        # sweep_range param directly (the existing pure-synthetic path, unchanged).
        cfg = self._maybe_make_cfg(snapshot)
        if cfg is not None:
            lo, hi = cfg.sweep_range
            # keep the pre-allocated row length (n_x), recompute only the values,
            # then write them back so the Plotter + the fit share one axis.
            times = np.linspace(float(lo), float(hi), result.n_x)
            result.x[:] = times
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
        return T1Node(env, self)

    def make_cfg(self, env: RunEnv, snapshot: Snapshot) -> T1CfgTemplate:
        """Lower the active context + this point's snapshot into the base run cfg.

        Mirrors the notebook's t1 ``cfg_maker`` (runs in ``produce``, where the
        snapshot is available): the drive ``pi_pulse`` and the ``readout`` are the
        latest-available modules (lenrabi / ro_optimize produce them, or an ml
        preset / default); ``relax_delay`` + ``sweep_range`` derive from the
        snapshot's smoothed ``t1``; ``reps`` / ``rounds`` come from the node's
        params. The flux ``dev`` entry and the ``length`` sweep are NOT here — the
        lower-layer ``run`` merges them per point.

        Raises if the ml is absent or a required module is unavailable — a real
        run needs concrete drive + readout modules (Fast Fail), unlike the
        synthetic path which fabricates a signal.
        """
        ml = env.ml
        params = env.params
        if ml is None:
            raise RuntimeError("t1.make_cfg needs an active ModuleLibrary")
        pi_pulse = snapshot.module("pi_pulse")
        if pi_pulse is None:
            raise RuntimeError("t1.make_cfg needs a pi_pulse module (none produced)")
        readout = snapshot.module("opt_readout")
        if readout is None:
            raise RuntimeError(
                "t1.make_cfg needs a readout module (none produced or preset)"
            )
        smoothed_t1 = float(snapshot["t1"])
        return ml.make_cfg(
            {
                "modules": {
                    "pi_pulse": pi_pulse,
                    "readout": readout,
                },
                "relax_delay": _resolve_relax_delay(smoothed_t1),
                "reps": int(params.get("reps", 1000)),
                "rounds": int(params.get("rounds", 10)),
                "sweep_range": _resolve_sweep_range(smoothed_t1),
            },
            T1CfgTemplate,
        )
