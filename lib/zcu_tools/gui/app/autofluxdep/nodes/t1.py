"""t1 — the simplest 1D Builder: acquire exp decay → fit_decay → t1.

Translates the notebook's T1Task cfg_maker. Sets this flux point's value on the
picked flux device, sets up devices, acquires an exponential decay vs relax time
with ``ModularProgramV2`` (Reset → pi_pulse → variable Delay → Readout), fits it
with the real ``fit_decay``, fills its sweep Result row in place, and returns the
raw ``t1`` Patch.

- needs the ``pi_pulse`` module (lenrabi produces it) — without a pi-pulse there
  is no excited state to relax. It carries a placeholder default, so it never
  actually skips when lenrabi is absent.
- reads ``t1`` declared ``smooth="ewma"`` (the notebook's smooth_t1) for the
  relax_delay guess + the planted decay constant; reports raw ``t1`` back.
- the ``opt_readout`` module is optional (ro_optimize produces it → ml preset →
  default).

The smoothing is consumer-declared and same-key: this Builder reads ``t1``
smoothed and provides raw ``t1`` — the orchestrator's SmoothingService projects
the smoothed estimate under the same key for the next point's readers (lenrabi /
ro_optimize / t2*), so no separate ``smooth_t1`` key exists.

``make_cfg`` lowers the active context + this point's snapshot into a runnable
``T1CfgTemplate``, mirroring the notebook's T1Task ``cfg_maker``:

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
in the template — ``produce`` merges them per point.

``produce`` sets this flux point on the picked flux device, sets up devices, and
acquires against a flux-aware MockSoc (offline) or real hardware. The cfg-derived
``sweep_range`` sets the relax-time axis (so the cfg drives the measurement
window).
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any, cast

import numpy as np

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.gui.app.autofluxdep.cfg import (
    FloatSpec,
    IntSpec,
    SweepSpec,
    SweepValue,
    flat_node_schema,
)
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgSchema, sweepcfg_to_axis
from zcu_tools.gui.app.autofluxdep.nodes.acquire import (
    SnrProbe,
    acquire_to_complex,
    axis_to_sweep,
    build_stop_checkers,
    fill_decay_fit_or_skip,
    make_on_round,
    require_flux_device,
    set_flux_by_name,
    signal2real_flip,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.module_aliases import (
    PI_PULSE_LIBRARY_ALIASES,
    READOUT_LIBRARY_ALIASES,
)
from zcu_tools.gui.app.autofluxdep.nodes.plotters import Decay1DPlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.fitting import fit_decay

logger = logging.getLogger(__name__)

_DEFAULT_T1 = 10.0  # us — md.t1 stand-in (the smoothed-t1 fallback)
_DEFAULT_EARLYSTOP_SNR = 20.0


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
    ``length`` sweep are merged in by ``produce`` (which derives the relax-time
    axis from ``sweep_range``).
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
    """One flux point's t1: set flux → real acquire → fit_decay → fill row → Patch.

    Mirrors the lower-layer ``T1Task`` ``measure_t1_fn`` + ``run``: a
    ``ModularProgramV2`` (Reset → pi_pulse → variable Delay → Readout) sweeps the
    relax delay (its axis spans ``5 × smoothed_t1``, from ``make_cfg``), and
    ``fit_decay`` recovers T1.
    """

    def __init__(self, env: RunEnv, builder: T1Builder) -> None:
        self._env = env
        self._builder = builder

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env
        _ = snapshot["t1"]  # smoothed (declared smooth="ewma") — dependency contract

        result: Sweep1DResult = env.result
        idx = env.flux_idx

        # Lower the active context into the run cfg (Fast Fail if unconfigured: a
        # real acquire needs concrete pi_pulse + readout modules). Its sweep_range
        # (= 5 × smoothed_t1) sets the relax-time axis; rebuild result.x over it so
        # the Plotter + the fit share one axis.
        cfg = self._builder.make_cfg(env, snapshot)
        lo, hi = cfg.sweep_range
        times = np.linspace(float(lo), float(hi), result.n_x)
        result.x[:] = times

        flux_device = require_flux_device(env, "t1")
        set_flux_by_name(
            cast("MutableMapping[str, Any] | None", cfg.dev), flux_device, env.flux
        )
        setup_devices(cfg, progress=False)

        # Sweep the relax delay over the relax-time axis (lower layer feeds
        # sweep2param("length") to the Delay module).
        length_sweep = axis_to_sweep(times)
        length_param = sweep2param("length", length_sweep)

        result.flux[idx] = env.flux

        probe = SnrProbe()
        on_round = make_on_round(
            result, idx, signal2real_flip, env.round_hook, probe=probe
        )
        stop_checkers = build_stop_checkers(env, probe, signal2real_flip)

        raw = ModularProgramV2(
            env.soccfg,
            cfg,
            modules=[
                Reset("reset", cfg.modules.reset),
                Pulse("pi_pulse", cfg.modules.pi_pulse),
                Delay("t1_delay", delay=length_param),
                Readout("readout", cfg.modules.readout),
            ],
            sweep=[("length", length_sweep)],
        ).acquire(
            env.soc,
            progress=False,
            round_hook=on_round,
            stop_checkers=stop_checkers,
        )
        real = signal2real_flip(acquire_to_complex(raw))

        t1, _t1err, fit_curve, _ = fit_decay(times, real)

        if not fill_decay_fit_or_skip(
            result, idx, real, float(t1), fit_curve, env.round_hook, logger, "t1"
        ):
            return Patch()  # partial: omit t1 → downstream fallback

        logger.debug("t1 fit @flux%d: t1=%.3f us", idx, float(t1))

        patch = Patch()
        patch.set("t1", float(t1))
        return patch


class T1Builder(Builder):
    """The t1 provider — acquire exp decay, real fit_decay, accumulating colormap."""

    name = "t1"
    provides = ("t1",)
    optional = (Dependency("t1", smooth="ewma", default=_default_t1),)
    requires_modules = (
        ModuleDep(
            "pi_pulse", default=_placeholder_pi_pulse, aliases=PI_PULSE_LIBRARY_ALIASES
        ),
    )
    optional_modules = (
        ModuleDep(
            "opt_readout", default=_default_readout, aliases=READOUT_LIBRARY_ALIASES
        ),
    )

    def make_default_schema(self) -> NodeCfgSchema:
        """The typed node-knob schema (defaults + types) — the param SSOT.

        ``sweep_range`` (a ``SweepSpec``, expts-defined) seeds the initial Result
        relax-time axis; the *cfg's* sweep_range is derived from the smoothed t1 in
        ``make_cfg`` (``_resolve_sweep_range``), not from this knob. Its default
        ``(0.5, 60, expts=101)`` reproduces the prototype axis; the dead
        ``num_expts`` knob (never read) is dropped.
        """
        return NodeCfgSchema(
            flat_node_schema(
                (
                    (
                        "sweep_range",
                        SweepSpec(label="Relax time sweep (us)"),
                        SweepValue(start=0.5, stop=60.0, expts=101),
                    ),
                    ("reps", IntSpec(label="Reps"), 1000),
                    ("rounds", IntSpec(label="Rounds"), 10),
                    (
                        "earlystop_snr",
                        FloatSpec(label="Early-stop SNR", optional=True),
                        _DEFAULT_EARLYSTOP_SNR,
                    ),
                )
            )
        )

    def make_init_result(
        self, schema: NodeCfgSchema, flux: Any, md: Any = None
    ) -> Sweep1DResult:
        knobs = schema.lower(None, md=md)
        times = sweepcfg_to_axis(knobs["sweep_range"])
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
        run needs concrete drive + readout modules (Fast Fail).
        """
        ml = env.ml
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
        knobs = env.schema.lower(ml, md=env.md)
        smoothed_t1 = float(snapshot["t1"])
        return ml.make_cfg(
            {
                "modules": {
                    "pi_pulse": pi_pulse,
                    "readout": readout,
                },
                "relax_delay": _resolve_relax_delay(smoothed_t1),
                "reps": knobs["reps"],
                "rounds": knobs["rounds"],
                "sweep_range": _resolve_sweep_range(smoothed_t1),
            },
            T1CfgTemplate,
        )
