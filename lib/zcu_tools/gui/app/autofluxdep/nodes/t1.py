"""t1 — exponential decay acquire and fit.

The Builder lowers the active context plus resolved modules into a T1 run cfg.
The short-lived Node applies flux, sweeps relax time, fills the Result row, and
emits trusted raw ``t1``. See ``CONTEXT.md`` for the Builder/Node boundary.

- needs the ``pi_pulse`` module (lenrabi or ModuleLibrary produces it) — without
  a concrete pi-pulse there is no excited state to relax, so the resolver skips.
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
from zcu_tools.experiment.v2_gui.adapters.twotone.time_domain.t1 import T1Adapter
from zcu_tools.gui.app.autofluxdep.cfg import (
    FloatSpec,
    OverridePath,
    OverridePlan,
    SweepValue,
    module_leaf_patches,
    str_choice_spec,
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
    round_progress,
    set_flux_by_name,
    signal2real_flip,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.defaults import (
    PULSE_MODULE_LEAF_PATHS,
    READOUT_PULSE_MODULE_LEAF_PATHS,
    adapter_node_schema,
    ctx_md_float,
    generation_field,
)
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
_DEFAULT_RELAX_FACTOR = 3.0
_DEFAULT_RELAX_MIN = 1.0
_DEFAULT_SWEEP_START = 0.5
_DEFAULT_SWEEP_STOP_FACTOR = 5.0
_DEFAULT_SWEEP_STOP_MIN = 1.0
_SWEEP_RANGE_MODE_AUTO_T1 = "auto_t1"
_SWEEP_RANGE_MODE_FIXED = "fixed"
_RELAX_DELAY_MODE_AUTO_T1 = "auto_t1"
_RELAX_DELAY_MODE_FIXED = "fixed"


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


def _default_t1() -> None:
    return None


def _seed_t1(ctx: Any | None) -> float:
    return ctx_md_float(ctx, "t1") or _DEFAULT_T1


def _snapshot_t1(snapshot: Snapshot, knobs: dict[str, Any]) -> float:
    value = snapshot.get("t1")
    if value is None:
        return float(knobs["t1_seed_us"])
    return float(value)


def _default_readout() -> Any | None:
    return None


def _resolve_sweep_range(
    smoothed_t1: float, *, start: float, stop_factor: float, stop_min: float
) -> tuple[float, float]:
    """The relax-time axis span from the smoothed t1 (the notebook's formula).

    ``(0.5, max(1.0, 5 * smooth_t1))`` — the sweep runs from just above zero to a
    few times the expected T1 so the decay is well resolved. Shared by ``make_cfg``
    (the cfg's ``sweep_range``) and the cfg-driven axis in ``produce``.
    """
    return (float(start), max(float(stop_min), float(stop_factor) * smoothed_t1))


def _resolve_relax_delay(smoothed_t1: float, *, factor: float, minimum: float) -> float:
    """The relax delay between shots from the smoothed t1 (the notebook's formula).

    ``max(1.0, 3 * smooth_t1)`` — wait a few T1 so the qubit fully decays before
    the next shot.
    """
    return max(float(minimum), float(factor) * smoothed_t1)


def _fixed_sweep_range(sweep: Any) -> tuple[float, float]:
    return (float(sweep.start), float(sweep.stop))


def _resolve_cfg_sweep_range(
    mode: str, *, smoothed_t1: float, fixed: Any, knobs: dict[str, Any]
) -> tuple[float, float]:
    if mode == _SWEEP_RANGE_MODE_AUTO_T1:
        return _resolve_sweep_range(
            smoothed_t1,
            start=float(knobs["sweep_start_us"]),
            stop_factor=float(knobs["sweep_stop_factor"]),
            stop_min=float(knobs["sweep_stop_min_us"]),
        )
    if mode == _SWEEP_RANGE_MODE_FIXED:
        return _fixed_sweep_range(fixed)
    raise RuntimeError(f"unsupported t1 sweep_range_mode: {mode!r}")


def _resolve_cfg_relax_delay(
    mode: str, *, smoothed_t1: float, fixed: float, knobs: dict[str, Any]
) -> float:
    if mode == _RELAX_DELAY_MODE_AUTO_T1:
        return _resolve_relax_delay(
            smoothed_t1,
            factor=float(knobs["relax_factor"]),
            minimum=float(knobs["relax_min_us"]),
        )
    if mode == _RELAX_DELAY_MODE_FIXED:
        return float(fixed)
    raise RuntimeError(f"unsupported t1 relax_delay_mode: {mode!r}")


def _raw_range_tuple(value: Any) -> tuple[float, float]:
    if hasattr(value, "start") and hasattr(value, "stop"):
        return (float(value.start), float(value.stop))
    lo, hi = value
    return (float(lo), float(hi))


def _pop_sweep_range(raw_cfg: dict[str, Any], key: str) -> tuple[float, float]:
    sweep = raw_cfg.pop("sweep", None)
    if not isinstance(sweep, dict) or key not in sweep:
        raise RuntimeError(f"t1 raw cfg has no sweep.{key}")
    return _raw_range_tuple(sweep[key])


class T1Node(Node):
    """One flux point's t1: set flux → real acquire → fit_decay → fill row → Patch.

    Mirrors the lower-layer T1 Schedule acquire + ``run``: a
    ``ModularProgramV2`` (Reset → pi_pulse → variable Delay → Readout) sweeps the
    relax delay (its axis spans ``5 × smoothed_t1``, from ``make_cfg``), and
    ``fit_decay`` recovers T1.
    """

    def __init__(self, env: RunEnv, builder: T1Builder) -> None:
        self._env = env
        self._builder = builder

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env

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
        stop_checkers = build_stop_checkers(env, probe, signal2real_flip)
        with round_progress(cfg.rounds, "t1", idx) as update_round_progress:
            on_round = make_on_round(
                result,
                idx,
                signal2real_flip,
                env.round_hook,
                probe=probe,
                round_progress_hook=update_round_progress,
            )
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
            result, idx, real, times, float(t1), fit_curve, env.round_hook, logger, "t1"
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
    requires_modules = (ModuleDep("pi_pulse", aliases=PI_PULSE_LIBRARY_ALIASES),)
    optional_modules = (
        ModuleDep(
            "opt_readout", default=_default_readout, aliases=READOUT_LIBRARY_ALIASES
        ),
    )

    def make_default_schema(self, ctx: Any | None = None) -> NodeCfgSchema:
        """Adapter-backed default cfg plus autofluxdep generation controls."""
        t1_seed = _seed_t1(ctx)
        return adapter_node_schema(
            T1Adapter,
            ctx,
            logical_paths={
                "reset": "modules.reset",
                "pi_pulse": "modules.pi_pulse",
                "readout": "modules.readout",
                "relax_delay": "relax_delay",
                "reps": "reps",
                "rounds": "rounds",
                "sweep_range": "sweep.length",
            },
            generation_fields=(
                generation_field(
                    "earlystop_snr",
                    "earlystop_snr",
                    FloatSpec(label="earlystop_snr", optional=True),
                    _DEFAULT_EARLYSTOP_SNR,
                    group="safety",
                ),
                generation_field(
                    "sweep_range_mode",
                    "sweep_range_mode",
                    str_choice_spec(
                        "sweep_range_mode",
                        (_SWEEP_RANGE_MODE_AUTO_T1, _SWEEP_RANGE_MODE_FIXED),
                    ),
                    _SWEEP_RANGE_MODE_AUTO_T1,
                    group="sweep",
                ),
                generation_field(
                    "relax_delay_mode",
                    "relax_delay_mode",
                    str_choice_spec(
                        "relax_delay_mode",
                        (_RELAX_DELAY_MODE_AUTO_T1, _RELAX_DELAY_MODE_FIXED),
                    ),
                    _RELAX_DELAY_MODE_AUTO_T1,
                    group="timing",
                ),
                generation_field(
                    "t1_seed_us",
                    "t1_seed_us",
                    FloatSpec(label="t1_seed_us"),
                    t1_seed,
                    group="timing",
                ),
                generation_field(
                    "relax_factor",
                    "relax_factor",
                    FloatSpec(label="relax_factor"),
                    _DEFAULT_RELAX_FACTOR,
                    group="timing",
                ),
                generation_field(
                    "relax_min_us",
                    "relax_min_us",
                    FloatSpec(label="relax_min_us"),
                    _DEFAULT_RELAX_MIN,
                    group="timing",
                ),
                generation_field(
                    "sweep_start_us",
                    "sweep_start_us",
                    FloatSpec(label="sweep_start_us"),
                    _DEFAULT_SWEEP_START,
                    group="sweep",
                ),
                generation_field(
                    "sweep_stop_factor",
                    "sweep_stop_factor",
                    FloatSpec(label="sweep_stop_factor"),
                    _DEFAULT_SWEEP_STOP_FACTOR,
                    group="sweep",
                ),
                generation_field(
                    "sweep_stop_min_us",
                    "sweep_stop_min_us",
                    FloatSpec(label="sweep_stop_min_us"),
                    _DEFAULT_SWEEP_STOP_MIN,
                    group="sweep",
                ),
            ),
            default_overrides={
                "rounds": 10,
                "relax_delay": _resolve_relax_delay(
                    t1_seed,
                    factor=_DEFAULT_RELAX_FACTOR,
                    minimum=_DEFAULT_RELAX_MIN,
                ),
                "sweep_range": SweepValue(
                    *_resolve_sweep_range(
                        t1_seed,
                        start=_DEFAULT_SWEEP_START,
                        stop_factor=_DEFAULT_SWEEP_STOP_FACTOR,
                        stop_min=_DEFAULT_SWEEP_STOP_MIN,
                    ),
                    expts=101,
                ),
            },
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

    def override_plan(self, schema: NodeCfgSchema) -> OverridePlan:
        knobs = schema.read_knobs()
        paths: list[OverridePath] = []
        paths.extend(
            OverridePath(
                f"modules.pi_pulse.{leaf}",
                "all_points",
                "pi_pulse module dependency",
                "pi pulse is resolved from workflow/module-library dependency",
            )
            for leaf in PULSE_MODULE_LEAF_PATHS
        )
        paths.extend(
            OverridePath(
                f"modules.readout.{leaf}",
                "all_points",
                "opt_readout module dependency",
                "readout module is resolved from workflow/module-library dependency",
            )
            for leaf in READOUT_PULSE_MODULE_LEAF_PATHS
        )
        if knobs.get("relax_delay_mode") == _RELAX_DELAY_MODE_AUTO_T1:
            paths.append(
                OverridePath(
                    "relax_delay",
                    "all_points",
                    "generation.timing.relax_delay_mode",
                    "relax delay is generated from T1 feedback",
                )
            )
        if knobs.get("sweep_range_mode") == _SWEEP_RANGE_MODE_AUTO_T1:
            paths.append(
                OverridePath(
                    "sweep.length",
                    "all_points",
                    "generation.sweep.sweep_range_mode",
                    "T1 sweep range is generated from T1 feedback",
                )
            )
        return OverridePlan(tuple(paths))

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
        knobs = env.knobs()
        smoothed_t1 = _snapshot_t1(snapshot, knobs)
        relax_delay = _resolve_cfg_relax_delay(
            str(knobs["relax_delay_mode"]),
            smoothed_t1=smoothed_t1,
            fixed=float(knobs["relax_delay"]),
            knobs=knobs,
        )
        sweep_range = _resolve_cfg_sweep_range(
            str(knobs["sweep_range_mode"]),
            smoothed_t1=smoothed_t1,
            fixed=knobs["sweep_range"],
            knobs=knobs,
        )
        patches: dict[str, object] = {}
        patches.update(
            module_leaf_patches(
                prefix="modules.pi_pulse",
                module=pi_pulse,
                leaf_paths=PULSE_MODULE_LEAF_PATHS,
            )
        )
        patches.update(
            module_leaf_patches(
                prefix="modules.readout",
                module=readout,
                leaf_paths=READOUT_PULSE_MODULE_LEAF_PATHS,
            )
        )
        if str(knobs["relax_delay_mode"]) == _RELAX_DELAY_MODE_AUTO_T1:
            patches["relax_delay"] = relax_delay
        if str(knobs["sweep_range_mode"]) == _SWEEP_RANGE_MODE_AUTO_T1:
            patches["sweep.length"] = sweep_range
        raw_cfg = self.point_cfg(env, patches)
        raw_cfg["sweep_range"] = _pop_sweep_range(raw_cfg, "length")
        return ml.make_cfg(raw_cfg, T1CfgTemplate)
