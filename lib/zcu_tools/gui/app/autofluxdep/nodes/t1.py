"""t1 — exponential decay acquire and fit.

The Builder lowers the active context plus resolved modules into a T1 run cfg.
The short-lived Node applies flux, sweeps relax time, fills the Result row, and
emits trusted raw ``t1`` plus ``t1err``. See ``CONTEXT.md`` for the Builder/Node
boundary.

- needs the ``pi_pulse`` module (lenrabi or ModuleLibrary produces it) — without
  a concrete pi-pulse there is no excited state to relax, so the resolver skips.
- reads ``t1`` declared ``smooth="ewma"`` (the notebook's smooth_t1) for the
  relax_delay guess + the planted decay constant; reports raw ``t1`` / ``t1err``
  back.
- the ``opt_readout`` module is optional (ro_optimize produces it → ml preset →
  default).

The smoothing is consumer-declared and same-key: this Builder reads ``t1``
smoothed and provides raw ``t1`` — the orchestrator's SmoothingService projects
the smoothed estimate under the same key for the next point's readers (lenrabi /
ro_optimize / t2*), so no separate ``smooth_t1`` key exists. ``t1err`` is
diagnostic and is not smoothed by default.

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
    str_choice_spec,
)
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgSchema, sweepcfg_to_axis
from zcu_tools.gui.app.autofluxdep.nodes.acquire import (
    SnrProbe,
    acquire_retry_generation_field,
    acquire_to_complex,
    axis_to_sweep,
    build_stop_condition,
    fill_decay_fit_or_skip,
    make_signal_update,
    require_flux_device,
    run_schedule_acquire,
    set_flux_by_name,
    signal2real_flip,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.defaults import (
    PULSE_READOUT_REF_LABELS,
    adapter_node_schema,
    generation_choice,
    logical_generation_field,
    pop_sweep_range,
    pulse_module_override_paths,
    pulse_module_patches,
    readout_module_override_paths,
    readout_module_patches,
)
from zcu_tools.gui.app.autofluxdep.nodes.dependency_defaults import (
    missing_info_value,
    missing_module_value,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.module_aliases import (
    PI_PULSE_LIBRARY_ALIASES,
    READOUT_LIBRARY_ALIASES,
)
from zcu_tools.gui.app.autofluxdep.nodes.plotters import Decay1DPlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
from zcu_tools.gui.app.autofluxdep.nodes.timing_defaults import (
    auto_relax_delay_from_t1,
    auto_stop_sweep_range,
    fixed_sweep_range,
    seed_md_float,
    snapshot_float,
)
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
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

    ``pi_pulse`` excites the qubit and ``readout`` measures it after the relax
    delay. Both modules are taken from the snapshot (lenrabi / ro_optimize
    produce them), not built from params.
    """

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


def _seed_t1(ctx: Any | None) -> float:
    return seed_md_float(ctx, "t1", _DEFAULT_T1)


def _resolve_cfg_sweep_range(
    mode: str, *, smoothed_t1: float, fixed: Any, knobs: dict[str, Any]
) -> tuple[float, float]:
    if mode == _SWEEP_RANGE_MODE_AUTO_T1:
        return auto_stop_sweep_range(
            smoothed_t1,
            start=float(knobs["sweep_start_us"]),
            stop_factor=float(knobs["sweep_stop_factor"]),
            stop_min=float(knobs["sweep_stop_min_us"]),
        )
    if mode == _SWEEP_RANGE_MODE_FIXED:
        return fixed_sweep_range(fixed)
    raise RuntimeError(f"unsupported t1 sweep_range_mode: {mode!r}")


def _resolve_cfg_relax_delay(
    mode: str, *, smoothed_t1: float, fixed: float, knobs: dict[str, Any]
) -> float:
    if mode == _RELAX_DELAY_MODE_AUTO_T1:
        return auto_relax_delay_from_t1(
            smoothed_t1,
            factor=float(knobs["relax_factor"]),
            minimum=float(knobs["relax_min_us"]),
        )
    if mode == _RELAX_DELAY_MODE_FIXED:
        return float(fixed)
    raise RuntimeError(f"unsupported t1 relax_delay_mode: {mode!r}")


class T1Node(Node):
    """One flux point's t1: set flux → real acquire → fit_decay → fill row → Patch.

    Mirrors the lower-layer T1 Schedule acquire + ``run``: a
    ``ModularProgramV2`` (pi_pulse → variable Delay → Readout) sweeps the
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
        stop_condition = build_stop_condition(env, probe, signal2real_flip)
        acquired = run_schedule_acquire(
            env=env,
            cfg=cfg,
            signal_shape=result.signal[idx].shape,
            dtype=np.complex128,
            configure_builder=lambda builder: builder.add(
                [
                    Pulse("pi_pulse", cfg.modules.pi_pulse),
                    Delay("t1_delay", delay=length_param),
                    Readout("readout", cfg.modules.readout),
                ]
            ).declare_sweep("length", length_sweep),
            raw2signal_fn=acquire_to_complex,
            on_update=make_signal_update(
                result,
                idx,
                signal2real_flip,
                env.round_hook,
                probe=probe,
            ),
            program_cls=ModularProgramV2,
            stop_condition=stop_condition,
        )
        if acquired.stopped:
            return Patch()
        if acquired.signal is None:
            raise RuntimeError("t1 Schedule acquire completed without signal")
        real = signal2real_flip(np.asarray(acquired.signal, dtype=np.complex128))

        t1, t1err, fit_curve, _ = fit_decay(times, real)

        if not fill_decay_fit_or_skip(
            result, idx, real, times, float(t1), fit_curve, env.round_hook, logger, "t1"
        ):
            return Patch()  # partial: omit t1 → downstream fallback

        logger.debug("t1 fit @flux%d: t1=%.3f us", idx, float(t1))

        patch = Patch()
        patch.set("t1", float(t1))
        patch.set("t1err", float(t1err))
        return patch


class T1Builder(Builder):
    """The t1 provider — acquire exp decay, real fit_decay, accumulating colormap."""

    name = "t1"
    provides = ("t1", "t1err")
    optional = (Dependency("t1", smooth="ewma", default=missing_info_value),)
    requires_modules = (ModuleDep("pi_pulse", aliases=PI_PULSE_LIBRARY_ALIASES),)
    optional_modules = (
        ModuleDep(
            "opt_readout", default=missing_module_value, aliases=READOUT_LIBRARY_ALIASES
        ),
    )

    def make_default_schema(self, ctx: Any | None = None) -> NodeCfgSchema:
        """Adapter-backed default cfg plus autofluxdep generation controls."""
        t1_seed = _seed_t1(ctx)
        return adapter_node_schema(
            T1Adapter,
            ctx,
            logical_paths={
                "pi_pulse": "modules.pi_pulse",
                "readout": "modules.readout",
                "relax_delay": "relax_delay",
                "reps": "reps",
                "rounds": "rounds",
                "sweep_range": "sweep.length",
            },
            generation_fields=(
                logical_generation_field(
                    "earlystop_snr",
                    FloatSpec(label="earlystop_snr", optional=True),
                    _DEFAULT_EARLYSTOP_SNR,
                    group="acquisition",
                ),
                acquire_retry_generation_field(),
                logical_generation_field(
                    "relax_delay_mode",
                    str_choice_spec(
                        "delay_mode",
                        (_RELAX_DELAY_MODE_AUTO_T1, _RELAX_DELAY_MODE_FIXED),
                    ),
                    _RELAX_DELAY_MODE_AUTO_T1,
                    group="relax",
                ),
                logical_generation_field(
                    "t1_seed_us",
                    FloatSpec(label="t1_seed_us"),
                    t1_seed,
                    group="relax",
                ),
                logical_generation_field(
                    "relax_factor",
                    FloatSpec(label="factor"),
                    _DEFAULT_RELAX_FACTOR,
                    group="relax",
                ),
                logical_generation_field(
                    "relax_min_us",
                    FloatSpec(label="min_us"),
                    _DEFAULT_RELAX_MIN,
                    group="relax",
                ),
                logical_generation_field(
                    "sweep_range_mode",
                    str_choice_spec(
                        "range_mode",
                        (_SWEEP_RANGE_MODE_AUTO_T1, _SWEEP_RANGE_MODE_FIXED),
                    ),
                    _SWEEP_RANGE_MODE_AUTO_T1,
                    group="sweep",
                    group_label="T1 sweep window",
                ),
                logical_generation_field(
                    "sweep_start_us",
                    FloatSpec(label="start_us"),
                    _DEFAULT_SWEEP_START,
                    group="sweep",
                ),
                logical_generation_field(
                    "sweep_stop_factor",
                    FloatSpec(label="stop_factor"),
                    _DEFAULT_SWEEP_STOP_FACTOR,
                    group="sweep",
                ),
                logical_generation_field(
                    "sweep_stop_min_us",
                    FloatSpec(label="stop_min_us"),
                    _DEFAULT_SWEEP_STOP_MIN,
                    group="sweep",
                ),
            ),
            generation_choices=(
                generation_choice(
                    "relax",
                    "relax_delay_mode",
                    {
                        _RELAX_DELAY_MODE_FIXED: (),
                        _RELAX_DELAY_MODE_AUTO_T1: (
                            "relax_factor",
                            "relax_min_us",
                        ),
                    },
                ),
                generation_choice(
                    "sweep",
                    "sweep_range_mode",
                    {
                        _SWEEP_RANGE_MODE_FIXED: (),
                        _SWEEP_RANGE_MODE_AUTO_T1: (
                            "sweep_start_us",
                            "sweep_stop_factor",
                            "sweep_stop_min_us",
                        ),
                    },
                ),
            ),
            default_overrides={
                "rounds": 10,
                "relax_delay": auto_relax_delay_from_t1(
                    t1_seed,
                    factor=_DEFAULT_RELAX_FACTOR,
                    minimum=_DEFAULT_RELAX_MIN,
                ),
                "sweep_range": SweepValue(
                    *auto_stop_sweep_range(
                        t1_seed,
                        start=_DEFAULT_SWEEP_START,
                        stop_factor=_DEFAULT_SWEEP_STOP_FACTOR,
                        stop_min=_DEFAULT_SWEEP_STOP_MIN,
                    ),
                    expts=101,
                ),
            },
            drop_paths=("modules.reset",),
            module_ref_labels={"modules.readout": PULSE_READOUT_REF_LABELS},
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
            pulse_module_override_paths(
                "pi_pulse",
                source="pi_pulse module dependency",
                reason="pi pulse is resolved from workflow/module-library dependency",
            )
        )
        paths.extend(
            readout_module_override_paths(
                source="opt_readout module dependency",
                reason="readout module is resolved from workflow/module-library dependency",
            )
        )
        if knobs.get("relax_delay_mode") == _RELAX_DELAY_MODE_AUTO_T1:
            paths.append(
                OverridePath(
                    "relax_delay",
                    "all_points",
                    "generation.relax.relax_delay_mode",
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
        smoothed_t1 = snapshot_float(snapshot, "t1", float(knobs["t1_seed_us"]))
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
        patches.update(pulse_module_patches("pi_pulse", pi_pulse))
        patches.update(readout_module_patches(readout))
        if str(knobs["relax_delay_mode"]) == _RELAX_DELAY_MODE_AUTO_T1:
            patches["relax_delay"] = relax_delay
        if str(knobs["sweep_range_mode"]) == _SWEEP_RANGE_MODE_AUTO_T1:
            patches["sweep.length"] = sweep_range
        raw_cfg = self.point_cfg(env, patches)
        raw_cfg["sweep_range"] = pop_sweep_range(raw_cfg, "length", node_name=self.name)
        return ml.make_cfg(raw_cfg, T1CfgTemplate)
