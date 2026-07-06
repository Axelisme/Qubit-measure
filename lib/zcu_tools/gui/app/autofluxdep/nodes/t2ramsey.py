"""t2ramsey — Ramsey fringe acquire and fit.

The Builder lowers resolved pi/2/readout modules plus timing knobs into the run
cfg. The short-lived Node applies flux, sweeps delay time, fits the fringe, and
emits trusted raw ``t2r`` / ``t2r_err`` / detune values.

- requires the ``pi2_pulse`` module (lenrabi or ModuleLibrary produces it); the
  resolver skips the node until a concrete pi/2 pulse is available.
- reads ``t1`` (smooth="ewma") and ``t2r`` (smooth="ewma") as optional deps:
  ``t2r`` seeds the planted t2 so the sweep tracks a plausible decoherence time;
  ``t1`` is available for cfg sanity checks (not used directly in the prototype).
- the ``opt_readout`` module is optional (ro_optimize produces it → ml preset →
  default).

``make_cfg`` lowers the active context (a populated ml + the ``pi2_pulse`` drive
module + the ``opt_readout`` readout module both present on the snapshot) + this
point's snapshot into the base ``T2RamseyCfgTemplate`` — exercising the real
``ml.make_cfg`` pipeline — and ``produce`` takes the delay-time window from the
cfg's ``sweep_range`` (which encodes ``2.5 * prev_t2r``), then acquires against a
flux-aware MockSoc (offline) or real hardware. ``make_cfg`` Fast Fails when the
context is unconfigured.

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
from collections.abc import MutableMapping
from typing import Any, cast

import numpy as np

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2_gui.adapters.twotone.time_domain.t2ramsey import (
    T2RamseyAdapter,
)
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
    build_stop_checkers,
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
    PI2_PULSE_LIBRARY_ALIASES,
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
    Readout,
    sweep2param,
)
from zcu_tools.program.v2.modules import PulseCfg, ReadoutCfg
from zcu_tools.utils.fitting import fit_decay_fringe

logger = logging.getLogger(__name__)

_DEFAULT_T1 = 10.0  # us — smoothed t1 fallback
_DEFAULT_T2R = 5.0  # us — smoothed t2r fallback
_SWEEP_T2R_FACTOR = 2.5  # notebook: sweep_range = (0, 2.5 * prev_t2r)
_DEFAULT_EARLYSTOP_SNR = 20.0
_DEFAULT_RELAX_FACTOR = 3.0
_DEFAULT_RELAX_MIN = 1.0
_DEFAULT_SWEEP_START = 0.0
_SWEEP_RANGE_MODE_AUTO_T2R = "auto_t2r"
_SWEEP_RANGE_MODE_FIXED = "fixed"
_RELAX_DELAY_MODE_AUTO_T1 = "auto_t1"
_RELAX_DELAY_MODE_FIXED = "fixed"


class T2RamseyModuleCfg(ConfigBase):
    """The module bundle a t2ramsey run cfg carries."""

    pi2_pulse: PulseCfg
    readout: ReadoutCfg


class T2RamseyCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    """The base Ramsey cfg t2ramsey lowers a context into.

    ``ProgramV2Cfg`` (reps/rounds/relax) + the ``ExpCfgModel`` device/save fields,
    plus the Ramsey ``modules`` (``pi2_pulse`` + ``readout``) and a free
    ``sweep_range`` (the delay-time span) — mirroring the lower-layer
    ``experiment/v2/autofluxdep`` ``T2RamseyCfgTemplate`` without the unused reset.
    The flux ``dev`` entry,
    the concrete ``length`` sweep, and ``activate_detune`` are merged in by the
    lower-layer ``run()`` (not here): this template is the cfg-maker output, and
    ``produce`` reads the planted-t2 baseline from ``sweep_range``.
    """

    modules: T2RamseyModuleCfg
    sweep_range: tuple[float, float]


def _seed_t1(ctx: Any | None) -> float:
    return seed_md_float(ctx, "t1", _DEFAULT_T1)


def _seed_t2r(ctx: Any | None) -> float:
    return seed_md_float(ctx, "t2r", _DEFAULT_T2R)


def _resolve_cfg_sweep_range(
    mode: str, *, t2r: float, fixed: Any, knobs: dict[str, Any]
) -> tuple[float, float]:
    if mode == _SWEEP_RANGE_MODE_AUTO_T2R:
        return auto_stop_sweep_range(
            t2r,
            start=float(knobs["sweep_start_us"]),
            stop_factor=float(knobs["sweep_stop_factor"]),
            stop_min=None,
        )
    if mode == _SWEEP_RANGE_MODE_FIXED:
        return fixed_sweep_range(fixed)
    raise RuntimeError(f"unsupported t2ramsey sweep_range_mode: {mode!r}")


def _resolve_cfg_relax_delay(
    mode: str, *, t1: float, fixed: float, knobs: dict[str, Any]
) -> float:
    if mode == _RELAX_DELAY_MODE_AUTO_T1:
        return auto_relax_delay_from_t1(
            t1,
            factor=float(knobs["relax_factor"]),
            minimum=float(knobs["relax_min_us"]),
        )
    if mode == _RELAX_DELAY_MODE_FIXED:
        return float(fixed)
    raise RuntimeError(f"unsupported t2ramsey relax_delay_mode: {mode!r}")


class T2RamseyNode(Node):
    """One flux point's t2ramsey: set flux → real acquire → fit_decay_fringe → Patch.

    Mirrors the lower-layer T2Ramsey Schedule acquire + ``run``: two
    pi/2 pulses bracket a swept delay, the second carries an activate-detune phase
    ramp (``360·detune·length``) so the fringe is resolvable, and
    ``fit_decay_fringe`` recovers T2Ramsey + the measured detune (the activate
    detune is subtracted back out, as the lower layer does).
    """

    def __init__(self, env: RunEnv, builder: T2RamseyBuilder) -> None:
        self._env = env
        self._builder = builder

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env

        result: Sweep1DResult = env.result
        times = result.x
        idx = env.flux_idx

        # Lower the active context into the run cfg (Fast Fail if unconfigured: a
        # real acquire needs a concrete pi/2 pulse + readout). sweep_range encodes
        # 2.5 × smoothed_t2r; rebuild the delay axis over it.
        cfg = self._builder.make_cfg(env, snapshot)
        lo, hi = cfg.sweep_range
        times = np.linspace(float(lo), float(hi), result.n_x)
        result.x[:] = times

        flux_device = require_flux_device(env, "t2ramsey")
        set_flux_by_name(
            cast("MutableMapping[str, Any] | None", cfg.dev), flux_device, env.flux
        )
        setup_devices(cfg, progress=False)

        # The Ramsey delay sweep + the activate-detune phase ramp on the 2nd pi/2
        # (lower layer: activate_detune = detune_ratio / len_sweep.step).
        length_sweep = axis_to_sweep(times)
        length_param = sweep2param("length", length_sweep)
        detune_ratio = float(env.knobs()["detune_ratio"])
        activate_detune = detune_ratio / length_sweep.step
        pi2_pulse = cfg.modules.pi2_pulse

        result.flux[idx] = env.flux

        probe = SnrProbe()
        stop_checkers = build_stop_checkers(env, probe, signal2real_flip)
        acquired = run_schedule_acquire(
            env=env,
            cfg=cfg,
            signal_shape=result.signal[idx].shape,
            dtype=np.complex128,
            configure_builder=lambda builder: builder.add(
                [
                    Pulse("pi2_pulse1", pi2_pulse),
                    Delay("t2r_delay", delay=length_param),
                    Pulse(
                        "pi2_pulse2",
                        pi2_pulse.with_updates(
                            phase=pi2_pulse.phase + 360 * activate_detune * length_param
                        ),
                    ),
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
            stop_checkers=stop_checkers,
        )
        if acquired.stopped:
            return Patch()
        if acquired.signal is None:
            raise RuntimeError("t2ramsey Schedule acquire completed without signal")
        real = signal2real_flip(np.asarray(acquired.signal, dtype=np.complex128))

        t2f, t2f_err, detune, _, fit_curve, _ = fit_decay_fringe(times, real)
        detune = detune - activate_detune  # back out the applied activate-detune

        if not fill_decay_fit_or_skip(
            result,
            idx,
            real,
            times,
            float(t2f),
            fit_curve,
            env.round_hook,
            logger,
            "t2ramsey",
        ):
            return Patch()  # partial: omit t2r/t2r_detune → downstream fallback

        logger.debug(
            "t2ramsey fit @flux%d: t2r=%.3f us detune=%.4f",
            idx,
            float(t2f),
            float(detune),
        )

        patch = Patch()
        patch.set("t2r", float(t2f))
        patch.set("t2r_err", float(t2f_err))
        patch.set("t2r_detune", float(detune))
        return patch


class T2RamseyBuilder(Builder):
    """The t2ramsey provider — acquire decay cosine, real fit_decay_fringe, accumulating
    colormap.  Reports the raw Ramsey t2r and the measured detuning detune.
    """

    name = "t2ramsey"
    provides = ("t2r", "t2r_err", "t2r_detune")
    optional = (
        Dependency("t1", smooth="ewma", default=missing_info_value),
        Dependency("t2r", smooth="ewma", default=missing_info_value),
    )
    requires_modules = (
        ModuleDep(
            "pi2_pulse",
            aliases=PI2_PULSE_LIBRARY_ALIASES,
        ),
    )
    optional_modules = (
        ModuleDep(
            "opt_readout", default=missing_module_value, aliases=READOUT_LIBRARY_ALIASES
        ),
    )

    def make_default_schema(self, ctx: Any | None = None) -> NodeCfgSchema:
        """Adapter-backed default cfg plus autofluxdep generation controls."""
        t1_seed = _seed_t1(ctx)
        t2r_seed = _seed_t2r(ctx)
        return adapter_node_schema(
            T2RamseyAdapter,
            ctx,
            logical_paths={
                "pi2_pulse": "modules.pi2_pulse",
                "readout": "modules.readout",
                "relax_delay": "relax_delay",
                "detune_ratio": "detune_ratio",
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
                        (_SWEEP_RANGE_MODE_AUTO_T2R, _SWEEP_RANGE_MODE_FIXED),
                    ),
                    _SWEEP_RANGE_MODE_AUTO_T2R,
                    group="sweep",
                    group_label="T2R sweep window",
                ),
                logical_generation_field(
                    "t2r_seed_us",
                    FloatSpec(label="seed_us"),
                    t2r_seed,
                    group="sweep",
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
                    _SWEEP_T2R_FACTOR,
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
                            "t1_seed_us",
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
                        _SWEEP_RANGE_MODE_AUTO_T2R: (
                            "t2r_seed_us",
                            "sweep_start_us",
                            "sweep_stop_factor",
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
                        t2r_seed,
                        start=_DEFAULT_SWEEP_START,
                        stop_factor=_SWEEP_T2R_FACTOR,
                        stop_min=None,
                    ),
                    expts=101,
                ),
            },
            drop_paths=("modules.reset",),
            module_ref_labels={"modules.readout": PULSE_READOUT_REF_LABELS},
        )

    def detune_ratio(self, schema: NodeCfgSchema, md: Any = None) -> float:
        """The activate-detune ratio for this placement (typed knob, default 0.05)."""
        return float(schema.lower(None, md=md)["detune_ratio"])

    def make_init_result(
        self, schema: NodeCfgSchema, flux: Any, md: Any = None
    ) -> Sweep1DResult:
        knobs = schema.lower(None, md=md)
        times = sweepcfg_to_axis(knobs["sweep_range"])
        return Sweep1DResult.allocate(flux, times, x_label="delay time (us)")

    def make_plotter(self, figure: Any) -> Decay1DPlotter:
        return Decay1DPlotter(
            figure, title="t2ramsey", value_label="T2Ramsey (us)", x_label="Time (us)"
        )

    def build_node(self, env: RunEnv) -> T2RamseyNode:
        return T2RamseyNode(env, self)

    def override_plan(self, schema: NodeCfgSchema) -> OverridePlan:
        knobs = schema.read_knobs()
        paths: list[OverridePath] = []
        paths.extend(
            pulse_module_override_paths(
                "pi2_pulse",
                source="pi2_pulse module dependency",
                reason="pi/2 pulse is resolved from workflow/module-library dependency",
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
        if knobs.get("sweep_range_mode") == _SWEEP_RANGE_MODE_AUTO_T2R:
            paths.append(
                OverridePath(
                    "sweep.length",
                    "all_points",
                    "generation.sweep.sweep_range_mode",
                    "T2Ramsey sweep range is generated from T2Ramsey feedback",
                )
            )
        return OverridePlan(tuple(paths))

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
        needs a concrete Ramsey sequence (Fast Fail).
        """
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
        knobs = env.knobs()
        t1 = snapshot_float(snapshot, "t1", float(knobs["t1_seed_us"]))
        t2r = snapshot_float(snapshot, "t2r", float(knobs["t2r_seed_us"]))
        relax_delay = _resolve_cfg_relax_delay(
            str(knobs["relax_delay_mode"]),
            t1=t1,
            fixed=float(knobs["relax_delay"]),
            knobs=knobs,
        )
        sweep_range = _resolve_cfg_sweep_range(
            str(knobs["sweep_range_mode"]),
            t2r=t2r,
            fixed=knobs["sweep_range"],
            knobs=knobs,
        )
        patches: dict[str, object] = {}
        patches.update(pulse_module_patches("pi2_pulse", pi2_pulse))
        patches.update(readout_module_patches(readout))
        if str(knobs["relax_delay_mode"]) == _RELAX_DELAY_MODE_AUTO_T1:
            patches["relax_delay"] = relax_delay
        if str(knobs["sweep_range_mode"]) == _SWEEP_RANGE_MODE_AUTO_T2R:
            patches["sweep.length"] = sweep_range
        raw_cfg = self.point_cfg(env, patches)
        raw_cfg.pop("detune_ratio", None)
        raw_cfg["sweep_range"] = pop_sweep_range(raw_cfg, "length", node_name=self.name)
        return ml.make_cfg(raw_cfg, T2RamseyCfgTemplate)
