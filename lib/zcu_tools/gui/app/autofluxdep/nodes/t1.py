"""t1 — exponential decay acquire and fit.

The Builder lowers the active context plus resolved modules into a T1 run cfg.
The short-lived Node applies flux, sweeps relax time, fills the Result row, and
emits trusted raw ``t1`` plus ``t1err``. See ``CONTEXT.md`` for the Builder/Node
boundary.

- needs this flux point's ``pi_pulse`` module from lenrabi — without a fresh
  pi-pulse there is no excited state to relax, so the resolver skips.
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
drive ``pi_pulse`` and ``readout`` are MODULES taken from the snapshot. The
``pi_pulse`` is current-point only and never falls back to ModuleLibrary;
``readout`` may come from ro_optimize, an ml preset, or a default. The
``relax_delay`` and the ``sweep_range`` derive from the snapshot's smoothed
``t1``; ``reps`` / ``rounds`` come from the node's params. The flux ``dev`` entry
and the ``length`` sweep are NOT in the template — ``produce`` merges them per point.

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
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.gui.app.autofluxdep.cfg import OverridePlan
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgSchema
from zcu_tools.gui.app.autofluxdep.nodes.acquire import (
    DEFAULT_ACQUIRE_RETRY,
    SnrProbe,
    acquire_retry,
    acquire_to_complex,
    axis_to_sweep,
    build_stop_condition,
    fill_decay_fit_or_skip,
    make_signal_update,
    require_flux_device,
    set_flux_by_name,
    signal2real_flip,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
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
from zcu_tools.gui.app.autofluxdep.nodes.spec import (
    Dependency,
    ModuleDep,
    ModuleFallback,
    Need,
)
from zcu_tools.gui.app.autofluxdep.nodes.timing_defaults import (
    auto_relax_delay_from_t1,
    auto_stop_sweep_range,
    auto_sweep_stop,
    fixed_sweep_range,
    seed_md_float,
    snapshot_float,
)
from zcu_tools.gui.app.autofluxdep.nodes.utils import (
    NodeOverridePlan,
    NodeSchemaBuilder,
    times_to_cycles_and_axis,
)
from zcu_tools.gui.app.autofluxdep.nodes.utils.override_plan import (
    pulse_module_patches,
    readout_module_patches,
)
from zcu_tools.gui.app.autofluxdep.nodes.utils.timing import pop_sweep_range
from zcu_tools.gui.cfg import SweepValue
from zcu_tools.program.v2 import (
    Delay,
    DelayAuto,
    LoadValue,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    sweep2param,
)
from zcu_tools.utils.fitting import fit_decay

logger = logging.getLogger(__name__)


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


def t1_delay_axis(
    *, start: float, stop: float, expts: int, uniform: bool
) -> NDArray[np.float64]:
    """Return the T1 delay axis requested by the generation mode."""
    start = float(start)
    stop = float(stop)
    expts = int(expts)
    if expts <= 0:
        raise ValueError(f"t1 sweep expts must be positive, got {expts}")
    if not np.isfinite(start) or not np.isfinite(stop):
        raise ValueError("t1 sweep start/stop must be finite")
    if uniform or expts == 1:
        return np.linspace(start, stop, expts, dtype=np.float64)

    expected_t1 = 0.2 * stop
    if not np.isfinite(expected_t1) or expected_t1 <= 0.0:
        raise ValueError(
            f"t1 non-uniform sweep requires a positive finite stop value, got {stop!r}"
        )
    y0 = np.exp(-start / expected_t1)
    yN = np.exp(-stop / expected_t1)
    y_seq = np.linspace(y0, yN, expts, endpoint=True, dtype=np.float64)
    return np.asarray(-expected_t1 * np.log(y_seq), dtype=np.float64)


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
        uniform_value = env.knob("uniform", True)
        if not isinstance(uniform_value, bool):
            raise ValueError(
                f"t1 uniform must be a bool, got {type(uniform_value).__name__}"
            )
        uniform = uniform_value
        times = t1_delay_axis(
            start=float(lo), stop=float(hi), expts=result.n_x, uniform=uniform
        )
        length_cycles: list[int] | None = None
        if not uniform:
            length_cycles, times = times_to_cycles_and_axis(env.soccfg, times)
        result.x[:] = times

        flux_device = require_flux_device(env, "t1")
        set_flux_by_name(
            cast("MutableMapping[str, Any] | None", cfg.dev), flux_device, env.flux
        )
        setup_devices(cfg, progress=False)

        result.flux[idx] = env.flux

        probe = SnrProbe()
        signal_buffer = SignalBuffer(
            result.signal[idx].shape,
            dtype=np.complex128,
            on_update=make_signal_update(
                result,
                idx,
                signal2real_flip,
                env.round_hook,
                probe=probe,
            ),
            update_interval=None,
        )
        with Schedule(cfg, signal_buffer) as sched:
            builder = sched.prog_builder(env.soc, env.soccfg, cfg=cfg)

            # Sweep the relax delay over the relax-time axis. Uniform mode uses the
            # program sweep; non-uniform mode loads one delay cycle per point.
            if uniform:
                length_sweep = axis_to_sweep(times)
                length_param = sweep2param("length", length_sweep)
                builder.add(
                    [
                        Pulse("pi_pulse", cfg.modules.pi_pulse),
                        Delay("t1_delay", delay=length_param),
                        Readout("readout", cfg.modules.readout),
                    ]
                ).declare_sweep("length", length_sweep)
            else:
                if length_cycles is None:
                    raise RuntimeError(
                        "t1 non-uniform sweep did not compute delay cycles"
                    )
                builder.add(
                    [
                        LoadValue(
                            "load_t1_delay",
                            length_cycles,
                            idx_reg="length_idx",
                            val_reg="t1_delay_cycle",
                            auto_compress=False,
                        ),
                        Pulse("pi_pulse", cfg.modules.pi_pulse),
                        DelayAuto("t1_delay", t="t1_delay_cycle"),
                        Readout("readout", cfg.modules.readout),
                    ]
                ).declare_sweep("length_idx", len(length_cycles))

            signal = builder.build_and_acquire(
                raw2signal_fn=acquire_to_complex,
                retry=acquire_retry(env),
                progress=False,
                progress_label=f"{env.node_name or 't1'} flux {idx + 1} rounds",
                progress_leave=False,
                stop_condition=build_stop_condition(env, probe, signal2real_flip),
            )
            outcome = sched.outcome

        if outcome.status == "stopped":
            return Patch()
        if outcome.status == "failed":
            reason = outcome.reason or "t1 Schedule acquire failed"
            raise RuntimeError(reason) from outcome.exception
        if outcome.status == "interrupted":
            reason = outcome.reason or "t1 Schedule acquire interrupted"
            raise RuntimeError(reason) from outcome.exception
        if outcome.status != "completed":
            raise RuntimeError(f"unsupported t1 Schedule outcome: {outcome.status!r}")

        real = signal2real_flip(np.asarray(signal, dtype=np.complex128))

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
    requires_modules = (
        ModuleDep("pi_pulse", need=Need.NOW, fallback=ModuleFallback.NONE),
    )
    optional_modules = (
        ModuleDep(
            "opt_readout", default=missing_module_value, aliases=READOUT_LIBRARY_ALIASES
        ),
    )

    def make_default_schema(self, ctx: Any | None = None) -> NodeCfgSchema:
        """Default cfg plus autofluxdep generation controls."""
        t1_seed = seed_md_float(ctx, "t1", 10.0)
        relax_factor = 3.0
        relax_min_us = 1.0
        sweep_start_us = 0.5
        sweep_stop_factor = 5.0
        sweep_stop_min_us = 1.0
        max_length_default = max(sweep_stop_min_us, t1_seed * sweep_stop_factor)

        return (
            NodeSchemaBuilder(ctx, label="T1")
            .pulse(
                "pi_pulse",
                "modules.pi_pulse",
                label="Pi Pulse",
                library_keys=PI_PULSE_LIBRARY_ALIASES,
            )
            .pulse_readout(
                "readout",
                "modules.readout",
                label="Readout",
                library_keys=READOUT_LIBRARY_ALIASES,
            )
            .float(
                "relax_delay",
                "relax_delay",
                label="Relax delay (us)",
                default=auto_relax_delay_from_t1(
                    t1_seed,
                    factor=relax_factor,
                    minimum=relax_min_us,
                ),
                decimals=3,
            )
            .sweep(
                "sweep_range",
                "sweep.length",
                label="Delay (us)",
                default=SweepValue(
                    *auto_stop_sweep_range(
                        t1_seed,
                        start=sweep_start_us,
                        stop_factor=sweep_stop_factor,
                        stop_min=sweep_stop_min_us,
                        stop_max=max_length_default,
                    ),
                    expts=101,
                ),
            )
            .int("reps", "reps", label="Reps", default=1000)
            .int("rounds", "rounds", label="Rounds", default=10)
            .acquisition(
                retry=DEFAULT_ACQUIRE_RETRY,
                early_stop_snr=20.0,
            )
            .auto_relax_from_t1(
                seed_us=t1_seed,
                factor=relax_factor,
                minimum_us=relax_min_us,
            )
            .choice(
                "sweep_range_mode",
                "generation.sweep.sweep_range_mode",
                label="range_mode",
                choices=("auto_t1", "fixed"),
                default="auto_t1",
                tooltip=(
                    "Auto derives the sweep stop from latest trusted T1; "
                    "start/expts stay in Default cfg."
                ),
            )
            .float(
                "sweep_stop_factor",
                "generation.sweep.sweep_stop_factor",
                label="stop_factor",
                default=sweep_stop_factor,
                tooltip="T1 multiplier for the auto sweep stop.",
            )
            .float(
                "sweep_stop_min_us",
                "generation.sweep.sweep_stop_min_us",
                label="stop_min_us",
                default=sweep_stop_min_us,
                tooltip="Minimum stop value for the auto T1 sweep.",
            )
            .float(
                "max_length",
                "generation.sweep.max_length",
                label="max_length",
                default=max_length_default,
                tooltip="Maximum stop value for the auto T1 sweep.",
            )
            .choice_fields(
                "generation.sweep",
                "sweep_range_mode",
                {
                    "fixed": (),
                    "auto_t1": (
                        "sweep_stop_factor",
                        "sweep_stop_min_us",
                        "max_length",
                    ),
                },
                section_label="T1 sweep window",
            )
            .bool(
                "uniform",
                "generation.sweep.uniform",
                label="uniform",
                default=True,
                tooltip=(
                    "Use a linear hardware sweep; disable for a non-uniform "
                    "delay table clustered around the decay."
                ),
            )
            .build()
        )

    def make_init_result(
        self, schema: NodeCfgSchema, flux: Any, md: Any = None
    ) -> Sweep1DResult:
        knobs = schema.lower(None, md=md)
        sweep = knobs["sweep_range"]
        uniform_value = knobs.get("uniform", True)
        if not isinstance(uniform_value, bool):
            raise ValueError(
                f"t1 uniform must be a bool, got {type(uniform_value).__name__}"
            )
        uniform = uniform_value
        times = t1_delay_axis(
            start=float(sweep.start),
            stop=float(sweep.stop),
            expts=int(sweep.expts),
            uniform=uniform,
        )
        return Sweep1DResult.allocate(flux, times, x_label="relax time (us)")

    def make_plotter(self, figure: Any) -> Decay1DPlotter:
        return Decay1DPlotter(
            figure, title="t1", value_label="T1 (us)", x_label="Time (us)"
        )

    def build_node(self, env: RunEnv) -> T1Node:
        return T1Node(env, self)

    def override_plan(self, schema: NodeCfgSchema) -> OverridePlan:
        knobs = schema.read_knobs()
        plan = NodeOverridePlan()
        plan.pulse_module_dependency("pi_pulse")
        plan.readout_dependency(source="opt_readout module dependency")
        plan.generated_if(
            knobs.get("relax_delay_mode") == "auto_t1",
            "relax_delay",
            source="generation.relax.relax_delay_mode",
            reason="relax delay is generated from T1 feedback",
        )
        plan.generated_if(
            knobs.get("sweep_range_mode") == "auto_t1",
            "sweep.length.stop",
            source="generation.sweep.sweep_range_mode",
            reason="T1 sweep stop is generated from T1 feedback",
        )
        return plan.build()

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
        knobs = env.knobs_view()
        smoothed_t1 = snapshot_float(snapshot, "t1", float(knobs["t1_seed_us"]))

        relax_delay_mode = str(knobs["relax_delay_mode"])
        if relax_delay_mode == "auto_t1":
            relax_delay = auto_relax_delay_from_t1(
                smoothed_t1,
                factor=float(knobs["relax_factor"]),
                minimum=float(knobs["relax_min_us"]),
            )
        elif relax_delay_mode == "fixed":
            relax_delay = float(knobs["relax_delay"])
        else:
            raise RuntimeError(f"unsupported t1 relax_delay_mode: {relax_delay_mode!r}")

        sweep_range_mode = str(knobs["sweep_range_mode"])
        if sweep_range_mode == "auto_t1":
            fixed_sweep = knobs["sweep_range"]
            sweep_range = (
                float(fixed_sweep.start),
                auto_sweep_stop(
                    smoothed_t1,
                    stop_factor=float(knobs["sweep_stop_factor"]),
                    stop_min=float(knobs["sweep_stop_min_us"]),
                    stop_max=float(knobs["max_length"]),
                ),
            )
        elif sweep_range_mode == "fixed":
            sweep_range = fixed_sweep_range(knobs["sweep_range"])
        else:
            raise RuntimeError(f"unsupported t1 sweep_range_mode: {sweep_range_mode!r}")

        patches: dict[str, object] = {}
        patches.update(pulse_module_patches("pi_pulse", pi_pulse))
        patches.update(readout_module_patches(readout))
        if relax_delay_mode == "auto_t1":
            patches["relax_delay"] = relax_delay
        if sweep_range_mode == "auto_t1":
            patches["sweep.length.stop"] = sweep_range[1]
        raw_cfg = self.point_cfg(env, patches)
        raw_cfg["sweep_range"] = pop_sweep_range(raw_cfg, "length", node_name=self.name)
        return ml.make_cfg(raw_cfg, T1CfgTemplate)
