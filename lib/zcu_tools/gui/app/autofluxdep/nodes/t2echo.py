"""t2echo — Hahn-echo decay/fringe acquire and fit.

The Builder lowers resolved pi/pi2/readout modules plus timing knobs into the
run cfg. The short-lived Node applies flux, sweeps echo delay, dispatches the
configured fit, fills the Result row, and emits trusted raw ``t2e`` /
``t2e_err``.

Unlike t2ramsey, the echo sequence refocuses static dephasing and typically
yields a longer coherence time; the difference is purely in the pulse sequence.
The default ``auto_by_detune`` fit method uses a pure decay fit when
``detune_ratio == 0`` and a fringe fit otherwise.

- needs the ``pi_pulse`` and ``pi2_pulse`` modules (lenrabi produces both) — the
  Hahn echo needs both a pi refocusing pulse and two pi/2 pulses. The resolver
  skips the node until concrete drive modules are available.
- reads ``t1`` (smooth="ewma") and ``t2e`` (smooth="ewma") as optional deps:
  ``t2e`` seeds the planted t2 so the sweep tracks a plausible echo time;
  ``t1`` is available for cfg sanity checks (not used directly in the prototype).
- the ``opt_readout`` module is optional (ro_optimize produces it → ml preset →
  default).

``produce`` lowers the active context (a populated ``ml`` + the upstream
``pi_pulse`` / ``pi2_pulse`` / ``opt_readout`` modules on the snapshot, real
``PulseCfg`` / ``ReadoutCfg`` lenrabi/ro_optimize output) into a runnable
``T2EchoCfgTemplate`` via ``ml.make_cfg`` (mirroring the notebook's T2EchoTask
cfg_maker), takes the delay-time window (``sweep_range``) from the built cfg, and
acquires against a flux-aware MockSoc (offline) or real hardware. The cfg is the
source of the measurement window; ``make_cfg`` Fast Fails when the context is
unconfigured.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any, cast

import numpy as np

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.gui.app.autofluxdep.cfg import OverridePlan
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgSchema, sweepcfg_to_axis
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
    is_lowerable_pulse_module,
    missing_info_value,
    missing_module_value,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.module_aliases import (
    PI2_PULSE_LIBRARY_ALIASES,
    PI_PULSE_LIBRARY_ALIASES,
    READOUT_LIBRARY_ALIASES,
)
from zcu_tools.gui.app.autofluxdep.nodes.plotters import Decay1DPlotter
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency, ModuleDep
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
)
from zcu_tools.gui.app.autofluxdep.nodes.utils.override_plan import (
    pulse_module_patches,
    readout_module_patches,
)
from zcu_tools.gui.app.autofluxdep.nodes.utils.timing import pop_sweep_range
from zcu_tools.gui.cfg import SweepValue
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    Readout,
    sweep2param,
)
from zcu_tools.program.v2.modules import PulseCfg, ReadoutCfg
from zcu_tools.utils.fitting import fit_decay, fit_decay_fringe

logger = logging.getLogger(__name__)


class T2EchoModuleCfg(ConfigBase):
    """The module bundle a t2echo run cfg carries.

    Mirrors the lower-layer ``experiment/v2/autofluxdep`` ``T2EchoModuleCfg``
    without the unused reset: the pi refocusing pulse, the pi/2 pulse (used
    twice in the Hahn-echo sequence), and the readout. ``pi_pulse`` / ``pi2_pulse`` are the
    lenrabi-produced drive pulses; ``readout`` is the (optionally optimised)
    readout module.
    """

    pi_pulse: PulseCfg
    pi2_pulse: PulseCfg
    readout: ReadoutCfg


class T2EchoCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    """The base Hahn-echo cfg t2echo lowers a context into.

    ``ProgramV2Cfg`` (reps/rounds/relax) + the ``ExpCfgModel`` device/save fields
    + the t2echo modules and the ``sweep_range`` delay window — same bases as the
    lower-layer ``experiment/v2/autofluxdep`` ``T2EchoCfgTemplate``. The flux
    ``dev`` entry and the concrete ``length`` sweep are merged in by ``produce``;
    here ``produce`` reads the ``sweep_range`` window to parameterise the acquire.
    """

    modules: T2EchoModuleCfg
    sweep_range: tuple[float, float]


class T2EchoNode(Node):
    """One flux point's t2echo: set flux → real acquire → configured fit → Patch.

    Mirrors the lower-layer T2Echo Schedule acquire + ``run``: a
    Hahn-echo sequence (pi/2 → τ/2 → pi → τ/2 → optional detuned pi/2) sweeps the
    total delay τ, and the configured fit method recovers T2Echo.
    """

    def __init__(self, env: RunEnv, builder: T2EchoBuilder) -> None:
        self._env = env
        self._builder = builder

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env

        result: Sweep1DResult = env.result
        idx = env.flux_idx

        # Lower the active context into the run cfg (Fast Fail if unconfigured: a
        # real acquire needs concrete pi / pi2 drive pulses + a readout). The cfg's
        # sweep_range = (0, 2.5 × smoothed_t2e) sets the total-delay axis.
        cfg = self._builder.make_cfg(env, snapshot)
        lo, hi = float(cfg.sweep_range[0]), float(cfg.sweep_range[1])
        times = np.linspace(lo, hi, result.n_x)
        result.x[:] = times

        flux_device = require_flux_device(env, "t2echo")
        set_flux_by_name(
            cast("MutableMapping[str, Any] | None", cfg.dev), flux_device, env.flux
        )
        setup_devices(cfg, progress=False)

        # The total-delay sweep, split across the two echo halves (Delay 0.5·τ each),
        # + the activate-detune phase ramp on the 2nd pi/2 (lower layer:
        # activate_detune = detune_ratio / len_sweep.step).
        length_sweep = axis_to_sweep(times)
        length_param = sweep2param("length", length_sweep)
        knobs = env.knobs_view()
        detune_ratio = float(knobs["detune_ratio"])
        activate_detune = detune_ratio / length_sweep.step
        pi2_pulse = cfg.modules.pi2_pulse

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
            builder = sched.prog_builder(
                env.soc,
                env.soccfg,
                cfg=cfg,
                program_cls=ModularProgramV2,
            )
            builder.add(
                [
                    Pulse("pi2_pulse1", pi2_pulse),
                    Delay("t2e_delay1", delay=0.5 * length_param),
                    Pulse("pi_pulse", cfg.modules.pi_pulse),
                    Delay("t2e_delay2", delay=0.5 * length_param),
                    Pulse(
                        "pi2_pulse2",
                        pi2_pulse.with_updates(
                            phase=pi2_pulse.phase + 360 * activate_detune * length_param
                        ),
                    ),
                    Readout("readout", cfg.modules.readout),
                ]
            ).declare_sweep("length", length_sweep)
            signal = builder.build_and_acquire(
                raw2signal_fn=acquire_to_complex,
                retry=acquire_retry(env),
                progress=False,
                progress_label=f"{env.node_name or 't2echo'} flux {idx + 1} rounds",
                progress_leave=False,
                stop_condition=build_stop_condition(env, probe, signal2real_flip),
            )
            outcome = sched.outcome

        if outcome.status == "stopped":
            return Patch()
        if outcome.status == "failed":
            reason = outcome.reason or "t2echo Schedule acquire failed"
            raise RuntimeError(reason) from outcome.exception
        if outcome.status == "interrupted":
            reason = outcome.reason or "t2echo Schedule acquire interrupted"
            raise RuntimeError(reason) from outcome.exception
        if outcome.status != "completed":
            raise RuntimeError(
                f"unsupported t2echo Schedule outcome: {outcome.status!r}"
            )

        real = signal2real_flip(np.asarray(signal, dtype=np.complex128))

        fit_method = str(knobs["fit_method"])
        if fit_method == "auto_by_detune":
            fit_method = "decay" if detune_ratio == 0.0 else "fringe"
        if fit_method == "decay":
            t2f, t2f_err, fit_curve, _ = fit_decay(times, real)
        elif fit_method == "fringe":
            t2f, t2f_err, _, _, fit_curve, _ = fit_decay_fringe(times, real)
        else:
            raise RuntimeError(f"unsupported t2echo fit_method: {fit_method!r}")

        if not fill_decay_fit_or_skip(
            result,
            idx,
            real,
            times,
            float(t2f),
            fit_curve,
            env.round_hook,
            logger,
            "t2echo",
        ):
            return Patch()  # partial: omit t2e → downstream fallback

        logger.debug("t2echo fit @flux%d: t2e=%.3f us", idx, float(t2f))

        patch = Patch()
        patch.set("t2e", float(t2f))
        patch.set("t2e_err", float(t2f_err))
        return patch


class T2EchoBuilder(Builder):
    """The t2echo provider — acquire echo decay/fringe traces and fit T2Echo.

    Reports only the raw echo t2e (detune is refocused and not reported).
    """

    name = "t2echo"
    provides = ("t2e", "t2e_err")
    optional = (
        Dependency("t1", smooth="ewma", default=missing_info_value),
        Dependency("t2e", smooth="ewma", default=missing_info_value),
    )
    requires_modules = (
        ModuleDep("pi_pulse", aliases=PI_PULSE_LIBRARY_ALIASES),
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
        """Default cfg plus autofluxdep generation controls."""
        t1_seed = seed_md_float(ctx, "t1", 10.0)
        t2e_seed = seed_md_float(ctx, "t2e", 5.0)
        sweep_stop_factor = 2.5  # notebook: sweep_range = (0, 2.5 * prev_t2e)
        relax_factor = 3.0
        relax_min_us = 1.0
        max_length_default = t2e_seed * sweep_stop_factor

        return (
            NodeSchemaBuilder(ctx, label="T2 Echo")
            .pulse(
                "pi_pulse",
                "modules.pi_pulse",
                label="Pi Pulse",
                library_keys=PI_PULSE_LIBRARY_ALIASES,
            )
            .pulse(
                "pi2_pulse",
                "modules.pi2_pulse",
                label="Pi/2 Pulse",
                library_keys=PI2_PULSE_LIBRARY_ALIASES,
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
            .float(
                "detune_ratio",
                "detune_ratio",
                label="Detune ratio (fringes/step)",
                default=0.1,
                decimals=3,
            )
            .sweep(
                "sweep_range",
                "sweep.length",
                label="Total delay (us)",
                default=SweepValue(
                    *auto_stop_sweep_range(
                        t2e_seed,
                        start=0.0,
                        stop_factor=sweep_stop_factor,
                        stop_min=None,
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
                choices=("auto_t2e", "fixed"),
                default="auto_t2e",
                tooltip=(
                    "Auto derives the echo sweep stop from latest trusted "
                    "T2E; start/expts stay in Default cfg."
                ),
            )
            .float(
                "t2e_seed_us",
                "generation.sweep.t2e_seed_us",
                label="initial_t2e_us",
                default=t2e_seed,
                tooltip="Initial T2E before measured feedback exists.",
            )
            .float(
                "sweep_stop_factor",
                "generation.sweep.sweep_stop_factor",
                label="stop_factor",
                default=sweep_stop_factor,
                tooltip="T2E multiplier for the auto sweep stop.",
            )
            .float(
                "max_length",
                "generation.sweep.max_length",
                label="max_length",
                default=max_length_default,
                tooltip="Maximum stop value for the auto echo sweep.",
            )
            .choice_fields(
                "generation.sweep",
                "sweep_range_mode",
                {
                    "fixed": (),
                    "auto_t2e": (
                        "t2e_seed_us",
                        "sweep_stop_factor",
                        "max_length",
                    ),
                },
            )
            .choice(
                "fit_method",
                "generation.fit.fit_method",
                label="method",
                choices=("auto_by_detune", "fringe", "decay"),
                default="auto_by_detune",
                tooltip="Choose echo fit model; auto follows detune ratio.",
            )
            .build()
        )

    def make_init_result(
        self, schema: NodeCfgSchema, flux: Any, md: Any = None
    ) -> Sweep1DResult:
        knobs = schema.lower(None, md=md)
        times = sweepcfg_to_axis(knobs["sweep_range"])
        return Sweep1DResult.allocate(flux, times, x_label="delay time (us)")

    def make_plotter(self, figure: Any) -> Decay1DPlotter:
        return Decay1DPlotter(
            figure, title="t2echo", value_label="T2 Echo (us)", x_label="Time (us)"
        )

    def build_node(self, env: RunEnv) -> T2EchoNode:
        return T2EchoNode(env, self)

    def override_plan(self, schema: NodeCfgSchema) -> OverridePlan:
        knobs = schema.read_knobs()
        plan = NodeOverridePlan()
        plan.pulse_module_dependency("pi_pulse")
        plan.pulse_module_dependency(
            "pi2_pulse",
            reason="pi/2 pulse is resolved from workflow/module-library dependency",
        )
        plan.readout_dependency(source="opt_readout module dependency")
        plan.generated_if(
            knobs.get("relax_delay_mode") == "auto_t1",
            "relax_delay",
            source="generation.relax.relax_delay_mode",
            reason="relax delay is generated from T1 feedback",
        )
        plan.generated_if(
            knobs.get("sweep_range_mode") == "auto_t2e",
            "sweep.length.stop",
            source="generation.sweep.sweep_range_mode",
            reason="T2Echo sweep stop is generated from T2Echo feedback",
        )
        return plan.build()

    def make_cfg(self, env: RunEnv, snapshot: Snapshot) -> T2EchoCfgTemplate:
        """Lower the active context + this point's snapshot into the base run cfg.

        Mirrors the notebook's t2echo ``cfg_maker``: the pi / pi2 drive pulses are
        the latest-available lenrabi-produced ``pi_pulse`` / ``pi2_pulse`` modules
        on the snapshot, the readout is the latest-available ``opt_readout``
        module, the relax delay is ``max(1.0, 3 * smoothed_t1)``, and the
        ``sweep_range`` delay window is ``(0, 2.5 * smoothed_t2e)``. The flux
        ``dev`` entry and the concrete ``length`` sweep are NOT here — the
        lower-layer ``run`` merges them.

        Raises if the ml / drive pulses / readout are unavailable — a real run
        needs concrete drive pulses (Fast Fail).
        """
        ml = env.ml
        if ml is None:
            raise RuntimeError("t2echo.make_cfg needs an active ModuleLibrary")
        pi_pulse = snapshot.module("pi_pulse")
        pi2_pulse = snapshot.module("pi2_pulse")
        readout = snapshot.module("opt_readout")
        if not is_lowerable_pulse_module(pi_pulse) or not is_lowerable_pulse_module(
            pi2_pulse
        ):
            raise RuntimeError(
                "t2echo.make_cfg needs concrete pi_pulse / pi2_pulse drive modules "
                "(lenrabi output)"
            )
        if readout is None:
            raise RuntimeError(
                "t2echo.make_cfg needs a readout module (none produced or preset)"
            )
        knobs = env.knobs_view()
        cur_t1 = snapshot_float(snapshot, "t1", float(knobs["t1_seed_us"]))
        prev_t2e = snapshot_float(snapshot, "t2e", float(knobs["t2e_seed_us"]))

        relax_delay_mode = str(knobs["relax_delay_mode"])
        if relax_delay_mode == "auto_t1":
            relax_delay = auto_relax_delay_from_t1(
                cur_t1,
                factor=float(knobs["relax_factor"]),
                minimum=float(knobs["relax_min_us"]),
            )
        elif relax_delay_mode == "fixed":
            relax_delay = float(knobs["relax_delay"])
        else:
            raise RuntimeError(
                f"unsupported t2echo relax_delay_mode: {relax_delay_mode!r}"
            )

        sweep_range_mode = str(knobs["sweep_range_mode"])
        if sweep_range_mode == "auto_t2e":
            fixed_sweep = knobs["sweep_range"]
            sweep_range = (
                float(fixed_sweep.start),
                auto_sweep_stop(
                    prev_t2e,
                    stop_factor=float(knobs["sweep_stop_factor"]),
                    stop_min=None,
                    stop_max=float(knobs["max_length"]),
                ),
            )
        elif sweep_range_mode == "fixed":
            sweep_range = fixed_sweep_range(knobs["sweep_range"])
        else:
            raise RuntimeError(
                f"unsupported t2echo sweep_range_mode: {sweep_range_mode!r}"
            )

        patches: dict[str, object] = {}
        patches.update(pulse_module_patches("pi_pulse", pi_pulse))
        patches.update(pulse_module_patches("pi2_pulse", pi2_pulse))
        patches.update(readout_module_patches(readout))
        if relax_delay_mode == "auto_t1":
            patches["relax_delay"] = relax_delay
        if sweep_range_mode == "auto_t2e":
            patches["sweep.length.stop"] = sweep_range[1]
        raw_cfg = self.point_cfg(env, patches)
        raw_cfg.pop("detune_ratio", None)
        raw_cfg["sweep_range"] = pop_sweep_range(raw_cfg, "length", node_name=self.name)
        return ml.make_cfg(raw_cfg, T2EchoCfgTemplate)
