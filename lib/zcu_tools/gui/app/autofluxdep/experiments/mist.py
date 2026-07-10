"""mist — disturbance-gain sweep with direct variance readout.

The experiment requires a ``pi_pulse`` and resolves an ``opt_readout`` module.
Its Builder owns the typed module cfg; its Node sweeps ``mist_pulse.gain``, maps
IQ scatter through the experiment-specific magnitude transform, fills the raw
result row, and emits ``success`` without a fit. Fit fields therefore stay NaN.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.gui.app.autofluxdep.cfg import OverridePlan
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgSchema, sweepcfg_to_axis
from zcu_tools.gui.app.autofluxdep.experiments._support.acquire import (
    DEFAULT_ACQUIRE_RETRY,
    acquire_retry,
    acquire_to_complex,
    axis_to_sweep,
    schedule_completed,
    setup_flux_point,
)
from zcu_tools.gui.app.autofluxdep.experiments._support.dependency_defaults import (
    missing_module_value,
)
from zcu_tools.gui.app.autofluxdep.experiments._support.module_aliases import (
    PI_PULSE_LIBRARY_ALIASES,
    READOUT_LIBRARY_ALIASES,
)
from zcu_tools.gui.app.autofluxdep.experiments._support.plotters import (
    ColormapLinePlotter,
)
from zcu_tools.gui.app.autofluxdep.experiments._support.readout_defaults import (
    seed_readout_freq,
)
from zcu_tools.gui.app.autofluxdep.experiments._support.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.experiments._support.utils import (
    NodeOverridePlan,
    NodeSchemaBuilder,
)
from zcu_tools.gui.app.autofluxdep.experiments._support.utils.override_plan import (
    pulse_module_patches,
    readout_module_patches,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.spec import ModuleDep
from zcu_tools.gui.cfg import EvalValue, SweepValue
from zcu_tools.gui.session.types import ExpContext
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    sweep2param,
)

logger = logging.getLogger(__name__)


def _mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """The MIST state-disturbance magnitude (lower-layer ``mist_signal2real``).

    The disturbance is read directly from the IQ scatter — no fit: the centred
    magnitude normalised by its own spread, so a flat (undisturbed) trace reads
    near zero and the onset shows as the magnitude rising past a gain threshold."""
    if np.all(np.isnan(signals)):
        return np.abs(signals)
    mag = np.abs(signals - np.mean(signals))
    std = float(np.std(mag))
    return mag / (std + 1e-12)


class MistModuleCfg(ConfigBase):
    """Typed preparation, disturbance, and readout modules for one MIST run."""

    pi_pulse: PulseCfg
    mist_pulse: PulseCfg
    readout: ReadoutCfg


class MistCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    """The base cfg mist lowers a context into (no sweep / dev yet).

    Mirrors the lower-layer ``experiment/v2/autofluxdep`` ``MistCfgTemplate``
    (``ProgramV2Cfg`` reps/rounds/relax + ``ExpCfgModel`` device fields) with the
    ``pi_pulse`` / ``mist_pulse`` / ``readout`` modules. The flux ``dev`` entry and
    the ``gain`` sweep (which sweeps ``mist_pulse.gain``) are merged in by the run
    layer downstream, not built here — exactly as the lower layer's ``run`` does.
    """

    modules: MistModuleCfg


class MistNode(Node):
    """Acquire one flux point's transformed disturbance curve without fitting."""

    def __init__(self, env: RunEnv, builder: MistBuilder) -> None:
        self._env = env
        self._builder = builder

    def produce(self, snapshot: Snapshot) -> Patch:
        env = self._env

        result: Sweep1DResult = env.result
        gains = result.x
        idx = env.flux_idx

        # Lower the active context into the run cfg (Fast Fail if unconfigured: a
        # real acquire needs concrete pi_pulse + mist_pulse + readout). MIST is a
        # known SimEngine gap — the .acquire below raises (not swallowed) under the
        # mock if the engine cannot model the disturbance program, surfacing the
        # gap rather than degrading to noise.
        cfg = self._builder.make_cfg(env, snapshot)

        setup_flux_point(cfg, env, "mist")

        # Sweep the disturbance pulse gain over the Result's gain axis (lower layer
        # sets sweep2param("gain") on mist_pulse).
        gain_sweep = axis_to_sweep(gains)
        cfg.modules.mist_pulse.set_param("gain", sweep2param("gain", gain_sweep))

        result.flux[idx] = env.flux
        # fit_value[idx] and fit_curve[idx] remain nan — mist has no fit scalar.
        # probe=None: a single-round scatter, so there is no SNR early-stop to feed.

        def on_update(curve_value: NDArray[Any]) -> None:
            np.copyto(result.signal[idx], np.asarray(curve_value, dtype=np.float64))
            if env.round_hook is not None:
                env.round_hook(idx)

        signal_buffer = SignalBuffer(
            result.signal[idx].shape,
            dtype=np.float64,
            on_update=on_update,
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
                    Pulse("pi_pulse", cfg.modules.pi_pulse),
                    Pulse("mist_pulse", cfg.modules.mist_pulse),
                    Readout("readout", cfg.modules.readout),
                ]
            ).declare_sweep("gain", gain_sweep)
            signal = builder.build_and_acquire(
                raw2signal_fn=lambda raw: _mist_signal2real(acquire_to_complex(raw)),
                retry=acquire_retry(env),
                progress=False,
                progress_label=f"{env.node_name or 'mist'} flux {idx + 1} rounds",
                progress_leave=False,
            )
            outcome = sched.outcome

        if not schedule_completed(outcome, "mist"):
            return Patch()

        curve = np.asarray(signal, dtype=np.float64)

        logger.debug(
            "mist @flux%d: success, variance range [%.3f, %.3f]",
            idx,
            float(curve.min()),
            float(curve.max()),
        )

        patch = Patch()
        patch.set("success", 1.0)  # float: consistent with info-value domain
        return patch


class MistBuilder(Builder):
    """The MIST provider — acquire variance curve, no fit, accumulating colormap.

    Sweeps a gain axis per flux point, acquires a state-disturbance curve, and
    records the variance directly (no fit). ``fit_value`` stays nan so the
    ``ColormapLinePlotter`` renders only the flux×gain colormap. Provides ``success``
    (float 1.0) to signal that the MIST pass completed; the ``opt_readout``
    module is consumed (from ro_optimize) to configure the readout during the real
    measurement.
    """

    name = "mist"
    provides = ("success",)
    provides_modules: tuple[str, ...] = ()
    requires_modules = (ModuleDep("pi_pulse", aliases=PI_PULSE_LIBRARY_ALIASES),)
    optional_modules = (
        ModuleDep(
            "opt_readout",
            default=missing_module_value,
            aliases=READOUT_LIBRARY_ALIASES,
        ),
    )

    def make_default_schema(self, ctx: Any | None = None) -> NodeCfgSchema:
        """Default cfg for the MIST power sweep."""
        relax_delay_default: float | EvalValue = 30.5
        if isinstance(ctx, ExpContext) and ctx.md.get("t1") is not None:
            relax_delay_default = EvalValue("5.0 * t1")

        return (
            NodeSchemaBuilder(ctx, label="MIST")
            .pulse(
                "pi_pulse",
                "modules.pi_pulse",
                label="Pi Pulse",
                library_keys=PI_PULSE_LIBRARY_ALIASES,
            )
            .pulse(
                "mist_pulse",
                "modules.mist_pulse",
                label="MIST Pulse",
                library_keys=("res_probe",),
                blank_overrides={
                    "ch": 0,
                    "nqz": 1,
                    "gain": 0.0,
                    "waveform.length": 1.0,
                },
                overrides={"freq": seed_readout_freq(ctx, fallback=6000.0)},
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
                default=relax_delay_default,
                decimals=3,
            )
            .sweep(
                "gain_sweep",
                "sweep.gain",
                label="Probe gain (a.u.)",
                default=SweepValue(start=0.0, stop=1.0, expts=151),
            )
            .int("reps", "reps", label="Reps", default=1000)
            .int("rounds", "rounds", label="Rounds", default=10)
            .knob("mist_ch", "modules.mist_pulse.ch")
            .knob("mist_nqz", "modules.mist_pulse.nqz")
            .knob("mist_freq", "modules.mist_pulse.freq")
            .knob("mist_gain", "modules.mist_pulse.gain")
            .knob("mist_length", "modules.mist_pulse.waveform.length")
            .acquisition(retry=DEFAULT_ACQUIRE_RETRY)
            .build()
        )

    def make_init_result(
        self, schema: NodeCfgSchema, flux: Any, md: Any = None
    ) -> Sweep1DResult:
        knobs = schema.lower(None, md=md)
        gains = sweepcfg_to_axis(knobs["gain_sweep"])
        return Sweep1DResult.allocate(flux, gains, x_label="gain")

    def make_plotter(self, figure: Any) -> ColormapLinePlotter:
        return ColormapLinePlotter(
            figure, title="mist", y_label="Readout Gain (a.u.)", num_lines=1
        )

    def build_node(self, env: RunEnv) -> MistNode:
        return MistNode(env, self)

    def override_plan(self, schema: NodeCfgSchema) -> OverridePlan:
        plan = NodeOverridePlan()
        plan.pulse_module_dependency("pi_pulse")
        plan.readout_dependency(source="opt_readout module dependency")
        return plan.build()

    def make_cfg(self, env: RunEnv, snapshot: Snapshot) -> MistCfgTemplate:
        """Lower the active context + this point's snapshot into the base run cfg.

        Mirrors the lower-layer ``experiment/v2/autofluxdep`` mist ``cfg_maker``
        (the notebook keeps mist commented out; the lower-layer ``MistTask`` is the
        ground truth): the ``pi_pulse`` is the latest-available pi-pulse module, the
        ``readout`` is the latest-available optimised readout, and the disturbance
        ``mist_pulse`` waveform / channel / gain / nqz / freq come from the node's
        params (the "設定頭"). The flux ``dev`` entry and the ``gain`` sweep (which
        sweeps ``mist_pulse.gain``) are NOT here — the run layer merges them, exactly
        as the lower layer's ``run`` does.

        Raises if the ``pi_pulse`` module is unavailable or the mist-drive params
        are unset — a real run needs a concrete disturbance pulse + an excited-state
        preparation pulse (Fast Fail).
        """
        ml = env.ml
        if ml is None:
            raise RuntimeError("mist.make_cfg needs an active ModuleLibrary")
        pi_pulse = snapshot.module("pi_pulse")
        if pi_pulse is None:
            raise RuntimeError(
                "mist.make_cfg needs a pi_pulse module (none produced or preset)"
            )
        # readout is optional in the snapshot (ro_optimize → ml preset → default);
        # when absent the lower-layer cfg's required `readout` field cannot be filled,
        # so a real run needs one — but _maybe_make_cfg only routes here once a
        # readout-bearing context exists. Fall back to the ml preset path is the
        # caller's responsibility; here we use whatever the snapshot resolved.
        readout = snapshot.module("opt_readout")
        if readout is None:
            raise RuntimeError(
                "mist.make_cfg needs a readout module (none produced or preset)"
            )
        patches: dict[str, object] = {}
        patches.update(pulse_module_patches("pi_pulse", pi_pulse))
        patches.update(readout_module_patches(readout))
        raw_cfg = self.point_cfg(env, patches)
        raw_cfg.pop("sweep", None)
        return ml.make_cfg(raw_cfg, MistCfgTemplate)


EXPERIMENT = MistBuilder()
