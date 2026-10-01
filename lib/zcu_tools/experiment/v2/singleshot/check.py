from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
)

from .util import classify_result, plot_with_classified, raw_shots_to_signal


@dataclass(frozen=True)
class CheckResult:
    shots: NDArray[np.int64]
    signals: NDArray[np.complex128]


class CheckModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class CheckCfg(ProgramV2Cfg, ExpCfgModel):
    modules: CheckModuleCfg
    shots: int


@dataclass(frozen=True)
class CheckAnalyzeOptions:
    g_center: complex
    e_center: complex
    radius: float
    max_point: int = 5000


class CheckExp(PersistableExperiment[CheckResult, CheckCfg]):
    AXES_SPEC = AxesSpec(
        axes=(Axis("shots", "shot", "point", dtype=np.int64),),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=CheckResult,
        cfg_type=CheckCfg,
        tag="singleshot/check",
    )

    def run(self, cfg: CheckCfg, *, context: RunContext) -> CheckResult:
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        # Validate and setup configuration
        if cfg.rounds != 1:
            warnings.warn(
                "rounds will be overwritten to 1 for singleshot measurement",
                stacklevel=2,
            )
            cfg.rounds = 1

        if cfg.reps != 1:
            warnings.warn(
                "reps will be overwritten by singleshot measurement shots", stacklevel=2
            )
        cfg.reps = cfg.shots

        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal.event,
        )

        signals_buffer = SignalBuffer((cfg.shots,))
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            modules = sched.cfg.modules
            program = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    Pulse("probe_pulse", modules.probe_pulse),
                    Readout("readout", modules.readout),
                )
                .build()
            )
            try:
                program.acquire(soc, progress=True, cancel_flag=sched.stop)
            except StoppedPartialAcquireError:
                sched.set_stop()
            else:
                signals_buffer.set(raw_shots_to_signal(program))
            signals = signals_buffer.array

        shots = np.arange(cfg.shots, dtype=np.int64)
        return CheckResult(shots=shots, signals=signals)

    def analyze(
        self,
        source: RunRecord[CheckCfg, CheckResult],
        options: CheckAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        signals = source.result.signals
        g_center, e_center, radius = options.g_center, options.e_center, options.radius
        _, ax = plots.subplots("fit", figsize=(6, 6))

        mask_g, mask_e, mask_o = classify_result(signals, g_center, e_center, radius)
        ng = mask_g.sum() / signals.shape[0]
        ne = mask_e.sum() / signals.shape[0]
        no = mask_o.sum() / signals.shape[0]

        plot_with_classified(
            ax, signals, g_center, e_center, radius, max_point=options.max_point
        )

        ax.set_title(
            f"Population: Ground: {ng:.1%}, Excited: {ne:.1%}, Other: {no:.1%}"
        )
