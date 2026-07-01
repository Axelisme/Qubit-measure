from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
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
    cfg_snapshot: CheckCfg | None = None


class CheckModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class CheckCfg(ProgramV2Cfg, ExpCfgModel):
    modules: CheckModuleCfg
    shots: int


class CheckExp(PersistableExperiment[CheckResult, CheckCfg]):
    AXES_SPEC = AxesSpec(
        axes=(Axis("shots", "shot", "point", dtype=np.int64),),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=CheckResult,
        cfg_type=CheckCfg,
        tag="singleshot/check",
    )

    @record_result
    def run(self, soc, soccfg, cfg: CheckCfg) -> CheckResult:
        cfg = deepcopy(cfg)
        # Validate and setup configuration
        if cfg.rounds != 1:
            warnings.warn("rounds will be overwritten to 1 for singleshot measurement")
            cfg.rounds = 1

        if cfg.reps != 1:
            warnings.warn("reps will be overwritten by singleshot measurement shots")
        cfg.reps = cfg.shots

        setup_devices(cfg, progress=True)

        signals_buffer = SignalBuffer((cfg.shots,))
        with Schedule(cfg, signals_buffer) as sched:
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
            program.acquire(soc, progress=True, stop_checkers=[sched.is_stop])
            signals_buffer.set(raw_shots_to_signal(program))
            signals = signals_buffer.array

        # Cache results
        shots = np.arange(cfg.shots, dtype=np.int64)
        self.last_result = CheckResult(shots=shots, signals=signals, cfg_snapshot=cfg)

        return self.last_result

    @retrieve_result
    def analyze(
        self,
        g_center: complex,
        e_center: complex,
        radius: float,
        result: CheckResult | None = None,
        max_point: int = 5000,
    ) -> Figure:
        assert result is not None, "no result found"

        signals = result.signals

        fig, ax = plt.subplots(figsize=(6, 6))

        mask_g, mask_e, mask_o = classify_result(signals, g_center, e_center, radius)
        ng = mask_g.sum() / signals.shape[0]
        ne = mask_e.sum() / signals.shape[0]
        no = mask_o.sum() / signals.shape[0]

        plot_with_classified(
            ax, signals, g_center, e_center, radius, max_point=max_point
        )

        ax.set_title(
            f"Population: Ground: {ng:.1%}, Excited: {ne:.1%}, Other: {no:.1%}"
        )

        return fig
