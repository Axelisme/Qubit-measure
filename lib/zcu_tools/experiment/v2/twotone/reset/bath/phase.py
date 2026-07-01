from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

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
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    BathReset,
    BathResetCfg,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.fitting.base import cosfunc, fitcos
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class PhaseResult:
    phases: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: PhaseCfg | None = None


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class PhaseModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    tested_reset: BathResetCfg
    readout: ReadoutCfg


class PhaseSweepCfg(ConfigBase):
    phase: SweepCfg


class PhaseCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PhaseModuleCfg
    sweep: PhaseSweepCfg


class PhaseExp(PersistableExperiment[PhaseResult, PhaseCfg]):
    AXES_SPEC = AxesSpec(
        axes=(Axis("phases", "Phase", "deg"),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=PhaseResult,
        cfg_type=PhaseCfg,
        tag="twotone/reset/bath/phase",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: PhaseCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> PhaseResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        phases = sweep2array(
            cfg.sweep.phase,
            "phase",
            {
                "soccfg": soccfg,
                "gen_ch": modules.tested_reset.pi2_cfg.ch,
            },
        )

        with LivePlot1D("Phase (deg)", "Signal (a.u.)") as viewer:
            signals_buffer = SignalBuffer(
                (len(phases),),
                on_update=lambda data: viewer.update(
                    phases, bathreset_signal2real(data)
                ),
            )
            with Schedule(cfg, signals_buffer) as sched:
                modules = sched.cfg.modules
                modules.tested_reset.set_param(
                    "pi2_phase", sweep2param("phase", sched.cfg.sweep.phase)
                )
                _ = (
                    sched.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        Pulse("init_pulse", modules.init_pulse),
                        BathReset("tested_reset", modules.tested_reset),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("phase", sched.cfg.sweep.phase)
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )
                signals = signals_buffer.array

        return PhaseResult(phases, signals, cfg_snapshot=orig_cfg)

    @retrieve_result
    def analyze(self, result: PhaseResult | None = None) -> tuple[float, float, Figure]:
        assert result is not None, "no result found"

        phases, signals = result.phases, result.signals

        real_signals = bathreset_signal2real(signals)

        pOpt, _ = fitcos(phases, real_signals, fixedparams=[None, None, 1 / 360, None])
        y_fit = cosfunc(phases, *pOpt)

        init_phase = float(pOpt[3])

        max_phase = -init_phase
        min_phase = 180 - init_phase

        while abs(max_phase) > abs(max_phase - 360):
            max_phase -= 360
        while abs(max_phase) > abs(max_phase + 360):
            max_phase += 360
        while abs(min_phase) > abs(min_phase - 360):
            min_phase -= 360
        while abs(min_phase) > abs(min_phase + 360):
            min_phase += 360

        fig, ax = plt.subplots()
        assert isinstance(fig, Figure)

        ax.plot(phases, real_signals, ".-", label="data")
        ax.plot(phases, y_fit, "-", label="fit")
        ax.axvline(
            max_phase, color="C1", linestyle="--", label=f"max: {max_phase:.2f} deg"
        )
        ax.axvline(
            min_phase, color="C2", linestyle="--", label=f"min: {min_phase:.2f} deg"
        )
        ax.set_xlabel("Phase (deg)")
        ax.set_ylabel("Signal (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return max_phase, min_phase, fig
