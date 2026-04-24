from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel
from typing_extensions import Any, Callable, Optional, TypeAlias

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data

# (freqs, signals)
SA_FreqResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


class SA_FreqModuleCfg(BaseModel):
    reset: Optional[ResetCfg] = None
    readout: PulseReadoutCfg


class SA_FreqSweepCfg(BaseModel):
    freq: SweepCfg


class SA_FreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: SA_FreqModuleCfg
    sweep: SA_FreqSweepCfg


def safreq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class SA_FreqExp(AbsExperiment[SA_FreqResult, SA_FreqCfg]):
    def run(self, soc, soccfg, cfg: SA_FreqCfg) -> SA_FreqResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        # Predicted frequency points (before mapping to ADC domain)
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {
                "soccfg": soccfg,
                "gen_ch": modules.readout.pulse_cfg.ch,
                "ro_ch": modules.readout.ro_cfg.ro_ch,
            },
        )

        def measure_fn(
            ctx: TaskState[Any, Any, SA_FreqCfg],
            update_hook: Optional[Callable],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            freq_sweep = cfg.sweep.freq
            freq_param = sweep2param("ro_freq", freq_sweep)
            modules.readout.set_param("ro_freq", freq_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    PulseReadout("readout", modules.readout),
                ],
                sweep=[("ro_freq", freq_sweep)],
            ).acquire(soc, progress=False, round_hook=update_hook)

        # run experiment
        with LivePlot1D("SA Frequency (MHz)", "Amplitude") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(freqs),),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    freqs, safreq_signal2real(ctx.root_data)
                ),
            )

        # record last cfg and result
        self.last_cfg = deepcopy(cfg)
        self.last_result = (freqs, signals)

        return freqs, signals

    def analyze(self, result: Optional[SA_FreqResult] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs, signals = result

        fig, ax = plt.subplots(figsize=config.figsize)

        amps = safreq_signal2real(signals)

        ax.plot(freqs, amps, label="signal", marker="o", markersize=3)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[SA_FreqResult] = None,
        comment: Optional[str] = None,
        tag: str = "onetone/sa_freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs, signals = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> SA_FreqResult:
        signals, freqs, _, comment = load_data(filepath, return_comment=True, **kwargs)
        assert len(freqs.shape) == 1 and len(signals.shape) == 1
        assert freqs.shape == signals.shape

        freqs = freqs * 1e-6  # Hz -> MHz

        freqs = freqs.astype(np.float64)
        signals = signals.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                self.last_cfg = SA_FreqCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (freqs, signals)

        return freqs, signals
