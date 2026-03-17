from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, Optional, TypeAlias

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import OneToneCfg, OneToneProgram, Readout, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data

# (fpts, signals)
SA_FreqResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


class SA_FreqCfg(OneToneCfg, TaskCfg):
    sweep: dict[str, SweepCfg]


def safreq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class SA_FreqExp(AbsExperiment[SA_FreqResult, SA_FreqCfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> SA_FreqResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "ro_freq")
        _cfg = check_type(deepcopy(cfg), SA_FreqCfg)

        # Predicted frequency points (before mapping to ADC domain)
        fpts: NDArray[np.float64] = sweep2array(_cfg["sweep"]["ro_freq"])  # MHz

        # set readout frequency as sweep param
        Readout.set_param(
            _cfg["modules"]["readout"],
            "ro_freq",
            sweep2param("ro_freq", _cfg["sweep"]["ro_freq"]),
        )

        # run experiment
        with LivePlotter1D("SA Frequency (MHz)", "Amplitude") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: OneToneProgram(
                        soccfg, ctx.cfg
                    ).acquire(soc, progress=False, callback=update_hook),
                    result_shape=(len(fpts),),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    fpts, safreq_signal2real(ctx.root_data)
                ),
            )

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (fpts, signals)

        return fpts, signals

    def analyze(self, result: Optional[SA_FreqResult] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        fig, ax = plt.subplots(figsize=config.figsize)

        amps = safreq_signal2real(signals)

        ax.plot(fpts, amps, label="signal", marker="o", markersize=3)
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

        fpts, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> SA_FreqResult:
        signals, fpts, _ = load_data(filepath, **kwargs)
        assert len(fpts.shape) == 1 and len(signals.shape) == 1
        assert fpts.shape == signals.shape

        fpts = fpts * 1e-6  # Hz -> MHz

        fpts = fpts.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (fpts, signals)

        return fpts, signals
