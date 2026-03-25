from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, NotRequired, Optional, TypeAlias, TypedDict

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data

PowerDepOvernightResult: TypeAlias = tuple[
    NDArray[np.int64], NDArray[np.float64], NDArray[np.complex128]
]


def mist_overnight_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * signals.shape[1]), 1)

    mist_signals = signals - np.mean(signals[:, :avg_len], axis=1, keepdims=True)

    return np.abs(mist_signals)


class PowerDepOvernightModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class PowerDepOvernightCfg(ModularProgramCfg, TaskCfg):
    modules: PowerDepOvernightModuleCfg
    interval: float
    sweep: dict[str, SweepCfg]


class PowerDepOvernightExp(
    AbsExperiment[PowerDepOvernightResult, PowerDepOvernightCfg]
):
    def run(
        self, soc, soccfg, cfg: dict[str, Any], *, num_times=50, fail_retry=3
    ) -> PowerDepOvernightResult:
        _cfg = check_type(deepcopy(cfg), PowerDepOvernightCfg)
        modules = _cfg["modules"]

        _cfg["sweep"] = format_sweep1D(_cfg["sweep"], "gain")

        iters = np.arange(num_times, dtype=np.int64)
        pdrs = sweep2array(_cfg["sweep"]["gain"])  # predicted amplitudes

        Pulse.set_param(
            modules["probe_pulse"], "gain", sweep2param("gain", _cfg["sweep"]["gain"])
        )

        with LivePlotter2DwithLine(
            "Pulse gain", "Iteration", line_axis=1, title="MIST Overnight"
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: (
                        (modules := ctx.cfg["modules"])
                        and ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", modules.get("reset")),
                                Pulse("init_pulse", modules.get("init_pulse")),
                                Pulse("probe_pulse", modules["probe_pulse"]),
                                Readout("readout", modules["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(pdrs),),
                )
                .auto_retry(max_retries=fail_retry)
                .repeat("repeat_over_time", num_times, _cfg["interval"]),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    iters.astype(np.float64),
                    pdrs,
                    mist_overnight_signal2real(np.asarray(ctx.root_data)),
                ),
            )
            signals = np.asarray(signals)

        # record the last result
        self.last_cfg = _cfg
        self.last_result = (iters, pdrs, signals)

        return iters, pdrs, signals

    def analyze(
        self,
        result: Optional[PowerDepOvernightResult] = None,
        *,
        g0=None,
        e0=None,
        ac_coeff=None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        _, pdrs, signals = result

        if g0 is None:
            g0 = np.mean(signals[:, 0])

        abs_diff = np.abs(signals - g0)

        if ac_coeff is None:
            xs = pdrs
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * pdrs**2
            xlabel = r"$\bar n$"

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(xs, abs_diff.T)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Signal difference (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if e0 is not None:
            ax.set_ylim(0, 1.1 * np.abs(g0 - e0))

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[PowerDepOvernightResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/mist/pdr_overnight",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        iters, pdrs, overnight_signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
            y_info={"name": "Iteration", "unit": "None", "values": iters},
            z_info={"name": "Signal", "unit": "a.u.", "values": overnight_signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> PowerDepOvernightResult:
        overnight_signals, pdrs, iters = load_data(filepath, **kwargs)
        assert pdrs is not None and iters is not None
        assert len(pdrs.shape) == 1 and len(iters.shape) == 1
        assert overnight_signals.shape == (len(iters), len(pdrs))

        iters = iters.astype(np.int64)
        pdrs = pdrs.astype(np.float64)
        overnight_signals = overnight_signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (iters, pdrs, overnight_signals)

        return iters, pdrs, overnight_signals
