from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import NotRequired

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import (
    HardTask,
    RepeatOverTime,
    ReTryIfFail,
    TaskConfig,
    run_task,
)
from zcu_tools.liveplot import LivePlotter2DwithLine
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
from zcu_tools.utils.datasaver import save_data

MISTPowerDepOvernightResultType = Tuple[
    NDArray[np.int64], NDArray[np.float64], NDArray[np.complex128]
]


def mist_overnight_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * signals.shape[1]), 1)

    mist_signals = signals - np.mean(signals[:, :avg_len], axis=1, keepdims=True)

    return np.abs(mist_signals)


class MISTPowerDepOvernightTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg
    interval: float


class MISTPowerDepOvernight(AbsExperiment):
    def run(
        self,
        soc,
        soccfg,
        cfg: MISTPowerDepOvernightTaskConfig,
        *,
        num_times=50,
        fail_retry=3,
    ) -> MISTPowerDepOvernightResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")

        iters = np.arange(num_times, dtype=np.int64)
        pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted amplitudes

        Pulse.set_param(
            cfg["probe_pulse"], "gain", sweep2param("gain", cfg["sweep"]["gain"])
        )

        with LivePlotter2DwithLine(
            "Pulse gain", "Iteration", line_axis=1, title="MIST Overnight"
        ) as viewer:
            signals = run_task(
                task=RepeatOverTime(
                    name="repeat_over_time",
                    num_times=num_times,
                    interval=cfg["interval"],
                    task=ReTryIfFail(
                        max_retries=fail_retry,
                        task=HardTask(
                            measure_fn=lambda ctx, update_hook: (
                                ModularProgramV2(
                                    soccfg,
                                    ctx.cfg,
                                    modules=[
                                        Reset(
                                            "reset",
                                            ctx.cfg.get("reset", {"type": "none"}),
                                        ),
                                        Pulse(
                                            "init_pulse", cfg=ctx.cfg.get("init_pulse")
                                        ),
                                        Pulse(
                                            "probe_pulse", cfg=ctx.cfg["probe_pulse"]
                                        ),
                                        Readout("readout", ctx.cfg["readout"]),
                                    ],
                                ).acquire(soc, progress=False, callback=update_hook)
                            ),
                            result_shape=(len(pdrs),),
                        ),
                    ),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    iters.astype(np.float64),
                    pdrs,
                    mist_overnight_signal2real(np.asarray(ctx.data)),
                ),
            )
            signals = np.asarray(signals)

        # record the last result
        self.last_cfg = cfg
        self.last_result = (iters, pdrs, signals)

        return iters, pdrs, signals

    def analyze(
        self,
        result: Optional[MISTPowerDepOvernightResultType] = None,
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
        result: Optional[MISTPowerDepOvernightResultType] = None,
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
