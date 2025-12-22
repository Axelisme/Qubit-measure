from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import NotRequired

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter1D
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

MISTPowerDepResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * len(signals)), 1)

    mist_signals = signals - np.mean(signals[:avg_len])

    return np.abs(mist_signals)


class MISTPowerDepTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class MISTPowerDepExp(AbsExperiment[MISTPowerDepResultType, MISTPowerDepTaskConfig]):
    def run(self, soc, soccfg, cfg: MISTPowerDepTaskConfig) -> MISTPowerDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted amplitudes

        Pulse.set_param(
            cfg["probe_pulse"], "gain", sweep2param("gain", cfg["sweep"]["gain"])
        )

        with LivePlotter1D("Pulse gain", "MIST") as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                                Pulse(name="init_pulse", cfg=ctx.cfg.get("init_pulse")),
                                Pulse(name="probe_pulse", cfg=ctx.cfg["probe_pulse"]),
                                Readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(pdrs),),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(pdrs, mist_signal2real(ctx.data)),
            )

        # record the last result
        self.last_cfg = cfg
        self.last_result = (pdrs, signals)

        return pdrs, signals

    def analyze(
        self,
        result: Optional[MISTPowerDepResultType] = None,
        *,
        g0=None,
        e0=None,
        ac_coeff=None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result

        if g0 is None:
            g0 = signals[0]

        amp_diff = np.abs(signals - g0)

        if ac_coeff is None:
            xs = pdrs
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * pdrs**2
            xlabel = r"$\bar n$"

        fig, ax = plt.subplots(figsize=config.figsize)
        ax.plot(xs, amp_diff.T, marker=".")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Signal difference (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if e0 is not None:
            ax.set_ylim(0, 1.1 * np.abs(g0 - e0))

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[MISTPowerDepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/mist/pdr",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> MISTPowerDepResultType:
        signals, pdrs, _ = load_data(filepath, **kwargs)
        assert pdrs is not None
        assert len(pdrs.shape) == 1 and len(signals.shape) == 1
        assert pdrs.shape == signals.shape

        pdrs = pdrs.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (pdrs, signals)

        return pdrs, signals
