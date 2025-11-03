from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data

from ...runner import HardTask, RepeatOverTime, Runner

MISTPowerDepResultType = Tuple[np.ndarray, np.ndarray]


def mist_signal2real(signals: np.ndarray) -> np.ndarray:
    avg_len = max(int(0.05 * len(signals)), 1)

    mist_signals = signals - np.mean(signals[:avg_len])

    return np.abs(mist_signals)


class MISTPowerDep(AbsExperiment[MISTPowerDepResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress=True
    ) -> MISTPowerDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted amplitudes

        cfg["probe_pulse"]["gain"] = sweep2param("gain", cfg["sweep"]["gain"])

        with LivePlotter1D("Pulse gain", "MIST", disable=not progress) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            cfg,
                            modules=[
                                make_reset("reset", reset_cfg=cfg.get("reset")),
                                Pulse(name="probe_pulse", cfg=cfg["probe_pulse"]),
                                make_readout("readout", readout_cfg=cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(pdrs),),
                ),
                update_hook=lambda ctx: viewer.update(
                    pdrs, mist_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

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
    ) -> None:
        if result is None:
            result = self.last_result

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
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Signal difference (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if e0 is not None:
            ax.set_ylim(0, 1.1 * np.abs(g0 - e0))

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


MISTPowerDepOvernightResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def mist_overnight_signal2real(signals: np.ndarray) -> np.ndarray:
    avg_len = max(int(0.05 * signals.shape[1]), 1)

    mist_signals = signals - np.mean(signals[:, :avg_len], axis=1, keepdims=True)

    return np.abs(mist_signals)


class MISTPowerDepOvernight(AbsExperiment[MISTPowerDepOvernightResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress=True,
        interval=300,
        num_times=50,
    ) -> MISTPowerDepOvernightResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        iters = np.arange(num_times)
        pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted amplitudes

        cfg["probe_pulse"]["gain"] = sweep2param("gain", cfg["sweep"]["gain"])

        with LivePlotter1D("Pulse gain", "MIST", disable=not progress) as viewer:
            signals = Runner(
                task=RepeatOverTime(
                    name="repeat_over_time",
                    num_times=num_times,
                    interval=interval,
                    task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            ModularProgramV2(
                                soccfg,
                                cfg,
                                modules=[
                                    make_reset("reset", reset_cfg=cfg.get("reset")),
                                    Pulse(name="probe_pulse", cfg=cfg["probe_pulse"]),
                                    make_readout("readout", readout_cfg=cfg["readout"]),
                                ],
                            ).acquire(soc, progress=False, callback=update_hook)
                        ),
                        result_shape=(len(pdrs),),
                    ),
                ),
                update_hook=lambda ctx: viewer.update(
                    pdrs, mist_overnight_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
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
    ) -> None:
        if result is None:
            result = self.last_result

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
        ax.plot(xs, abs_diff.T)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Signal difference (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if e0 is not None:
            ax.set_ylim(0, 1.1 * np.abs(g0 - e0))

        plt.tight_layout()
        plt.show()

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
