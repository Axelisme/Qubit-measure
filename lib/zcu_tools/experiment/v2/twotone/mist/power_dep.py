from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.program.v2 import TwoToneProgram, sweep2param
from zcu_tools.utils.process import rotate2real

from ...template import sweep_hard_template


def mist_signal2real(signal: np.ndarray) -> np.ndarray:
    return rotate2real(signal).real  # type: ignore


MISTPowerDepResultType = Tuple[np.ndarray, np.ndarray]


class MISTPowerDep(AbsExperiment[MISTPowerDepResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress=True
    ) -> MISTPowerDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        qub_pulse = cfg["qub_pulse"]

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        pdr_sweep = cfg["sweep"]["gain"]

        qub_pulse["gain"] = sweep2param("gain", pdr_sweep)

        pdrs = sweep2array(pdr_sweep)  # predicted amplitudes

        prog = TwoToneProgram(soccfg, cfg)

        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter1D("Pulse gain", "MIST", disable=not progress),
            ticks=(pdrs,),
            signal2real=mist_signal2real,
            catch_interrupt=progress,
        )

        # get the actual amplitudes
        pdrs = prog.get_pulse_param("qubit_pulse", "gain", as_array=True)  # type: ignore
        assert isinstance(pdrs, np.ndarray), "pdrs should be an array"

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

        signals = rotate2real(signals)

        if g0 is None:
            g0 = signals[0]

        amp_diff = np.abs(signals - g0)

        if ac_coeff is None:
            xs = pdrs
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * pdrs**2
            xlabel = r"$\image.pngbar n$"

        fig, ax = plt.subplots(figsize=config.figsize)
        ax.plot(xs, amp_diff.T, marker=".")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Signal difference (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if e0 is not None:
            ax.set_ylim(0, 1.1 * np.abs(g0 - e0))

    def analyze_overnight(
        self,
        pdrs: np.ndarray,
        signals: np.ndarray,
        *,
        g0=None,
        e0=None,
        ac_coeff=None,
    ) -> None:
        signals = rotate2real(signals)

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
