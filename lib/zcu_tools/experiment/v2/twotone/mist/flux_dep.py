from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot.jupyter import LivePlotter2DwithLine
from zcu_tools.program.v2 import TwoToneProgram, sweep2param
from zcu_tools.utils.process import rotate2real

from ...template import sweep2D_soft_hard_template


def mist_signal2real(signal: np.ndarray) -> np.ndarray:
    return rotate2real(signal).real  # type: ignore


MISTFluxPowerDepResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


class MISTFluxPowerDep(AbsExperiment[MISTFluxPowerDepResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress=True
    ) -> MISTFluxPowerDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        pdr_sweep = cfg["sweep"]["gain"]
        flx_sweep = cfg["sweep"]["flux"]

        del cfg["sweep"]["flux"]  # use soft loop
        cfg["qub_pulse"]["gain"] = sweep2param("gain", pdr_sweep)

        As = sweep2array(flx_sweep, allow_array=True)  # predicted currents
        pdrs = sweep2array(pdr_sweep)  # predicted gains

        cfg["dev"]["flux"] = As[0]  # set initial flux

        def updateCfg(cfg, _, mA):
            cfg["dev"]["flux"] = mA * 1e-3  # convert to A

        def signal2real(signal: ndarray) -> ndarray:
            return np.abs(signal - signal[:, 0][:, None])

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = TwoToneProgram(soccfg, cfg)
            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        signals2D = sweep2D_soft_hard_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                "Flux (mA)", "Drive power (a.u.)", line_axis=1, num_lines=2
            ),
            xs=1e3 * As,
            ys=pdrs,
            updateCfg=updateCfg,
            signal2real=signal2real,
        )

        # get the actual lengths
        prog = TwoToneProgram(soccfg, cfg)
        pdrs = prog.get_pulse_param("qubit_pulse", "gain", as_array=True)  # type: ignore
        assert isinstance(pdrs, np.ndarray), "pdrs should be an array"

        # record the last result
        self.last_cfg = cfg
        self.last_result = (As, pdrs, signals2D)

        return As, pdrs, signals2D

    def analyze(
        self,
        result: Optional[MISTFluxPowerDepResultType] = None,
        *,
        ac_coeff=None,
    ) -> None:
        if result is None:
            result = self.last_result

        flxs, pdrs, signals = result

        amp_diff = np.abs(signals - signals[:, 0][:, None])
        amp_diff = -np.clip(amp_diff, 0.01, 0.2)

        fig, ax = plt.subplots(figsize=config.figsize)

        if ac_coeff is None:
            ys = pdrs
            ylabel = "probe gain (a.u.)"
            ax.set_ylim(0.01, 1)
        else:
            ys = ac_coeff * pdrs**2
            ylabel = r"$\bar n$"
            ax.set_ylim(2, np.max(ys))

        ax.imshow(
            amp_diff.T,
            origin="lower",
            interpolation="none",
            aspect="auto",
            extent=[flxs[0], flxs[-1], ys[0], ys[-1]],
        )
        ax.set_xlabel(r"$\phi$", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_yscale("log")
        ax.tick_params(axis="both", which="major", labelsize=12)
        plt.tight_layout()

        return fig, ax
