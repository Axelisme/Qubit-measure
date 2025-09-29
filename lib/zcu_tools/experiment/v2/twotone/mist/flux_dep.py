from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from numpy import ndarray

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.liveplot.jupyter import LivePlotter2DwithLine
from zcu_tools.program.v2 import TwoToneProgram, sweep2param
from zcu_tools.simulate import mA2flx
from zcu_tools.utils.datasaver import save_data
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

        dev_values = sweep2array(flx_sweep, allow_array=True)  # predicted currents
        pdrs = sweep2array(pdr_sweep)  # predicted gains

        def updateCfg(cfg, _, value):
            set_flux_in_dev_cfg(cfg["dev"], value)

        updateCfg(cfg, 0, dev_values[0])  # set initial flux value

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
                "Flux device value",
                "Drive power (a.u.)",
                line_axis=1,
                num_lines=2,
            ),
            xs=dev_values,
            ticks=(pdrs,),
            updateCfg=updateCfg,
            signal2real=signal2real,
        )

        # get the actual lengths
        prog = TwoToneProgram(soccfg, cfg)
        pdrs = prog.get_pulse_param("qubit_pulse", "gain", as_array=True)  # type: ignore
        assert isinstance(pdrs, np.ndarray), "pdrs should be an array"

        # record the last result
        self.last_cfg = cfg
        self.last_result = (dev_values, pdrs, signals2D)

        return dev_values, pdrs, signals2D

    def analyze_only_mist(
        self,
        dev_values: np.ndarray,
        pdrs: np.ndarray,
        signals: np.ndarray,
        *,
        mA_c: Optional[float] = None,
        period: Optional[float] = None,
        ac_coeff: Optional[float] = None,
        **kwargs,
    ) -> plt.Figure:
        if mA_c is not None and period is not None:
            xs = mA2flx(dev_values, mA_c, period)
        else:
            xs = dev_values

        amp_diff = -np.abs(signals - signals[:, 0][:, None])

        fig, ax = plt.subplots(figsize=config.figsize)

        if ac_coeff is None:
            ys = pdrs
            ylabel = "probe gain (a.u.)"
            ax.set_ylim(0.01, 1)
        else:
            ys = ac_coeff * pdrs**2
            ylabel = r"$\bar n$"
            ax.set_ylim(2, np.max(ys))

        dx = np.ptp(pdrs) / (len(pdrs) - 1)
        dy = np.ptp(xs) / (len(xs) - 1)
        ax.imshow(
            amp_diff.T,
            origin="lower",
            interpolation="none",
            aspect="auto",
            extent=[
                xs.min() - 0.5 * dy,
                xs.max() + 0.5 * dy,
                ys.min() - 0.5 * dx,
                ys.max() + 0.5 * dx,
            ],
        )
        if mA_c is not None and period is not None:
            ax.set_xlabel(r"$\phi$", fontsize=14)
        else:
            ax.set_xlabel(r"$A$ (mA)", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_yscale("log")
        ax.tick_params(axis="both", which="major", labelsize=12)

        return fig

    def analyze_with_simulation(
        self,
        dev_values: np.ndarray,
        pdrs: np.ndarray,
        signals: np.ndarray,
        *,
        mA_c: Optional[float],
        period: Optional[float],
        ac_coeff: Optional[float],
        sim_kwargs: Dict[str, Any],
        **kwargs,
    ) -> go.Figure:
        flxs = mA2flx(dev_values, mA_c, period)

        amp_diff = np.abs(signals - signals[:, 0][:, None])
        photons = ac_coeff * pdrs**2

        from zcu_tools.notebook.analysis.branch import plot_cn_with_mist

        fig = plot_cn_with_mist(
            **sim_kwargs,
            mist_flxs=flxs,
            mist_photons=photons,
            mist_amps=amp_diff,
        )

        return fig

    def analyze(
        self,
        result: Optional[MISTFluxPowerDepResultType] = None,
        *,
        mA_c: Optional[float] = None,
        period: Optional[float] = None,
        ac_coeff: Optional[float] = None,
        with_simulation: bool = False,
        **kwargs,
    ) -> Union[plt.Figure, go.Figure]:
        if result is None:
            result = self.last_result

        values, pdrs, signals = result

        if with_simulation:
            return self.analyze_with_simulation(
                values,
                pdrs,
                signals,
                mA_c=mA_c,
                period=period,
                ac_coeff=ac_coeff,
                **kwargs,
            )
        else:
            return self.analyze_only_mist(
                values,
                pdrs,
                signals,
                mA_c=mA_c,
                period=period,
                ac_coeff=ac_coeff,
                **kwargs,
            )

    def save(
        self,
        filepath: str,
        result: Optional[MISTFluxPowerDepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/mist/flx_pdr",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, pdrs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Probe gain (a.u.)", "unit": "a.u.", "values": pdrs},
            y_info={"name": "Flux device value", "unit": "a.u.", "values": values},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
