from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.liveplot.jupyter import LivePlotter2DwithLine
from zcu_tools.program.v2 import TwoToneProgram, sweep2param
from zcu_tools.simulate import mA2flx
from zcu_tools.utils.datasaver import save_data

from ...runner import HardTask, Runner, SoftTask


def mist_signal2real(signal: np.ndarray) -> np.ndarray:
    return np.abs(signal - signal[:, 0][:, None])


MISTFluxPowerDepResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


class MISTFluxPowerDep(AbsExperiment[MISTFluxPowerDepResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress=True
    ) -> MISTFluxPowerDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        flx_sweep = cfg["sweep"].pop("flux")

        values = sweep2array(flx_sweep, allow_array=True)
        pdrs = sweep2array(cfg["sweep"]["gain"])

        cfg["qub_pulse"]["gain"] = sweep2param("gain", cfg["sweep"]["gain"])

        with LivePlotter2DwithLine(
            "Flux device value", "Drive power (a.u.)", line_axis=1, num_lines=2
        ) as viewer:
            signals = Runner(
                task=SoftTask(
                    sweep_name="flux",
                    sweep_values=values,
                    update_cfg_fn=lambda _, ctx, flx: set_flux_in_dev_cfg(
                        ctx.cfg["dev"], flx
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            TwoToneProgram(soccfg, ctx.cfg).acquire(
                                soc, progress=False, callback=update_hook
                            )
                        ),
                        result_shape=(len(pdrs),),
                    ),
                ),
                update_hook=lambda ctx: viewer.update(
                    values, pdrs, mist_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # record the last result
        self.last_cfg = cfg
        self.last_result = (values, pdrs, signals)

        return values, pdrs, signals

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
