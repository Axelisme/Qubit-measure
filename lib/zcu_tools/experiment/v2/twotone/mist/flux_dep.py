from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, Runner, SoftTask, TaskContext
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program.v2 import ModularProgramV2, Pulse, Readout, Reset, sweep2param
from zcu_tools.simulate import mA2flx
from zcu_tools.utils.datasaver import save_data

MistFluxDepResultType = Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]


def mist_signal2real(signals: np.ndarray) -> np.ndarray:
    avg_len = max(int(0.05 * signals.shape[1]), 1)
    std_len = max(int(0.3 * signals.shape[1]), 5)

    mist_signals = np.abs(
        signals - np.mean(signals[:, :avg_len], axis=1, keepdims=True)
    )
    if np.all(np.isnan(mist_signals)):
        return mist_signals
    mist_signals = np.clip(mist_signals, 0, 5 * np.nanstd(mist_signals[:, :std_len]))

    return mist_signals


class MistFluxDepExperiment(AbsExperiment[MistFluxDepResultType]):
    def run(self, soc, soccfg, cfg: Dict[str, Any]) -> MistFluxDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        flx_sweep = cfg["sweep"]["flux"]
        cfg["sweep"] = {"gain": cfg["sweep"]["gain"]}

        # predict sweep points
        values = sweep2array(flx_sweep)
        gains = sweep2array(cfg["sweep"]["gain"])

        def updateCfg(i: int, ctx: TaskContext, value: float) -> None:
            set_flux_in_dev_cfg(ctx.cfg["dev"], value)

        Pulse.set_param(
            cfg["probe_pulse"], "gain", sweep2param("gain", cfg["sweep"]["gain"])
        )

        with LivePlotter2DwithLine(
            "Flux device value",
            "Readout power (a.u.)",
            line_axis=1,
            num_lines=5,
            title="MIST over FLux",
        ) as viewer:
            results = Runner(
                task=SoftTask(
                    sweep_name="flux",
                    sweep_values=values,
                    update_cfg_fn=updateCfg,
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    Reset(
                                        "reset", ctx.cfg.get("reset", {"type": "none"})
                                    ),
                                    Pulse("init_pulse", ctx.cfg.get("init_pulse")),
                                    Pulse("probe_pulse", ctx.cfg["probe_pulse"]),
                                    Readout("readout", ctx.cfg["readout"]),
                                ],
                            ).acquire(soc, progress=False, callback=update_hook)
                        ),
                        result_shape=(len(gains),),
                    ),
                ),
                update_hook=lambda ctx: viewer.update(
                    values, gains, mist_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(results)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (values, gains, signals)

        return values, gains, signals

    def analyze(
        self,
        result: Optional[MistFluxDepResultType] = None,
        *,
        mA_c: Optional[float] = None,
        period: Optional[float] = None,
        ac_coeff: Optional[float] = None,
        fig: Optional[go.Figure] = None,
    ) -> go.Figure:
        if result is None:
            result = self.last_result

        dev_values, pdrs, signals = result

        if mA_c is not None and period is not None:
            xs = mA2flx(dev_values, mA_c, period)
        else:
            xs = dev_values

        amp_diff = mist_signal2real(signals)

        if fig is None:
            fig = go.Figure()

        if mA_c is not None and period is not None:
            xlabel = r"$\phi$ (a.u.)"
        else:
            xlabel = r"$A$ (mA)"
        fig.update_xaxes(title_text=xlabel, title_font_size=14, range=[xs[0], xs[-1]])

        if ac_coeff is None:
            ys = pdrs
            ylabel = "probe gain (a.u.)"
        else:
            ys = ac_coeff * pdrs**2
            ylabel = r"$\bar n$"
        fig.update_yaxes(title_text=ylabel, title_font_size=12, range=[ys[0], ys[-1]])

        fig.add_trace(
            go.Heatmap(x=xs, y=ys, z=amp_diff.T, showscale=False, colorscale="Greys")
        )

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[MistFluxDepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep/mist",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, gains, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Power", "unit": "a.u.", "values": gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
