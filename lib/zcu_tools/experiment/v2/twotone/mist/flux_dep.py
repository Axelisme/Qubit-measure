from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from typing_extensions import NotRequired

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.notebook.analysis.fluxdep import add_secondary_xaxis
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
from zcu_tools.simulate import mA2flx
from zcu_tools.utils.datasaver import load_data, save_data

MistFluxDepResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * signals.shape[1]), 1)

    mist_signals = np.abs(
        signals - np.mean(signals[:, :avg_len], axis=1, keepdims=True)
    )
    if np.all(np.isnan(mist_signals)):
        return mist_signals

    import matplotlib.pyplot as plt

    ref_signals = np.sort(mist_signals.flatten())[: int(0.5 * mist_signals.size)]
    mist_signals = np.clip(mist_signals, 0, 10 * np.nanmedian(ref_signals))

    return mist_signals


class MistFluxDepTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg

    dev: Mapping[str, DeviceInfo]


class MistFluxDepExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: MistFluxDepTaskConfig) -> MistFluxDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        flx_sweep = cfg["sweep"]["flux"]
        cfg["sweep"] = {"gain": cfg["sweep"]["gain"]}

        # predict sweep points
        values = sweep2array(flx_sweep)
        gains = sweep2array(cfg["sweep"]["gain"])

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
            signals = run_task(
                task=SoftTask(
                    sweep_name="flux",
                    sweep_values=values.tolist(),
                    update_cfg_fn=lambda _, ctx, value: set_flux_in_dev_cfg(
                        ctx.cfg["dev"], value
                    ),
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
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    values, gains, mist_signal2real(np.asarray(ctx.data))
                ),
            )
            signals = np.asarray(signals)

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
        secondary_xaxis: bool = True,
        auto_range: bool = True,
        **fig_kwargs,
    ) -> go.Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

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
        fig.update_xaxes(title_text=xlabel, title_font_size=14)

        if ac_coeff is None:
            ys = pdrs
            ylabel = "probe gain (a.u.)"
        else:
            ys = ac_coeff * pdrs**2
            ylabel = r"$\bar n$"
        fig.update_yaxes(title_text=ylabel, title_font_size=12)

        fig.add_trace(
            go.Heatmap(x=xs, y=ys, z=amp_diff.T, showscale=False, colorscale="Greys"),
            **fig_kwargs,
        )

        if secondary_xaxis:
            assert mA_c is not None and period is not None
            add_secondary_xaxis(fig, xs, dev_values, **fig_kwargs)

        if auto_range:
            fig.update_xaxes(range=[xs[0], xs[-1]])
            fig.update_yaxes(range=[ys[0], ys[-1]])

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

    def load(self, filepath: str, **kwargs) -> MistFluxDepResultType:
        signals, values, gains = load_data(filepath, **kwargs)
        assert values is not None and gains is not None
        assert len(values.shape) == 1 and len(gains.shape) == 1
        assert signals.shape == (len(values), len(gains))

        values = values.astype(np.float64)
        gains = gains.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (values, gains, signals)

        return values, gains, signals
