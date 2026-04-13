from __future__ import annotations

from copy import deepcopy

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    Mapping,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.notebook.analysis.fluxdep import add_secondary_xaxis
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
from zcu_tools.simulate import value2flux
from zcu_tools.utils.datasaver import load_data, save_data

FluxDepResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * signals.shape[1]), 1)

    mist_signals = np.abs(
        signals - np.mean(signals[:, :avg_len], axis=1, keepdims=True)
    )
    if np.all(np.isnan(mist_signals)):
        return mist_signals

    ref_signals = np.sort(mist_signals.flatten())[: int(0.5 * mist_signals.size)]
    mist_signals = np.clip(mist_signals, 0, 10 * np.nanmedian(ref_signals))

    return mist_signals


class FluxDepModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class FluxDepCfg(ModularProgramCfg, TaskCfg):
    modules: FluxDepModuleCfg
    dev: Mapping[str, DeviceInfo]
    sweep: dict[str, SweepCfg]


class FluxDepExp(AbsExperiment[FluxDepResult, FluxDepCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> FluxDepResult:
        _cfg = check_type(deepcopy(cfg), FluxDepCfg)
        modules = _cfg["modules"]

        # predict sweep points
        values = sweep2array(_cfg["sweep"]["flux"], allow_array=True)
        gains = sweep2array(
            _cfg["sweep"]["gain"],
            "gain",
            {"soccfg": soccfg, "gen_ch": modules["probe_pulse"].ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg: FluxDepCfg = cast(FluxDepCfg, ctx.cfg)
            modules = cfg["modules"]

            assert update_hook is not None

            gain_sweep = cfg["sweep"]["gain"]
            gain_param = sweep2param("gain", gain_sweep)
            modules["probe_pulse"].set_param("gain", gain_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse("init_pulse", modules.get("init_pulse")),
                    Pulse("probe_pulse", modules["probe_pulse"]),
                    Readout("readout", modules["readout"]),
                ],
                sweep=[("gain", gain_sweep)],
            ).acquire(
                soc, progress=False, callback=update_hook, **(acquire_kwargs or {})
            )

        with LivePlot2DwithLine(
            "Flux device value",
            "Readout power (a.u.)",
            line_axis=1,
            num_lines=5,
            title="MIST over FLux",
        ) as viewer:
            signals = run_task(
                task=Task(measure_fn=measure_fn, result_shape=(len(gains),)).scan(
                    "flux",
                    values.tolist(),
                    before_each=lambda _, ctx, value: set_flux_in_dev_cfg(
                        ctx.cfg["dev"], value
                    ),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    values, gains, mist_signal2real(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (values, gains, signals)

        return values, gains, signals

    def analyze(
        self,
        result: Optional[FluxDepResult] = None,
        *,
        flux_half: Optional[float] = None,
        flux_period: Optional[float] = None,
        ac_coeff: Optional[float] = None,
        fig: Optional[go.Figure] = None,
        secondary_xaxis: bool = True,
        auto_range: bool = True,
        **fig_kwargs,
    ) -> go.Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        dev_values, gains, signals = result

        if flux_half is not None and flux_period is not None:
            xs = value2flux(dev_values, flux_half, flux_period)
        else:
            xs = dev_values

        amp_diff = mist_signal2real(signals)

        if fig is None:
            fig = go.Figure()

        if flux_half is not None and flux_period is not None:
            xlabel = r"$\phi$ (a.u.)"
        else:
            xlabel = r"$A$ (mA)"
        fig.update_xaxes(title_text=xlabel, title_font_size=14)

        if ac_coeff is None:
            ys = gains
            ylabel = "probe gain (a.u.)"
        else:
            ys = ac_coeff * gains**2
            ylabel = r"$\bar n$"
        fig.update_yaxes(title_text=ylabel, title_font_size=12)

        fig.add_trace(
            go.Heatmap(x=xs, y=ys, z=amp_diff.T, showscale=False, colorscale="Greys"),
            **fig_kwargs,
        )

        if secondary_xaxis:
            assert flux_half is not None and flux_period is not None
            add_secondary_xaxis(fig, xs, dev_values, **fig_kwargs)

        if auto_range:
            fig.update_xaxes(range=[xs[0], xs[-1]])
            fig.update_yaxes(range=[ys[0], ys[-1]])

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[FluxDepResult] = None,
        comment: Optional[str] = None,
        tag: str = "mist/flux_dep",
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

    def load(self, filepath: str, **kwargs) -> FluxDepResult:
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
