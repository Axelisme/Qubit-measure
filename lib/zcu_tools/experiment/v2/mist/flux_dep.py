from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from pydantic import Field

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import (
    set_flux_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runner import MeasureSession, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.notebook.analysis.fluxdep import add_secondary_xaxis
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.simulate import value2flux


@dataclass(frozen=True)
class FluxDepResult:
    values: NDArray[np.float64]
    gains: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: FluxDepCfg | None = None


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


class FluxDepModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class FluxDepSweepCfg(ConfigBase):
    flux: SweepCfg
    gain: SweepCfg


class FluxDepCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FluxDepModuleCfg
    # Field(...) makes dev required in this subclass, overriding the Optional
    # default from ExpCfgModel — intentional Pydantic pattern (type: ignore[override]).
    dev: Mapping[str, DeviceInfo] = Field(...)  # type: ignore[override]
    sweep: FluxDepSweepCfg


class FluxDepExp(PersistableExperiment[FluxDepResult, FluxDepCfg]):
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("gains", "Power", "a.u."),
            Axis("values", "Flux value", "a.u."),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=FluxDepResult,
        cfg_type=FluxDepCfg,
        tag="mist/flux_dep",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: FluxDepCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FluxDepResult:
        cfg = deepcopy(cfg)
        modules = cfg.modules

        # predict sweep points
        values = sweep2array(cfg.sweep.flux, allow_array=True)
        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, FluxDepCfg],
            update_hook: Callable[[int, list[NDArray[np.float64]]], None] | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            setup_devices(cfg, progress=False)
            modules = cfg.modules

            assert update_hook is not None

            gain_sweep = cfg.sweep.gain
            gain_param = sweep2param("gain", gain_sweep)
            modules.probe_pulse.set_param("gain", gain_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    Pulse("probe_pulse", modules.probe_pulse),
                    Readout("readout", modules.readout),
                ],
                sweep=[("gain", gain_sweep)],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot2DwithLine(
            "Flux device value",
            "Readout power (a.u.)",
            line_axis=1,
            num_lines=5,
            title="MIST over FLux",
        ) as viewer:
            with MeasureSession(cfg) as run:
                signals_buffer = run.buffer(
                    (len(values), len(gains)),
                    dtype=np.complex128,
                    on_update=lambda data: viewer.update(
                        values, gains, mist_signal2real(data)
                    ),
                )
                for step in run.scan("flux", values.tolist()):
                    set_flux_in_dev_cfg(step.cfg.dev, step.value)
                    signals_buffer[step].measure(measure_fn, pbar_n=step.cfg.rounds)
                signals = signals_buffer.array

        return FluxDepResult(
            values=values, gains=gains, signals=signals, cfg_snapshot=cfg
        )

    @retrieve_result
    def analyze(
        self,
        result: FluxDepResult | None = None,
        *,
        flux_half: float | None = None,
        flux_period: float | None = None,
        ac_coeff: float | None = None,
        fig: go.Figure | None = None,
        secondary_xaxis: bool = True,
        auto_range: bool = True,
        **fig_kwargs,
    ) -> go.Figure:
        assert result is not None, "no result found"

        dev_values = result.values
        gains = result.gains
        signals = result.signals

        if flux_half is not None and flux_period is not None:
            xs = np.asarray(value2flux(dev_values, flux_half, flux_period))
        else:
            xs = dev_values

        amp_diff = mist_signal2real(signals)

        if fig is None:
            fig = go.Figure()
        assert fig is not None

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
        ys = np.asarray(ys)
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
