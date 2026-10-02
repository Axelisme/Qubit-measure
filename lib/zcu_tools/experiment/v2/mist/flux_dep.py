from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import (
    set_flux_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
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


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * signals.shape[1]), 1)

    mist_signals = np.abs(
        signals - np.mean(signals[:, :avg_len], axis=1, keepdims=True)
    )
    if np.all(np.isnan(mist_signals)):
        return mist_signals

    ref_signals = np.sort(mist_signals.flatten())[: int(0.5 * mist_signals.size)]
    return np.clip(mist_signals, 0, 10 * np.nanmedian(ref_signals))


@dataclass(frozen=True)
class FluxDepAnalyzeOptions:
    flux_half: float | None = None
    flux_period: float | None = None
    ac_coeff: float | None = None
    secondary_xaxis: bool = True
    auto_range: bool = True


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

    def run(
        self,
        config: FluxDepCfg,
        *,
        context: RunContext,
    ) -> FluxDepResult:
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg.modules

        # predict sweep points
        values = sweep2array(cfg.sweep.flux, allow_array=True)
        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
        )

        viewer = context.plots.liveplot_2d_with_line(
            "measurement",
            "Flux device value",
            "Readout power (a.u.)",
            line_axis=1,
            num_lines=5,
            title="MIST over FLux",
        )
        signals_buffer = SignalBuffer(
            (len(values), len(gains)),
            on_update=lambda data: viewer.update(values, gains, mist_signal2real(data)),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for flux, step in sched.scan("flux", values.tolist()):
                set_flux_in_dev_cfg(step.cfg.dev, flux)
                setup_devices(
                    step.cfg,
                    context.devices,
                    progress=False,
                    cancel_signal=context.cancel_signal,
                )
                modules = step.cfg.modules
                modules.probe_pulse.set_param(
                    "gain", sweep2param("gain", step.cfg.sweep.gain)
                )
                _ = (
                    step.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        Pulse("init_pulse", modules.init_pulse),
                        Pulse("probe_pulse", modules.probe_pulse),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("gain", step.cfg.sweep.gain)
                    .build_and_acquire()
                )

        return FluxDepResult(values=values, gains=gains, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[FluxDepCfg, FluxDepResult],
        options: FluxDepAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        result = source.result
        flux_half, flux_period = options.flux_half, options.flux_period
        if options.secondary_xaxis and (flux_half is None or flux_period is None):
            raise ValueError("Secondary flux axis requires flux_half and flux_period")

        dev_values, gains, signals = result.values, result.gains, result.signals
        if flux_half is not None and flux_period is not None:
            xs = np.asarray(value2flux(dev_values, flux_half, flux_period))
            xlabel = r"$\phi$ (a.u.)"
        else:
            xs = dev_values
            xlabel = r"$A$ (mA)"

        amp_diff = mist_signal2real(signals)
        if options.ac_coeff is None:
            ys = gains
            ylabel = "probe gain (a.u.)"
        else:
            ys = options.ac_coeff * gains**2
            ylabel = r"$\bar n$"
        ys = np.asarray(ys)

        _, ax = plots.subplots("fit", figsize=config.figsize)
        ax.pcolormesh(xs, ys, amp_diff.T, shading="nearest", cmap="Greys")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=12)

        if options.secondary_xaxis:
            # Match the source samples and scientific labels of the former overlay.
            count = len(xs)
            if count <= 12:
                tick_indices = np.arange(count)
            else:
                tick_indices = np.unique(
                    np.round(np.linspace(0, count - 1, 12)).astype(int)
                )
            secondary = ax.secondary_xaxis("top")
            secondary.set_xticks(
                xs[tick_indices],
                labels=[f"{value:.1e}" for value in dev_values[tick_indices]],
            )

        if options.auto_range:
            ax.set_xlim(xs[0], xs[-1])
            ax.set_ylim(ys[0], ys[-1])
