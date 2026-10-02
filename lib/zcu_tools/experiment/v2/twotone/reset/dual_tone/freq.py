from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
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
    TwoPulseReset,
    sweep2param,
)
from zcu_tools.program.v2.modules import TwoPulseResetCfg
from zcu_tools.utils.process import (
    SmoothMethod,
    minus_background,
    rotate2real,
    smooth_signal_nd,
)


def dual_reset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals))


@dataclass(frozen=True)
class FreqResult:
    freqs1: NDArray[np.float64]
    freqs2: NDArray[np.float64]
    signals: NDArray[np.complex128]


class FreqModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    tested_reset: TwoPulseResetCfg
    readout: ReadoutCfg


class FreqSweepCfg(ConfigBase):
    freq1: SweepCfg
    freq2: SweepCfg


class FreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqModuleCfg
    sweep: FreqSweepCfg
    method: Literal["soft", "hard"] = "soft"


@dataclass(frozen=True)
class FreqAnalyzeOptions:
    smooth: float = 1.0
    smooth_method: SmoothMethod = "wavelet"
    wavelet: str = "sym4"
    wavelet_level: int = 0
    xname: str | None = None
    yname: str | None = None
    corner_as_background: bool = False


@dataclass(frozen=True)
class FreqAnalysis:
    freq1: float
    freq2: float


class FreqExp(PersistableExperiment[FreqResult, FreqCfg]):
    Options: ClassVar[type[FreqAnalyzeOptions]] = FreqAnalyzeOptions

    # signals memory layout is (Nfreq1, Nfreq2) = (outer, inner); native save/load
    # expect z == (outer, inner) == reversed(axes lengths), so axes order is
    # (freqs2 inner, freqs1 outer). both axes store MHz on disk (disk Hz).
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs2", "Frequency2", "Hz", scale=MHZ_TO_HZ),
            Axis("freqs1", "Frequency1", "Hz", scale=MHZ_TO_HZ),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        tag="twotone/reset/dual_tone/freq",
    )

    def run_soft(self, cfg: FreqCfg, *, context: RunContext) -> FreqResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )
        modules = cfg.modules

        freq1_sweep = cfg.sweep.freq1
        freq2_sweep = cfg.sweep.freq2

        reset_cfg = modules.tested_reset
        freqs1 = sweep2array(
            freq1_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse1_cfg.ch},
        )
        freqs2 = sweep2array(
            freq2_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse2_cfg.ch},
            allow_array=True,
        )

        viewer = context.plots.liveplot_2d_with_line(
            "measurement", "Frequency1 (MHz)", "Frequency2 (MHz)", line_axis=0
        )
        signals_buffer = SignalBuffer(
            (len(freqs2), len(freqs1)),
            on_update=lambda data: viewer.update(
                freqs1, freqs2, dual_reset_signal2real(data.T)
            ),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for freq2, step in sched.scan("freq2", freqs2.tolist()):
                modules = step.cfg.modules
                modules.tested_reset.set_param("freq2", freq2)
                modules.tested_reset.set_param(
                    "freq1", sweep2param("freq1", step.cfg.sweep.freq1)
                )
                _ = (
                    step.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        Pulse("init_pulse", modules.init_pulse),
                        TwoPulseReset("tested_reset", modules.tested_reset),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("freq1", step.cfg.sweep.freq1)
                    .build_and_acquire()
                )

        return FreqResult(freqs1, freqs2, signals_buffer.array.T)

    def run_hard(self, cfg: FreqCfg, *, context: RunContext) -> FreqResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )
        modules = cfg.modules

        reset_cfg = modules.tested_reset
        freqs1 = sweep2array(
            cfg.sweep.freq1,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse1_cfg.ch},
        )
        freqs2 = sweep2array(
            cfg.sweep.freq2,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse2_cfg.ch},
        )

        viewer = context.plots.liveplot_2d(
            "measurement", "Frequency1 (MHz)", "Frequency2 (MHz)"
        )
        signals_buffer = SignalBuffer(
            (len(freqs1), len(freqs2)),
            on_update=lambda data: viewer.update(
                freqs1, freqs2, dual_reset_signal2real(data)
            ),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            modules = sched.cfg.modules
            modules.tested_reset.set_param(
                "freq1", sweep2param("freq1", sched.cfg.sweep.freq1)
            )
            modules.tested_reset.set_param(
                "freq2", sweep2param("freq2", sched.cfg.sweep.freq2)
            )
            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    TwoPulseReset("tested_reset", modules.tested_reset),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("freq1", sched.cfg.sweep.freq1)
                .declare_sweep("freq2", sched.cfg.sweep.freq2)
                .build_and_acquire()
            )

        return FreqResult(freqs1, freqs2, signals_buffer.array)

    def run(self, cfg: FreqCfg, *, context: RunContext) -> FreqResult:
        if cfg.method == "soft":
            return self.run_soft(cfg, context=context)
        return self.run_hard(cfg, context=context)

    def analyze(
        self,
        source: RunRecord[FreqCfg, FreqResult],
        options: FreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> FreqAnalysis:
        result = source.result

        freqs1, freqs2, signals = result.freqs1, result.freqs2, result.signals

        # Apply smoothing for peak finding
        signals_smooth = smooth_signal_nd(
            signals,
            method=options.smooth_method,
            sigma=options.smooth,
            axes=(0, 1),
            wavelet=options.wavelet,
            wavelet_level=options.wavelet_level,
        )

        # Find peak in amplitude
        if options.corner_as_background:
            amps = np.abs(signals_smooth - signals_smooth[0, 0])
        else:
            amps = np.abs(minus_background(signals_smooth))

        freq1_opt = freqs1[np.argmax(np.max(amps, axis=1))]
        freq2_opt = freqs2[np.argmax(np.max(amps, axis=0))]

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        ax.imshow(
            rotate2real(signals.T).real,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(freqs1[0], freqs1[-1], freqs2[0], freqs2[-1]),
        )
        peak_label = f"({freq1_opt:.1f}, {freq2_opt:.1f}) MHz"
        ax.scatter(freq1_opt, freq2_opt, color="r", s=40, marker="*", label=peak_label)
        if options.xname is not None:
            ax.set_xlabel(f"{options.xname} Frequency (MHz)", fontsize=14)
        if options.yname is not None:
            ax.set_ylabel(f"{options.yname} Frequency (MHz)", fontsize=14)
        ax.legend(fontsize="x-large")
        ax.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

        return FreqAnalysis(freq1=float(freq1_opt), freq2=float(freq2_opt))
