from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    TwoPulseReset,
)
from zcu_tools.utils.process import (
    SmoothingOptions,
    SmoothMethod,
    minus_background,
    rotate2real,
    smooth_signal_nd,
)

from zcu_lab.v2.support import live_heatmap, live_line


def dual_reset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals))


@dataclass(frozen=True)
class FreqResult:
    freqs1: NDArray[np.float64]
    freqs2: NDArray[np.float64]
    signals: NDArray[np.complex128]


class FreqCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init pulse")] = None
    tested_reset: Annotated[TwoPulseReset.Cfg, FieldMeta(label="Tested reset")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    freq1: Annotated[SweepCfg, FieldMeta(label="Freq1", unit="MHz")]
    freq2: Annotated[SweepCfg, FieldMeta(label="Freq2", unit="MHz")]
    method: Literal["soft", "hard"] = "soft"

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0


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


class FreqExp(Experiment[FreqResult, FreqCfg]):
    Options: ClassVar[type[FreqAnalyzeOptions]] = FreqAnalyzeOptions

    # signals memory layout is (Nfreq1, Nfreq2) = (outer, inner); native save/load
    # expect z == (outer, inner) == reversed(axes lengths), so axes order is
    # (freqs2 inner, freqs1 outer). both axes store MHz on disk (disk Hz).
    AXES_SPEC: ClassVar[AxesSpec[FreqResult, FreqCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs2", label="Frequency2", unit="Hz", scale=1e6),
            AxesSpec.Axis("freqs1", label="Frequency1", unit="Hz", scale=1e6),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        cfg_schema_version="3.0",
        tag="twotone/reset/dual_tone/freq",
    )

    def run(self, cfg: FreqCfg, *, context: RunContext) -> FreqResult:
        """Map the averaged readout response against both reset-tone frequencies.

        Apply the optional reset and preparation pulse, the tested two-pulse
        reset, then read out. Soft mode scans frequency 2 on the host and sweeps
        frequency 1 in hardware; hard mode sweeps both in hardware. Return the
        quantized MHz axes and mean complex IQ shaped (Nfreq1, Nfreq2).
        Cancellation leaves unacquired points as NaN.
        """
        # 1. Prepare hardware-grid axes without changing the caller's cfg.
        cfg = deepcopy(cfg)
        soccfg = context.soccfg
        reset_cfg = cfg.tested_reset
        freqs1 = sweep2array(
            cfg.freq1,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse1_cfg.ch},
        )
        freqs2 = sweep2array(
            cfg.freq2,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse2_cfg.ch},
            allow_array=cfg.method == "soft",
        )

        if cfg.method == "soft":
            return self._run_soft(cfg, freqs1, freqs2, context=context)
        return self._run_hard(cfg, freqs1, freqs2, context=context)

    def _run_soft(
        self,
        cfg: FreqCfg,
        freqs1: NDArray[np.float64],
        freqs2: NDArray[np.float64],
        *,
        context: RunContext,
    ) -> FreqResult:
        # Host rows are frequency 2; the Result restores frequency-1 rows.
        soc, soccfg = context.soc, context.soccfg
        plots = context.plots
        fig, _ = plots.subplots(nrows=1, ncols=2, num="measurement")
        map_ax, line_ax = fig.axes
        map_ax.set(xlabel="Frequency1 (MHz)", ylabel="Frequency2 (MHz)")
        line_ax.set_xlabel("Frequency1 (MHz)")
        viewer = live_heatmap(map_ax)
        line = live_line(line_ax)
        line_ax.lines[0].set(label="current line", marker=".", color="C0")

        @plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
            real = dual_reset_signal2real(data.T)
            valid = np.flatnonzero(~np.all(np.isnan(real), axis=0))
            current = real[:, valid[-1]] if valid.size else np.full(len(freqs1), np.nan)
            return (*viewer.set(freqs1, freqs2, real), *line.set(freqs1, current))

        fig.show()
        signals_buffer = SignalBuffer(
            (len(freqs2), len(freqs1)),
            on_update=update,
        )
        with Schedule(None, signals_buffer, scope=context.scope) as sched:
            for freq2, step in sched.scan("freq2", freqs2.tolist()):
                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                )
                freq1 = builder.add_sweep("freq1", cfg.freq1)
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.emit(Pulse("init_pulse", cfg.init_pulse))
                builder.emit(
                    TwoPulseReset(
                        "tested_reset",
                        cfg.tested_reset,
                        freq1=freq1,
                        freq2=freq2,
                    )
                )
                builder.emit(Readout("readout", cfg.readout))
                step.run_acquire(builder.build(), soc, rounds=cfg.rounds)

        return FreqResult(freqs1, freqs2, signals_buffer.array.T)

    def _run_hard(
        self,
        cfg: FreqCfg,
        freqs1: NDArray[np.float64],
        freqs2: NDArray[np.float64],
        *,
        context: RunContext,
    ) -> FreqResult:
        # Sweep frequency 1 outside frequency 2 in one FPGA program.
        soc, soccfg = context.soc, context.soccfg
        plots = context.plots
        fig, ax = plots.subplots(num="measurement")
        ax.set(xlabel="Frequency1 (MHz)", ylabel="Frequency2 (MHz)")
        viewer = live_heatmap(ax)

        @plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
            return viewer.set(freqs1, freqs2, dual_reset_signal2real(data))

        fig.show()
        signals_buffer = SignalBuffer(
            (len(freqs1), len(freqs2)),
            on_update=update,
        )
        with Schedule(None, signals_buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            freq1 = builder.add_sweep("freq1", cfg.freq1)
            freq2 = builder.add_sweep("freq2", cfg.freq2)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(
                TwoPulseReset(
                    "tested_reset",
                    cfg.tested_reset,
                    freq1=freq1,
                    freq2=freq2,
                )
            )
            builder.emit(Readout("readout", cfg.readout))
            sched.run_acquire(builder.build(), soc, rounds=cfg.rounds)

        return FreqResult(freqs1, freqs2, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[FreqCfg, FreqResult],
        options: FreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> FreqAnalysis:
        """Find the reset-tone pair with the largest smoothed IQ contrast.

        Smooth the two-frequency map using options, then subtract either its
        first corner or the estimated background. Return the peak coordinates
        in MHz and plot the measured IQ map; this contrast is not reset fidelity.
        The source cfg may be absent.
        """
        result = source.result

        freqs1, freqs2, signals = result.freqs1, result.freqs2, result.signals

        # 1. Smooth the measured map before selecting a peak.
        signals_smooth = smooth_signal_nd(
            signals,
            options=SmoothingOptions(
                method=options.smooth_method,
                sigma=options.smooth,
                wavelet=options.wavelet,
                wavelet_level=options.wavelet_level,
            ),
            axes=(0, 1),
        )

        # 2. Select the largest contrast against the chosen background.
        if options.corner_as_background:
            amps = np.abs(signals_smooth - signals_smooth[0, 0])
        else:
            amps = np.abs(minus_background(signals_smooth))

        freq1_opt = freqs1[np.argmax(np.max(amps, axis=1))]
        freq2_opt = freqs2[np.argmax(np.max(amps, axis=0))]

        # 3. Mark the selected pair on the original, unsmoothed IQ map.
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
