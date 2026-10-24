from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
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
    TwoPulseReset,
    sweep2param,
)
from zcu_tools.utils.process import (
    SmoothingOptions,
    SmoothMethod,
    minus_background,
    rotate2real,
    smooth_signal_nd,
)

from zcu_lab.v2._support import live_heatmap, live_heatmap_with_line


def dual_reset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals))


@dataclass(frozen=True)
class FreqResult:
    freqs1: NDArray[np.float64]
    freqs2: NDArray[np.float64]
    signals: NDArray[np.complex128]


class FreqCfg(ConfigBase):
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
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init pulse")] = None
    tested_reset: Annotated[TwoPulseReset.Cfg, FieldMeta(label="Tested reset")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    freq1: Annotated[SweepCfg, FieldMeta(label="Freq1", unit="MHz")]
    freq2: Annotated[SweepCfg, FieldMeta(label="Freq2", unit="MHz")]
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


class FreqExp(Experiment[FreqResult, FreqCfg]):
    Options: ClassVar[type[FreqAnalyzeOptions]] = FreqAnalyzeOptions

    # signals memory layout is (Nfreq1, Nfreq2) = (outer, inner); native save/load
    # expect z == (outer, inner) == reversed(axes lengths), so axes order is
    # (freqs2 inner, freqs1 outer). both axes store MHz on disk (disk Hz).
    AXES_SPEC: ClassVar[AxesSpec[FreqResult, FreqCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs2", "Frequency2", "Hz", scale=1e6),
            AxesSpec.Axis("freqs1", "Frequency1", "Hz", scale=1e6),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
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
        soc, soccfg = context.soc, context.soccfg
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
            # 2. Scan frequency 2 on the host; keep frequency 1 as the inner loop.
            soft_viewer = live_heatmap_with_line(
                context.plots,
                "measurement",
                "Frequency1 (MHz)",
                "Frequency2 (MHz)",
                line_axis=0,
            )
            signals_buffer = SignalBuffer(
                (len(freqs2), len(freqs1)),
                on_update=lambda data: soft_viewer.update(
                    freqs1, freqs2, dual_reset_signal2real(data.T)
                ),
            )
            with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
                for freq2, step in sched.scan("freq2", freqs2.tolist()):
                    modules = step.cfg
                    modules.tested_reset.set_param("freq2", freq2)
                    modules.tested_reset.set_param(
                        "freq1", sweep2param("freq1", step.cfg.freq1)
                    )

                    # 3. Prepare, apply the tested reset, and collect averaged IQ.
                    _ = (
                        step.prog_builder(
                            soc,
                            soccfg,
                            reps=step.cfg.reps,
                            relax_delay=step.cfg.relax_delay,
                        )
                        .add(
                            Reset("reset", modules.reset),
                            Pulse("init_pulse", modules.init_pulse),
                            TwoPulseReset("tested_reset", modules.tested_reset),
                            Readout("readout", modules.readout),
                        )
                        .declare_sweep("freq1", step.cfg.freq1)
                        .build_and_acquire(rounds=cfg.rounds)
                    )

            # Host rows are frequency 2; restore the canonical frequency-1 rows.
            return FreqResult(freqs1, freqs2, signals_buffer.array.T)

        # 2. Sweep both frequencies in hardware, frequency 1 outside frequency 2.
        hard_viewer = live_heatmap(
            context.plots, "measurement", "Frequency1 (MHz)", "Frequency2 (MHz)"
        )
        signals_buffer = SignalBuffer(
            (len(freqs1), len(freqs2)),
            on_update=lambda data: hard_viewer.update(
                freqs1, freqs2, dual_reset_signal2real(data)
            ),
        )
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            modules = sched.cfg
            modules.tested_reset.set_param(
                "freq1", sweep2param("freq1", sched.cfg.freq1)
            )
            modules.tested_reset.set_param(
                "freq2", sweep2param("freq2", sched.cfg.freq2)
            )

            # 3. Prepare, apply the tested reset, and collect the averaged IQ map.
            _ = (
                sched.prog_builder(
                    soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
                )
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    TwoPulseReset("tested_reset", modules.tested_reset),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("freq1", sched.cfg.freq1)
                .declare_sweep("freq2", sched.cfg.freq2)
                .build_and_acquire(rounds=cfg.rounds)
            )

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
