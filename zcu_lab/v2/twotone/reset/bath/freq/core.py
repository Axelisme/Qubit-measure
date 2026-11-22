from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass
from qick.asm_v2 import QickSweep1D
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    BathReset,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)
from zcu_tools.utils.process import SmoothingOptions, SmoothMethod, smooth_signal_nd

from zcu_lab.v2.support import live_heatmap

_TITLE = "Bath reset cavity sweep · qubit · tomography contrast"
_XLABEL = "Cavity drive RF frequency (MHz)"
_YLABEL = "Cavity drive gain (a.u.)"
_COLOR_LABEL = "Opposite-phase IQ contrast (a.u.²)"


@dataclass(frozen=True)
class FreqGainResult:
    """Mean IQ (phase, gain, frequency); phases are offsets in degrees, freqs in MHz."""

    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


@pydantic_dataclass(frozen=True)
class FreqGainAnalyzeOptions:
    smooth: float = 1.0
    smooth_method: SmoothMethod = "wavelet"
    wavelet: str = "sym4"
    wavelet_level: int = 0


@dataclass(frozen=True)
class FreqGainAnalysis:
    gain: float
    freq: float


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convert mean complex IQ to the real contrast used by live and analysis plots."""
    return (
        np.abs(signals[2] - signals[0]) ** 2 + np.abs(signals[3] - signals[1]) ** 2
    )  # (gain, freq)


class FreqGainCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select a preparation reset before each shot, or leave blank to skip it.",
        ),
    ] = None
    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Preparation pulse",
            description="Select the pulse after preparation reset, such as calibrated Xπ, or leave blank to skip it.",
        ),
    ] = None
    tested_reset: Annotated[
        BathReset.Cfg,
        FieldMeta(
            label="Tested reset",
            description="Fill in the simultaneous cavity and qubit drives and the final tomography π/2 pulse.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Fill in the calibrated resonator readout for the end of each shot.",
        ),
    ]
    gain: Annotated[
        SweepCfg,
        FieldMeta(
            label="Cavity drive gain",
            unit="a.u.",
            description="Enter the signed gain range of the cavity drive (a.u.).",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Cavity drive RF frequency",
            unit="MHz",
            description="Enter the RF frequency range of the cavity drive (MHz).",
        ),
    ]

    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per sweep point and sequence branch in each acquisition round (count).",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of acquisition rounds to average (count).",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Enter the delay after each shot to let the qubit relax (µs).",
        ),
    ]


class FreqGainExp(Experiment[FreqGainResult, FreqGainCfg]):
    Options: ClassVar[type[FreqGainAnalyzeOptions]] = FreqGainAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[FreqGainResult, FreqGainCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                "freqs",
                label="Cavity Frequency",
                unit="Hz",
                scale=1e6,
                dtype=np.float64,
            ),
            AxesSpec.Axis(
                "gains", label="Cavity drive Gain", unit="a.u.", dtype=np.float64
            ),
            AxesSpec.Axis(
                values=(0.0, 90.0, 180.0, 270.0),
                label="Pi2 Phase",
                unit="deg",
                dtype=np.float64,
            ),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u.", dtype=np.complex128),
        result_type=FreqGainResult,
        cfg_type=FreqGainCfg,
        cfg_schema_version="3.0",
        tag="twotone/reset/bath/freq_gain",
    )

    def run(
        self,
        config: FreqGainCfg,
        *,
        context: RunContext,
    ) -> FreqGainResult:
        """Measure bath-reset tomography contrast versus cavity gain and frequency.

        After optional reset and initialization, start the cavity and qubit
        tones together, apply the final pi/2 pulse after the cavity tone,
        then read out. Sweep
        its phase by 0, 90, 180 and 270 degrees relative to the configured
        phase. Return mean complex IQ with shape (4, Ngain, Nfreq),
        gains in a.u. within [-1, 1] and frequencies in MHz. Opposite-phase differences probe
        transverse contrast, not reset fidelity.
        """
        # 1. Prepare cavity-drive coordinates and four relative tomography phases.
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg

        reset_cfg = cfg.tested_reset
        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": reset_cfg.cavity_tone_cfg.ch},
        )
        freqs = sweep2array(
            cfg.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.cavity_tone_cfg.ch},
        )

        # 2. Display opposite-phase contrast while retaining all four complex IQ planes.
        plots = context.plots
        fig, ax = plots.subplots(num="measurement")
        ax.set(xlabel=_XLABEL, ylabel=_YLABEL, title=_TITLE)
        viewer = live_heatmap(ax)
        fig.colorbar(ax.images[0], ax=ax, label=_COLOR_LABEL)
        ax.grid(False)

        @plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
            """Refresh the measurement artists from the current acquired data."""
            acquired = int(np.count_nonzero(np.isfinite(data).all(axis=0)))
            title = ax.set_title(
                f"{_TITLE}\nAcquired cells: {acquired}/{len(freqs) * len(gains)} count"
            )
            return (*viewer.set(freqs, gains, bathreset_signal2real(data).T), title)

        fig.show()
        signals_buffer = SignalBuffer((4, len(gains), len(freqs)), on_update=update)

        # 3. Resolve reset tomography across cavity gain and frequency in one program.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            builder.add_loop("phase", 4)
            gain = builder.add_sweep("gain", cfg.gain)
            freq = builder.add_sweep("freq", cfg.freq)
            phase = QickSweep1D("phase", 0.0, 270.0) + cfg.tested_reset.pi2_cfg.phase
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(
                BathReset(
                    "tested_reset",
                    cfg.tested_reset,
                    res_gain=gain,
                    res_freq=freq,
                    pi2_phase=phase,
                )
            )
            builder.emit(Readout("readout", cfg.readout))
            sched.run_acquire(builder.build(), soc, rounds=cfg.rounds)

        return FreqGainResult(
            gains=gains,
            freqs=freqs,
            signals=signals_buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[FreqGainCfg, FreqGainResult],
        options: FreqGainAnalyzeOptions,
        *,
        plots: Plots,
    ) -> FreqGainAnalysis:
        """Select cavity gain and frequency from the strongest smoothed contrast.

        options controls smoothing along gain and frequency, not phase.
        Return gain in a.u. within [-1, 1] and MHz frequency from the four-phase contrast
        map, and publish it as fit; source.cfg may be None.
        """
        result = source.result

        gains, freqs, signals = result.gains, result.freqs, result.signals

        # 1. Locate the strongest reset tomography contrast after spatial smoothing.
        smooth_signals = smooth_signal_nd(
            signals,
            options=SmoothingOptions(
                method=options.smooth_method,
                sigma=options.smooth,
                wavelet=options.wavelet,
                wavelet_level=options.wavelet_level,
            ),
            axes=(1, 2),
        ).astype(np.complex128)
        real_signals = bathreset_signal2real(smooth_signals)

        gain_opt = gains[np.argmax(np.max(real_signals, axis=1))]
        freq_opt = freqs[np.argmax(np.max(real_signals, axis=0))]

        # 2. Identify the optimal cavity gain and frequency on the smoothed contrast map.
        fig, ax = plots.subplots(num="fit")

        viewer = live_heatmap(ax)
        (image,) = viewer.set(freqs, gains, bathreset_signal2real(signals).T)
        left, right, bottom, top = image.get_extent()
        ax.set(xlim=(left, right), ylim=(bottom, top))
        fig.colorbar(image, ax=ax, label=_COLOR_LABEL)
        ax.grid(False)
        ax.scatter(
            freq_opt,
            gain_opt,
            color="white",
            edgecolors="black",
            s=64,
            marker="*",
            label=f"Selected cavity drive = {freq_opt:.3f} MHz, {gain_opt:.4g} a.u.",
        )
        ax.set(xlabel=_XLABEL, ylabel=_YLABEL, title=_TITLE)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16))
        fig.tight_layout()

        return FreqGainAnalysis(gain=float(gain_opt), freq=float(freq_opt))
