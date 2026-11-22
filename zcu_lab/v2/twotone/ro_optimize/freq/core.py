from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_as_signal
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    Pulse,
    PulseReadout,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
    sweep2param,
)
from zcu_tools.utils.process import SmoothingOptions, SmoothMethod, smooth_signal1d

from zcu_lab.v2.support import live_line


@dataclass(frozen=True)
class FreqResult:
    """Readout frequencies in MHz and moment-SNR scores (frequency,)."""

    freqs: NDArray[np.float64]
    signals: NDArray[np.float64]


@pydantic_dataclass(frozen=True)
class FreqAnalyzeOptions:
    smooth: float = 1.0
    smooth_method: SmoothMethod = "wavelet"
    wavelet: str = "sym4"
    wavelet_level: int = 0


@dataclass(frozen=True)
class FreqAnalysis:
    best_freq: float


class FreqCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Choose the qubit reset before preparation, or leave blank to skip reset.",
        ),
    ] = None
    qub_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit π pulse",
            description="Select the calibrated π pulse used to prepare the excited branch.",
        ),
    ]
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout pulse and ADC window",
            description="Select the resonator pulse and ADC timing for reading each qubit preparation.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Readout frequency",
            unit="MHz",
            description="Enter the readout RF sweep in MHz when a mixer is set, or the DDS sweep without a mixer.",
        ),
    ]
    skew_penalty: Annotated[
        float,
        FieldMeta(
            label="IQ skew penalty weight",
            unit="a.u.",
            decimals=3,
            description="Set the penalty weight for asymmetric IQ distributions; zero disables the penalty.",
        ),
    ] = Field(default=0.0, ge=0.0)

    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the number of shots per qubit preparation and setting in each round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of software acquisition rounds.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Qubit relaxation wait",
            unit="µs",
            decimals=3,
            description="Enter the qubit relaxation time in µs to wait after each preparation and readout.",
        ),
    ]


class FreqExp(Experiment[FreqResult, FreqCfg]):
    Options: ClassVar[type[FreqAnalyzeOptions]] = FreqAnalyzeOptions

    # freq stores Hz on disk -> scale=1e6 (disk = memory * 1e6)
    AXES_SPEC: ClassVar[AxesSpec[FreqResult, FreqCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u.", dtype=np.float64),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/ro_optimize/freq",
    )

    def run(
        self,
        cfg: FreqCfg,
        *,
        context: RunContext,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FreqResult:
        """Measure readout-frequency dependence of ground/excited IQ separation.

        At each frequency, apply the optional reset, omit or apply qub_pulse,
        then read out. Configure qub_pulse for the intended excitation.
        Return MHz frequencies on the common generator/ADC grid and
        moment-based SNR shaped (Nfreq,). With mixer_freq set, the sweep and
        returned frequencies are absolute RF.
        skew_penalty optionally reduces scores for distorted IQ distributions.
        This score is not assignment fidelity. acquire_kwargs is forwarded
        to summed acquisition.
        """
        # 1. Quantize absolute RF on the common generator/ADC grid, including mixer.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)

        freqs = sweep2array(
            cfg.freq,
            "freq",
            {
                "soccfg": soccfg,
                "gen_ch": cfg.readout.gen_ch,
                "ro_ch": cfg.readout.ro_ch,
                "mixer_freq": cfg.readout.mixer_freq,
            },
        )

        # 2. Show the separation score rather than either preparation's mean IQ.
        fig, ax = context.plots.subplots(num="measurement")
        fig.suptitle("Readout frequency sweep · Resonator · IQ separation")
        ax.set(
            xlabel="Readout frequency (MHz)",
            ylabel="g/e IQ separation SNR",
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        line = live_line(ax)
        ax.lines[0].set_marker("o")

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Refresh the measurement artists from the current acquired data."""
            progress = ax.set_title(
                f"{np.isfinite(data).sum()}/{len(freqs)} settings scored"
            )
            return (*line.set(freqs, np.abs(data)), progress)

        fig.show()
        signals_buffer = SignalBuffer(
            (len(freqs),),
            dtype=np.float64,
            on_update=update,
        )

        # 3. Compare pulse-off/on branches at each hardware-swept readout frequency.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            freq_sweep = cfg.freq
            freq_param = sweep2param("freq", freq_sweep)
            cfg.readout.set_param("freq", freq_param)

            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            builder.add_loop("ge", 2)
            builder.add_sweep("freq", freq_sweep)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            with (
                builder.branch(by="ge", align_timeline=False) as cases,
                cases.case(1),
            ):
                builder.emit(Pulse("qub_pulse", cfg.qub_pulse))
            builder.emit(Readout("readout", cfg.readout))
            program = builder.build()
            sched.run_acquire_summed(
                program,
                soc,
                reducer=shot_sums(program, [IQ("readout", order=3)]),
                rounds=cfg.rounds,
                options=RunOptions(
                    converter=lambda sums: snr_as_signal(
                        sums,
                        skew_penalty=cfg.skew_penalty,
                    )
                ),
                **(acquire_kwargs or {}),
            )

        # 4. Return the moment-derived score; raw shots and IQ moments are not retained.
        signals = signals_buffer.array

        return FreqResult(freqs, signals)

    def analyze(
        self,
        source: RunRecord[FreqCfg, FreqResult],
        options: FreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> FreqAnalysis:
        """Select the frequency with maximum smoothed SNR and plot it.

        options controls smoothing; unmeasured NaN scores become zero before
        smoothing. Return best_freq in MHz. source.cfg may be None.
        """
        # 1. Select the readout frequency maximizing smoothed preparation separation.
        result = source.result
        freqs, signals = result.freqs, result.signals

        snrs = np.abs(signals)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = smooth_signal1d(
            snrs,
            options=SmoothingOptions(
                method=options.smooth_method,
                sigma=options.smooth,
                wavelet=options.wavelet,
                wavelet_level=options.wavelet_level,
            ),
            axis=0,
        )

        max_id = np.argmax(snrs)
        max_freq = float(freqs[max_id])
        max_snr = float(snrs[max_id])

        # 2. Inspect the SNR maximum on the smoothed frequency response.
        fig, ax = plots.subplots(num="fit", figsize=config.figsize)
        fig.suptitle("Readout frequency sweep · Resonator · IQ separation")
        raw = np.abs(signals)
        ax.plot(freqs, raw, "o", color="C0", label="Measured SNR")
        if not np.array_equal(raw, snrs, equal_nan=True):
            ax.plot(freqs, snrs, color="C1", label="Selection SNR")
        ax.axvline(max_freq, color="C3", ls="--", label="Selected setting")
        ax.set(xlabel="Readout frequency (MHz)", ylabel="g/e IQ separation SNR")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(
            f"Selected readout frequency = {max_freq:.6g} MHz\n"
            f"Selection SNR = {max_snr:.2f}"
        )
        ax.legend(loc="best")

        return FreqAnalysis(max_freq)
