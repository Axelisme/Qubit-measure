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
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.utils.process import SmoothingOptions, SmoothMethod, smooth_signal1d

from zcu_lab.v2.support import live_line


@dataclass(frozen=True)
class LengthResult:
    """ADC integration windows in us and moment-SNR scores (length,)."""

    lengths: NDArray[np.float64]
    signals: NDArray[np.float64]


@pydantic_dataclass(frozen=True)
class LengthAnalyzeOptions:
    t0: Annotated[float | None, Field(ge=0)] = None
    smooth: float = 1.0
    smooth_method: SmoothMethod = "wavelet"
    wavelet: str = "sym4"
    wavelet_level: int = 0


@dataclass(frozen=True)
class LengthAnalysis:
    best_length: float


class LengthCfg(ConfigBase):
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
    length: Annotated[
        SweepCfg,
        FieldMeta(
            label="ADC window",
            unit="µs",
            description="Enter the ADC integration window sweep in µs; the RF pulse stays fixed for this scan.",
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


class LengthExp(Experiment[LengthResult, LengthCfg]):
    Options: ClassVar[type[LengthAnalyzeOptions]] = LengthAnalyzeOptions

    # lengths stored in seconds on disk -> scale=1e-6 (disk = memory * 1e-6)
    AXES_SPEC: ClassVar[AxesSpec[LengthResult, LengthCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("lengths", label="Readout Length", unit="s", scale=1e-6),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u.", dtype=np.float64),
        result_type=LengthResult,
        cfg_type=LengthCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/ro_optimize/length",
    )

    def run(
        self,
        cfg: LengthCfg,
        *,
        context: RunContext,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> LengthResult:
        """Measure readout SNR versus the ADC integration window.

        Host-scan the window in us. At each point, apply optional reset,
        omit or apply qub_pulse, then read out. Configure qub_pulse for the
        intended excitation. Keep the RF pulse at the largest window plus
        0.11 us throughout the scan. Return lengths and moment-based SNR
        shaped (Nlength,), optionally reduced by skew_penalty for distorted
        IQ distributions. SNR is not assignment fidelity.
        acquire_kwargs is forwarded to summed acquisition.
        """
        # 1. Quantize integration windows on the ADC clock, not the generator clock.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)

        readout_cfg = cfg.readout
        lengths = sweep2array(
            cfg.length,
            "time",
            {"soccfg": soccfg, "ro_ch": readout_cfg.ro_ch},
        )

        # 2. Display the score for each integration window, not averaged state IQ.
        fig, ax = context.plots.subplots(num="measurement")
        fig.suptitle("ADC window sweep · Resonator · IQ separation")
        ax.set(
            xlabel="ADC window (µs)",
            ylabel="g/e IQ separation SNR",
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        line = live_line(ax)
        ax.lines[0].set_marker("o")

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Refresh the measurement artists from the current acquired data."""
            progress = ax.set_title(
                f"{np.isfinite(data).sum()}/{len(lengths)} settings scored"
            )
            return (*line.set(lengths, np.abs(data)), progress)

        fig.show()
        signals_buffer = SignalBuffer(
            (len(lengths),),
            dtype=np.float64,
            on_update=update,
        )

        # 3. Rebuild per window; hold RF duration fixed while comparing pulse-off/on.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("length", lengths.tolist()):
                cfg.readout.set_param("ro_length", step.value)
                cfg.readout.set_param("length", lengths.max() + 0.11)

                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                )
                builder.add_loop("ge", 2)
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                with (
                    builder.branch(by="ge", align_timeline=False) as cases,
                    cases.case(1),
                ):
                    builder.emit(Pulse("qub_pulse", cfg.qub_pulse))
                builder.emit(PulseReadout("readout", cfg.readout))
                program = builder.build()
                step.run_acquire_summed(
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

        return LengthResult(lengths, signals)

    def analyze(
        self,
        source: RunRecord[LengthCfg, LengthResult],
        options: LengthAnalyzeOptions,
        *,
        plots: Plots,
    ) -> LengthAnalysis:
        """Choose the integration window from the smoothed SNR curve.

        options controls smoothing and t0 (us). None or zero selects pure SNR;
        positive t0 maximizes SNR/sqrt(length + t0), favoring shorter windows
        more strongly for smaller t0. Options reject negative t0 at construction.
        NaN scores become zero before smoothing. Return best_length in us
        and publish the figure. source.cfg may be None.
        """
        # 1. Optimize smoothed readout separation, optionally accounting for cycle-time overhead.
        result = source.result

        lengths, signals = result.lengths, result.signals

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

        if options.t0 is None or options.t0 == 0.0:
            max_id = np.argmax(snrs)
        else:
            max_id = np.argmax(snrs / np.sqrt(lengths + options.t0))

        max_length = float(lengths[max_id])
        max_snr = float(snrs[max_id])

        # 2. Inspect the chosen integration window on the smoothed SNR curve.
        fig, ax = plots.subplots(num="fit", figsize=config.figsize)
        fig.suptitle("ADC window sweep · Resonator · IQ separation")
        raw = np.abs(signals)
        ax.plot(lengths, raw, "o-", color="C0", label="Measured SNR")
        if not np.array_equal(raw, snrs, equal_nan=True):
            ax.plot(lengths, snrs, color="C1", label="Selection SNR")
        ax.axvline(max_length, color="C3", ls="--", label="Selected setting")
        ax.set(xlabel="ADC window (µs)", ylabel="g/e IQ separation SNR")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        objective = "Maximizes selection SNR"
        if options.t0:
            objective = f"Maximizes SNR / √(window + {options.t0:g} µs)"
        ax.set_title(
            f"Selected ADC window = {max_length:.6g} µs\n"
            f"Selection SNR = {max_snr:.2f}"
            f" · {objective}"
        )
        ax.legend(loc="best")

        return LengthAnalysis(max_length)
