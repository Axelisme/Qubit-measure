from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_as_signal
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.device import set_rf_frequency


@dataclass(frozen=True)
class FreqResult:
    """Pump frequencies in MHz and moment-based GE SNR in acquisition order."""

    freqs: NDArray[np.float64]
    signals: NDArray[np.float64]


@dataclass(frozen=True)
class FreqAnalysis:
    best_freq: float


class FreqCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select a qubit reset before readout; leave empty to skip it.",
        ),
    ] = None
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit π pulse",
            description="Select the calibrated g→e pulse for the excited preparation only.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Select the calibrated resonator pulse and IQ integration for both preparations.",
        ),
    ]
    jpa_rf_dev: Annotated[
        DeviceName | None,
        FieldMeta(
            label="JPA pump RF source",
            description="Select the RF source that drives the JPA pump; leave empty to scan coordinates without tuning it.",
        ),
    ] = ""
    jpa_freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="JPA pump RF frequency",
            unit="MHz",
            description="Enter Start and Stop in absolute pump RF MHz and Expts as the number of frequency samples (count).",
        ),
    ]
    skew_penalty: Annotated[
        float,
        FieldMeta(
            label="IQ shape penalty",
            unit="a.u.",
            decimals=3,
            description="Enter a nonnegative weight for unequal IQ widths and skew; 0 uses separation SNR alone.",
        ),
    ] = Field(default=0.0, ge=0.0)
    seed: Annotated[
        int | None,
        FieldMeta(
            label="Scan-order seed",
            description=(
                "Enter a nonnegative integer to repeat the shuffled interior order; "
                "leave blank for a new order."
            ),
        ),
    ] = None
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per preparation at each setting in one round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of acquisition rounds per setting.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation wait",
            unit="µs",
            decimals=3,
            description="Enter the wait after readout for relaxation before the next shot (µs).",
        ),
    ]


class FreqExp(Experiment[FreqResult, FreqCfg]):
    # jpa_freq stored as Hz on disk -> scale=1e6; signals are SNR (float64)
    AXES_SPEC: ClassVar[AxesSpec[FreqResult, FreqCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("freqs", label="JPA Frequency", unit="Hz", scale=1e6),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u.", dtype=np.float64),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        tag="jpa/freq",
        cfg_schema_version="3.0",
    )

    def run(self, cfg: FreqCfg, *, context: RunContext) -> FreqResult:
        """Measure qubit readout SNR versus JPA pump frequency.

        At each pump setting, reset, select no pulse or a pi pulse, then
        read out both preparation branches. Their IQ moments give the SNR,
        with the configured skew penalty; this is not assignment fidelity.
        Return pump frequencies in MHz and one real SNR per point.

        Interior frequencies are randomized; endpoints stay in place.
        ``jpa_rf_dev`` names a device in ``context.devices``; None skips
        pump tuning without changing the requested frequency coordinates.
        """
        # 1. Prepare pump settings and preserve randomized acquisition order.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        jpa_freqs = sweep2array(cfg.jpa_freq, allow_array=True)
        np.random.default_rng(cfg.seed).shuffle(jpa_freqs[1:-1])

        # Show independent points because neighboring samples need not be nearby.
        fig, ax = context.plots.subplots(num="measurement")
        title = "JPA sweep · pump frequency · g/e separation score"
        ax.set(
            title=title,
            xlabel="JPA pump RF frequency (MHz)",
            ylabel="g/e separation score (a.u.)",
        )
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        line = live_line(ax)
        ax.lines[0].set(linestyle="None", marker="o", markersize=4, color="C0")
        retained_shots = np.zeros(len(jpa_freqs), dtype=np.int64)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]) -> Iterable[Artist]:
            """Draw acquired SNR on the measurement axes; return changed artists.

            Read the scan-shaped data without changing it. Update the live
            artists over the prepared coordinates; plot errors propagate.
            """
            progress = ax.set_title(
                f"{title}\n{np.count_nonzero(np.isfinite(data))}/{len(jpa_freqs)} settings sampled"
                f" · {retained_shots.sum():,} retained shots"
            )
            return (*line.set(jpa_freqs, np.abs(data)), progress)

        fig.show()
        signals_buffer = SignalBuffer(
            (len(jpa_freqs),),
            dtype=np.float64,
            on_update=update,
        )

        # 2. Tune the external pump once per host-side scan point.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for index, (jpa_freq, step) in enumerate(
                sched.scan("Pump RF frequency", jpa_freqs.tolist())
            ):
                if cfg.jpa_rf_dev is not None:
                    set_rf_frequency(context.devices[cfg.jpa_rf_dev], jpa_freq * 1e6)

                # 3. Compare reset-only and pi-prepared readout in one GE sweep.
                builder = TimelineBuilder(
                    soccfg,
                    reps=cfg.reps,
                    relax_delay=cfg.relax_delay,
                )
                builder.add_loop("ge", 2)
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.delay_by_timeline()
                # Empty g case has no pi-length padding; preserve each case's readout time.
                with (
                    builder.branch(by="ge", align_timeline=False) as cases,
                    cases.case(1),
                ):
                    builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
                builder.delay_by_timeline()
                builder.emit(Readout("readout", cfg.readout))
                prog = builder.build()

                def score(
                    sums: NDArray[np.float64], *, point: int = index
                ) -> NDArray[np.float64]:
                    """Track cumulative retained shots from both preparations and return SNR."""
                    retained_shots[point] = int(sums[..., 0].sum())
                    return snr_as_signal(sums, skew_penalty=cfg.skew_penalty)

                step.run_acquire_summed(
                    prog,
                    soc,
                    reducer=shot_sums(prog, [IQ("readout", order=3)]),
                    options=RunOptions(converter=score),
                    rounds=cfg.rounds,
                )

        # 4. Keep SNR paired with the actual acquisition-order coordinates.
        signals = signals_buffer.array

        return FreqResult(freqs=jpa_freqs, signals=signals)

    def analyze(
        self, source: RunRecord[FreqCfg, FreqResult], options: None, *, plots: Plots
    ) -> FreqAnalysis:
        """Plot measured SNR and return the sampled pump frequency with peak SNR.

        ``source`` supplies this run's frequencies and signals; no fit or
        interpolation is applied. ``options`` is None. All-NaN SNR fails.
        """
        # 1. Select the sampled pump frequency with peak SNR, ignoring missing points.
        del options
        result = source.result

        jpa_freqs = result.freqs
        signals = result.signals

        real_signals = np.abs(signals)

        max_idx = np.nanargmax(real_signals)
        best_jpa_freq = jpa_freqs[max_idx]

        # 2. Mark the best frequency without connecting randomized pump settings.
        _, ax = plots.subplots(num="fit", figsize=config.figsize)
        ax.set(
            title="JPA sweep · pump frequency · g/e separation score",
            xlabel="JPA pump RF frequency (MHz)",
            ylabel="g/e separation score (a.u.)",
        )
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.plot(
            jpa_freqs, real_signals, "o", color="C0", markersize=4, label="Measured"
        )
        ax.axvline(
            best_jpa_freq,
            color="C3",
            linestyle="--",
            label=f"Peak pump RF frequency = {best_jpa_freq:.3f} MHz",
        )
        ax.margins(y=0.25)
        ax.legend(loc="upper center")
        return FreqAnalysis(best_freq=float(best_jpa_freq))
