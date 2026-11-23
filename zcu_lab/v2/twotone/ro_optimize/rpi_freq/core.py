"""Readout frequency sweep of an ...R π R π... chain with a calibrated π.

Each rep is a 64-read cycle of readout R and fixed π pulses, with no reset or
relax delay between reps. Lags stay within each cycle.
readout_chain.rpi_analysis gives every frequency an alternating-contrast score
and its invalid reasons. The best frequency is the valid frequency with the
largest score.
"""

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.readout_chain import RPiAnalysis
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    PulseReadout,
    SweepCfg,
    TimelineBuilder,
    rpi_sums,
)

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.ro_optimize._rpi_common import (
    READS_PER_REP,
    RECORD_COLUMNS,
    SCORE_LABEL,
    SweepAxis,
    analyze_records,
    best_index,
    fix_sweep_xlim,
    invalid_reason_counts,
    not_plotted_title,
    records_variable,
    valid_values,
)

FREQ_AXIS = SweepAxis(
    setting_name="frequency", label="Readout frequency (MHz)", unit="MHz", digits=7
)


@dataclass(frozen=True)
class RPiFreqResult:
    """Pooled R-pi sums of every readout frequency.

    freqs: Readout frequency (F,) in MHz, in sweep order, absolute RF when the
        readout sets mixer_freq.
    records: Float64 (F, 35), one additive rpi_sums record per frequency, summed
        over finished rounds. Read it only through RPiSums.from_processor.
    """

    freqs: NDArray[np.float64]
    records: NDArray[np.float64]


@dataclass(frozen=True)
class RPiFreqAnalysis:
    """Best frequency of an R-pi sweep. No writeback is proposed.

    The size does not grow with the scan. The "score" figure shows which
    frequency has which invalid reason, and the saved records reproduce every
    frequency.

    best_freq: Valid frequency with the largest score in MHz; None when no
        frequency is valid. There is no fallback frequency.
    best: rpi_analysis result of best_freq; None when no frequency is valid.
    freq_count: Number of frequencies in the sweep.
    not_plotted: Number of invalid frequencies, left off every curve and never
        ranked.
    invalid_reasons: Number of frequencies reporting each invalid reason. One
        frequency can report several reasons. Empty when every one is valid.
    """

    best_freq: float | None
    best: RPiAnalysis | None
    freq_count: int
    not_plotted: int
    invalid_reasons: dict[str, int]


class RPiFreqCfg(ConfigBase):
    """Readout and pi modules, the readout frequency sweep and the averaging counts.

    readout: Readout module. Run replaces its generator and ADC frequency with
        the sweep. The wait between the readout generator end and the pi is its
        post_delay plus the pre_delay of pi_pulse, both in us.
    pi_pulse: Calibrated pi pulse, the same at every frequency. Run keeps all of
        its settings; its pre_delay in us is usually 0.
    freq: Readout frequency sweep in MHz, absolute RF when readout.mixer_freq is
        set and the DDS frequency otherwise.
    reps: Cycles of 64 R-pi per frequency in each round, at least 1.
    rounds: Software rounds, at least 1.
    """

    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout pulse and ADC window",
            description="Select the resonator pulse and ADC timing for the read before every π pulse.",
        ),
    ]
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit π pulse",
            description="Select the calibrated g-to-e π pulse applied after each readout.",
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
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Cycles of 64 R-π pairs in each round.",
        ),
    ] = Field(ge=1)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Software rounds, each followed by a live update.",
        ),
    ] = Field(default=1, ge=1)


def restore_result(data: GroupedLoadData[Any]) -> RPiFreqResult:
    """Rebuild the frequencies and records from the saved "records" variable."""
    records = data.variable("records")
    return RPiFreqResult(freqs=records.axes[1], records=records.z)


class RPiFreqExp(Experiment[RPiFreqResult, RPiFreqCfg]):
    """Measure an R-pi chain at each readout frequency and score each frequency."""

    # freqs are MHz in memory and Hz on disk; records keep the rpi_sums layout.
    AXES_SPEC: ClassVar[GroupedAxesSpec[RPiFreqResult, RPiFreqCfg]] = GroupedAxesSpec(
        variables=(
            records_variable(
                VariableAxisSpec(
                    label="Readout frequency",
                    unit="Hz",
                    field_name="freqs",
                    scale=1e6,
                )
            ),
        ),
        result_type=RPiFreqResult,
        cfg_type=RPiFreqCfg,
        tag="twotone/ro_optimize/rpi_freq",
        cfg_schema_version="3.0",
        result_builder=restore_result,
    )

    def run(self, cfg: RPiFreqCfg, *, context: RunContext) -> RPiFreqResult:
        """Read an R-pi chain at each readout frequency and keep its sums.

        One rep is one cycle of 64 pairs of cfg.readout at the swept frequency
        and cfg.pi_pulse. The sweep sets the generator and ADC frequency of the
        readout together. The pi frequency, gain and length and every other
        readout setting stay fixed. All cfg.reps reps of one frequency run
        before the next frequency, with no reset or relax delay. Rep, round
        and frequency boundaries never contribute lag pairs.

        Inside a rep, each pi starts cfg.readout.post_delay plus
        cfg.pi_pulse.pre_delay us after the readout generator pulse ends, even
        when the ADC window is still open. The pi pre_delay is usually 0, and
        Run keeps it as set. The next readout starts cfg.pi_pulse.post_delay
        plus cfg.readout.pre_delay us after the pi ends, both usually 0. At a
        rep boundary the shot closure first waits for the last ADC window to
        close, so that step can be longer and reads are not guaranteed to be
        evenly spaced across reps. This wait is not a relax delay and falls
        outside every lag. Lag 1..15 pairs stay within each cycle; their sums
        accumulate across reps and rounds.

        After each round the live plot shows the score of each valid frequency,
        from all finished rounds. Return the sums of all finished rounds;
        frequencies are in MHz, rounded to the common generator and ADC grid.
        With cfg.readout.mixer_freq set, cfg.freq and the returned frequencies
        are absolute RF. An interpolating generator raises RuntimeError for a
        frequency outside its DDS range; with a mixer that range applies to RF
        minus the rounded mixer frequency. A stop keeps finished rounds. A stop
        before the first round raises RuntimeError. ADC windows that would
        overlap raise ValueError when the program builds. The sums cannot reveal
        a physical break in the chain that does not appear in the IQ data.
        """
        # 1. Quantize readout frequencies on the common generator/ADC grid.
        freqs = sweep2array(
            cfg.freq,
            "freq",
            {
                "soccfg": context.soccfg,
                "gen_ch": cfg.readout.gen_ch,
                "ro_ch": cfg.readout.ro_ch,
                "mixer_freq": cfg.readout.mixer_freq,
            },
        )

        # 2. Track valid R-pi readout scores and flag unplottable frequency settings.
        fig, ax = context.plots.subplots(num="measurement")
        fig.suptitle("R–π frequency sweep · Resonator · Alternating contrast")
        ax.set(xlabel=FREQ_AXIS.label, ylabel=SCORE_LABEL)
        fix_sweep_xlim(ax, freqs)
        viewer = live_line(ax)
        ax.lines[0].set_marker("o")

        @context.plots.on_plot_thread
        def draw(analyses: list[RPiAnalysis]) -> tuple[Artist, ...]:
            """Refresh the measurement artists from the current acquired data."""
            status = ax.set_title(not_plotted_title(analyses, FREQ_AXIS.setting_name))
            return (*viewer.set(freqs, valid_values(analyses, "score")), status)

        def update(records: NDArray[np.float64]) -> None:
            # Analyze here, on the acquisition thread, so the GUI thread only draws.
            """Refresh the measurement artists from the current acquired data."""
            draw(analyze_records(records))

        fig.show()
        record_sums = SignalBuffer(
            (len(freqs), RECORD_COLUMNS), dtype=np.float64, on_update=update
        )

        # 3. Measure alternating readout-pi correlations at each frequency without relaxation waits.
        with Schedule(record_sums, scope=context.scope) as sched:
            # reps_innermost: all reps of one frequency run before the next one.
            # relax_delay 0: no added wait between reps. The shot closure can still
            # wait for the last ADC window, outside every lag pair.
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=0.0, reps_innermost=True
            )
            freq = builder.add_sweep("freq", cfg.freq)
            # PulseReadout applies freq to both its generator and its ADC.
            readout = PulseReadout("readout", cfg.readout, freq=freq)
            pi_pulse = Pulse("pi_pulse", cfg.pi_pulse)
            with builder.repeat("rpi", READS_PER_REP):
                builder.emit(readout)  # Ends at generator end plus post_delay.
                builder.emit(pi_pulse)
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                reducer=rpi_sums(prog, "readout"),
                rounds=cfg.rounds,
            )

        # 4. Keep the sums of all finished rounds; an unfinished buffer stays NaN.
        if np.isnan(record_sums.array).any():
            raise RuntimeError(
                "Stopped before the first round finished; no sums to keep"
            )
        return RPiFreqResult(freqs=freqs, records=record_sums.array)

    def analyze(
        self,
        source: RunRecord[RPiFreqCfg, RPiFreqResult],
        options: None,
        *,
        plots: Plots,
    ) -> RPiFreqAnalysis:
        """Pick the valid frequency with the largest score and plot every diagnostic.

        source.cfg may be None. Each frequency is analyzed from its own saved
        record. Publish the "score" figure: the score with the best frequency
        marked, the invalid reasons of each frequency, and Q_eff, q, amplitude
        and noise variance, all against the readout frequency in MHz. Invalid
        frequencies are left off every curve, are never ranked, and are counted
        in the figure title. When no frequency is valid there is no best
        frequency and no fallback: best_freq and best are None and no error is
        raised.
        """
        # 1. Rank readout frequencies using valid R-pi lag-correlation scores.
        del options  # This analysis has no configurable options.
        freqs = source.result.freqs
        analyses = analyze_records(source.result.records)
        best = best_index(analyses)

        # 2. Compare valid scores with the reasons other settings were rejected.
        fig, _ = plots.subplots(2, 1, num="score", sharex=True, figsize=(8, 6))
        ax_score, ax_reasons = fig.axes
        fig.suptitle("R–π frequency sweep · Resonator · Alternating contrast")
        scores = valid_values(analyses, "score")
        ax_score.plot(freqs, scores, "o-", color="C0")
        ax_score.set_ylabel(SCORE_LABEL)
        if best is not None:
            setting = float(freqs[best])
            score = scores[best]
            ax_score.axvline(setting, color="C3", ls="--")
            ax_score.set_title(
                f"Selected readout frequency = {setting:.{FREQ_AXIS.digits}g} {FREQ_AXIS.unit}\n"
                f"Alternating-contrast score = {score:.4g}"
            )
        else:
            ax_score.set_title("No valid frequency to select")

        reasons = {
            "gate_failed": "Contrast gate",
            "q_eff_out_of_range": "Qeff outside 0–1",
            "nonpositive_amplitude": "Amplitude ≤ 0",
            "nonpositive_noise_variance": "Noise variance ≤ 0",
            "zero_denominator": "Missing pair or zero divisor",
            "nonpositive_same_read_variance": "Same-read variance ≤ 0",
        }
        rows = sorted(invalid_reason_counts(analyses))
        for row, reason in enumerate(rows):
            rejected = [
                setting
                for setting, analysis in zip(freqs, analyses, strict=True)
                if reason in analysis.invalid_reasons
            ]
            ax_reasons.plot(rejected, [row] * len(rejected), "x", color="C3")
        ax_reasons.set_yticks(range(len(rows)), [reasons[reason] for reason in rows])
        ax_reasons.set_ylim(-0.5, max(len(rows), 1) - 0.5)
        ax_reasons.set(
            xlabel=FREQ_AXIS.label,
            title="Why settings were rejected" if rows else "No settings rejected",
        )

        # 3. Inspect the two-state model estimates at the valid settings.
        fig, _ = plots.subplots(2, 2, num="diagnostics", sharex=True, figsize=(8, 6))
        fig.suptitle("R–π frequency sweep · Resonator · Correlation diagnostics")
        for ax, name, ylabel in zip(
            fig.axes,
            ("q_eff", "q", "amplitude", "noise_variance"),
            (
                "Effective retention Qeff",
                "Amplitude/noise q",
                "Amplitude (a.u.²)",
                "Noise variance (a.u.²)",
            ),
            strict=True,
        ):
            ax.plot(freqs, valid_values(analyses, name), "o-", color="C0")
            ax.set_ylabel(ylabel)
            if best is not None:
                ax.axvline(float(freqs[best]), color="C3", ls="--")
        for ax in fig.axes[2:]:
            ax.set_xlabel(FREQ_AXIS.label)
        fig.tight_layout()

        return RPiFreqAnalysis(
            best_freq=None if best is None else float(freqs[best]),
            best=None if best is None else analyses[best],
            freq_count=len(analyses),
            not_plotted=sum(not analysis.valid for analysis in analyses),
            invalid_reasons=invalid_reason_counts(analyses),
        )
