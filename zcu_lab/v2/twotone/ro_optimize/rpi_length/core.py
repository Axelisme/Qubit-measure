"""Readout length sweep of an ...R π R π... chain with a calibrated π.

Each rep is a 64-read cycle of readout R and fixed π pulses, with no reset or
relax delay between reps. Lags stay within each cycle.
readout_chain.rpi_analysis gives every window an alternating-contrast score
and its invalid reasons. The best window is the valid window with the
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

# The readout pulse is the ADC window plus this margin, in us, at every setting.
PULSE_MARGIN_US = 0.11

LENGTH_AXIS = SweepAxis(
    setting_name="window", label="ADC window (us)", unit="us", digits=5
)


@dataclass(frozen=True)
class RPiLengthResult:
    """Pooled R-pi sums of every ADC window with at least one finished round.

    lengths: ADC window (F,) in us, in sweep order. A stopped run keeps only the
        windows with a finished round, so F can be smaller than the sweep.
    records: Float64 (F, 35), one additive rpi_sums record per window, summed
        over finished rounds. Read it only through RPiSums.from_processor.
    """

    lengths: NDArray[np.float64]
    records: NDArray[np.float64]


@dataclass(frozen=True)
class RPiLengthAnalysis:
    """Best ADC window of an R-pi sweep. No writeback is proposed.

    The size does not grow with the scan. The "score" figure shows which window
    has which invalid reason, and the saved records reproduce every window.

    best_length: Valid ADC window with the largest score in us; None when no
        window is valid. There is no fallback window.
    best: rpi_analysis result of best_length; None when no window is valid.
    length_count: Number of windows in the saved result.
    not_plotted: Number of invalid windows, left off every curve and never ranked.
    invalid_reasons: Number of windows reporting each invalid reason. One window
        can report several reasons. Empty when every window is valid.
    """

    best_length: float | None
    best: RPiAnalysis | None
    length_count: int
    not_plotted: int
    invalid_reasons: dict[str, int]


class RPiLengthCfg(ConfigBase):
    """Readout and pi modules, the ADC window sweep and the averaging counts.

    readout: Readout module. Run replaces its ADC window with the sweep and its
        pulse length with that window plus 0.11 us. The wait between the readout
        generator end and the pi is its post_delay plus the pre_delay of
        pi_pulse, both in us.
    pi_pulse: Calibrated pi pulse, the same at every window. Run keeps all of its
        settings; its pre_delay in us is usually 0.
    length: ADC window sweep in us.
    reps: Cycles of 64 R-pi per window in each round, at least 1.
    rounds: Software rounds, at least 1.
    """

    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Pi pulse",
            description="Calibrated g-to-e pi pulse after every readout in the R-pi cycle.",
        ),
    ]
    length: Annotated[
        SweepCfg,
        FieldMeta(
            label="ADC window",
            unit="us",
            description="ADC window sweep.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            description="Cycles of 64 R-π pairs in each round.",
        ),
    ] = Field(ge=1)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            description="Software rounds, each followed by a live update.",
        ),
    ] = Field(default=1, ge=1)


def restore_result(data: GroupedLoadData[Any]) -> RPiLengthResult:
    """Rebuild the windows and records from the saved "records" variable."""
    records = data.variable("records")
    return RPiLengthResult(lengths=records.axes[1], records=records.z)


def _finished(records: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Mark the rows of (F, 35) records with a finished round; others stay NaN."""
    return np.logical_not(np.isnan(records).any(axis=1))


class RPiLengthExp(Experiment[RPiLengthResult, RPiLengthCfg]):
    """Measure an R-pi chain at each ADC window and score each window."""

    # lengths are us in memory and s on disk; records keep the rpi_sums layout.
    AXES_SPEC: ClassVar[GroupedAxesSpec[RPiLengthResult, RPiLengthCfg]] = (
        GroupedAxesSpec(
            variables=(
                records_variable(
                    VariableAxisSpec(
                        label="ADC window",
                        unit="s",
                        field_name="lengths",
                        scale=1e-6,
                    )
                ),
            ),
            result_type=RPiLengthResult,
            cfg_type=RPiLengthCfg,
            tag="twotone/ro_optimize/rpi_length",
            cfg_schema_version="3.0",
            result_builder=restore_result,
        )
    )

    def run(self, cfg: RPiLengthCfg, *, context: RunContext) -> RPiLengthResult:
        """Read an R-pi chain at each ADC window and keep its sums.

        The host builds one program per window. Each program sets the ADC window
        to the swept value, rounded to the ADC clock, and the readout pulse to
        that window plus 0.11 us. The pi pulse and every other readout setting,
        including post_delay, stay fixed. One rep is one cycle of 64 pairs of
        the readout and cfg.pi_pulse. All cfg.reps reps of one window run before
        the next window, with no reset or relax delay. Rep, round and window boundaries
        never contribute lag pairs.

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

        After each round the live plot shows the score of each valid window with
        a finished round, from all finished rounds. Return the sums of the
        windows with a finished round; windows are in us. A stop keeps those
        windows and drops the others, which were not measured and are not
        invalid. A stop before any window finishes a round raises RuntimeError.
        The sums cannot reveal a physical break in the chain that does not
        appear in the IQ data.
        """
        # 1. Quantize readout integration windows on the ADC clock.
        lengths = sweep2array(
            cfg.length, "time", {"soccfg": context.soccfg, "ro_ch": cfg.readout.ro_ch}
        )

        # 2. Track valid R-pi scores from completed windows and flag rejected settings.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel=LENGTH_AXIS.label, ylabel=SCORE_LABEL)
        ax.grid(True)
        fix_sweep_xlim(ax, lengths)
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def draw(
            payload: tuple[NDArray[np.float64], list[RPiAnalysis]],
        ) -> tuple[Artist, ...]:
            """Refresh the measurement artists from the current acquired data."""
            windows, analyses = payload
            ax.set_title(not_plotted_title(analyses, LENGTH_AXIS.setting_name))
            return (*viewer.set(windows, valid_values(analyses, "score")), ax.title)

        def update(records: NDArray[np.float64]) -> None:
            # Analyze here, on the acquisition thread, so the GUI thread only draws.
            """Refresh the measurement artists from the current acquired data."""
            finished = _finished(records)
            draw((lengths[finished], analyze_records(records[finished])))

        fig.show()
        record_sums = SignalBuffer(
            (len(lengths), RECORD_COLUMNS), dtype=np.float64, on_update=update
        )

        # 3. Measure alternating readout-pi correlations with a pulse matched to each ADC window.
        with Schedule(record_sums, scope=context.scope) as sched:
            for _, step in sched.scan("length", lengths.tolist()):
                # reps_innermost: all reps of this window run together.
                # relax_delay 0: no added wait between reps. The shot closure can
                # still wait for the last ADC window, outside every lag pair.
                builder = TimelineBuilder(
                    context.soccfg, reps=cfg.reps, relax_delay=0.0, reps_innermost=True
                )
                # This window's pulse, not the longest pulse of the scan.
                readout = PulseReadout(
                    "readout",
                    cfg.readout,
                    ro_length=step.value,
                    length=step.value + PULSE_MARGIN_US,
                )
                pi_pulse = Pulse("pi_pulse", cfg.pi_pulse)
                with builder.repeat("rpi", READS_PER_REP):
                    builder.emit(readout)  # Ends at generator end plus post_delay.
                    builder.emit(pi_pulse)
                prog = builder.build()
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    processor=rpi_sums(prog, "readout"),
                    rounds=cfg.rounds,
                )

        # 4. Keep the windows with a finished round; the others were not measured.
        finished = _finished(record_sums.array)
        if not finished.any():
            raise RuntimeError(
                "Stopped before any window finished a round; no sums to keep"
            )
        return RPiLengthResult(
            lengths=lengths[finished], records=record_sums.array[finished]
        )

    def analyze(
        self,
        source: RunRecord[RPiLengthCfg, RPiLengthResult],
        options: None,
        *,
        plots: Plots,
    ) -> RPiLengthAnalysis:
        """Pick the valid ADC window with the largest score and plot every diagnostic.

        source.cfg may be None. Each window is analyzed from its own saved
        record. Publish the "score" figure: the score with the best window
        marked, the invalid reasons of each window, and Q_eff, q, amplitude and
        noise variance, all against the ADC window in us. Invalid windows are
        left off every curve, are never ranked, and are counted in the figure
        title. When no window is valid there is no best window and no fallback:
        best_length and best are None and no error is raised.
        """
        # 1. Rank completed readout windows using valid R-pi lag-correlation scores.
        del options  # This analysis has no configurable options.
        lengths = source.result.lengths
        analyses = analyze_records(source.result.records)
        best = best_index(analyses)

        # 2. Diagnose window-dependent score through effective Q, correlation and noise.
        fig, _ = plots.subplots(3, 2, num="score", sharex=True, figsize=(11, 8))
        ax_score, ax_reasons, ax_q_eff, ax_q, ax_amplitude, ax_noise = fig.axes

        ax_score.plot(lengths, valid_values(analyses, "score"), "o-", color="C0")
        ax_score.set_ylabel(SCORE_LABEL)
        for ax, name, ylabel in (
            (ax_q_eff, "q_eff", "Q_eff (includes π error)"),
            (ax_q, "q", "q = a / σ²"),
            (ax_amplitude, "amplitude", "Amplitude a (ADC units²)"),
            (ax_noise, "noise_variance", "Noise variance σ² (ADC units²)"),
        ):
            ax.plot(lengths, valid_values(analyses, name), "o-")
            ax.set_ylabel(ylabel)

        # 3. Explain rejected windows alongside the best valid integration duration.
        rows = sorted(invalid_reason_counts(analyses))
        for row, reason in enumerate(rows):
            rejected = [
                setting
                for setting, analysis in zip(lengths, analyses, strict=True)
                if reason in analysis.invalid_reasons
            ]
            ax_reasons.plot(rejected, [row] * len(rejected), "x", color="C3")
        ax_reasons.set_yticks(range(len(rows)), rows)
        ax_reasons.set_ylim(-0.5, max(len(rows), 1) - 0.5)
        ax_reasons.set_title(f"Invalid reason per {LENGTH_AXIS.setting_name}")

        for ax in fig.axes:
            ax.grid(True)
            if best is not None:
                ax.axvline(float(lengths[best]), color="r", ls="--")
        if best is not None:
            value = f"{float(lengths[best]):.{LENGTH_AXIS.digits}g}"
            ax_score.legend(
                [ax_score.lines[-1]], [f"largest score = {value} {LENGTH_AXIS.unit}"]
            )
        ax_amplitude.set_xlabel(LENGTH_AXIS.label)
        ax_noise.set_xlabel(LENGTH_AXIS.label)
        title = not_plotted_title(analyses, LENGTH_AXIS.setting_name)
        if title:
            fig.suptitle(title)
        fig.tight_layout()

        return RPiLengthAnalysis(
            best_length=None if best is None else float(lengths[best]),
            best=None if best is None else analyses[best],
            length_count=len(analyses),
            not_plotted=sum(not analysis.valid for analysis in analyses),
            invalid_reasons=invalid_reason_counts(analyses),
        )
