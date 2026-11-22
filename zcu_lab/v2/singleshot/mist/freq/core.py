from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.datafile import DataVariable
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    PulseReadout,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules.readout import (
    require_circle_classifiers,
)

from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.support.singleshot.population_analysis import (
    CLASSIFIED_FIGURE,
    UNAVAILABLE_NOTICE,
    circle_readout_series,
)

POPULATION_STATES = (0, 1, 2)
POPULATION_COUNT = len(POPULATION_STATES)


@dataclass(frozen=True)
class FreqResult:
    """Rounded freqs in MHz and class counts (Nfreq, 3).

    The final state axis is [g,e,other], with fixed coordinates [0,1,2].
    Counts sum over shots and completed rounds, without normalization.
    Unmeasured points remain NaN after cancellation.
    """

    freqs: NDArray[np.float64]
    counts: NDArray[np.float64]


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class FreqAnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None


class FreqCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select the reset module applied before preparation, or leave empty to skip reset.",
        ),
    ] = None
    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Initial preparation pulse",
            description="Select the pulse applied before the probe, or leave empty to omit it.",
        ),
    ] = None
    probe_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="MIST tone",
            description="Enter the cavity drive applied after preparation and before the classification readout.",
        ),
    ]
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Enter the readout module applied at the end of the shot.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="MIST drive RF frequency",
            unit="MHz",
            description="Set the MIST tone RF sweep in MHz while holding its gain fixed.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps", unit="count", description="Number of repetitions per round."
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
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]


class FreqDepExp(Experiment[FreqResult, FreqCfg]):
    Options: ClassVar[type[FreqAnalyzeOptions]] = FreqAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[FreqResult, FreqCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                values=POPULATION_STATES,
                label="Population",
                unit="None",
                dtype=np.int64,
            ),
            AxesSpec.Axis(
                "freqs", label="Drive Freq", unit="Hz", scale=1e6, dtype=np.float64
            ),
        ),
        z=AxesSpec.Z("counts", label="Counts", unit="shots", dtype=np.float64),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        data_variable=DataVariable("counts"),
        cfg_schema_version="4.0",
        tag="singleshot/mist/freq",
    )

    def run(self, cfg: FreqCfg, *, context: RunContext) -> FreqResult:
        """Measure MIST populations versus probe frequency.

        Apply optional reset and init, sweep the probe frequency on the board,
        then read out. Return rounded frequencies (MHz) and classified G/E/Other
        counts shaped (Nfreq, 3), summed over shots and completed rounds.

        cfg.readout supplies a circle classifier per ADC with a positive radius.
        Other counts unclassified shots, not calibrated leakage;
        optional confusion correction belongs to subsequent analysis.
        """

        # 1. Prepare the hardware frequency grid before collecting populations.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        # 2. Keep the G/E/Other contract for every declared ADC.
        require_circle_classifiers(cfg.readout)
        read_axis = Classify(
            "readout",
        )

        freqs = sweep2array(
            cfg.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": cfg.probe_pulse.ch},
        )

        # 3. Show the measured G/E/Other fractions in classifier order.

        figure, ax = context.plots.subplots(num="measurement", layout="constrained")
        progress = figure.suptitle(
            "MIST probe spectrum · qubit · classified populations"
        )
        ax.set(xlabel="MIST drive RF frequency (MHz)", ylabel="Outcome probability")
        viewer = live_errorbar(ax, num_lines=POPULATION_COUNT)
        for line, bars, label, color in zip(
            ax.lines,
            ax.collections,
            ("P(g)", "P(e)", "P(other)"),
            ("tab:blue", "tab:red", "tab:green"),
            strict=True,
        ):
            line.set(label=label, color=color, marker="o", markersize=3)
            bars.set_color(color)
        ax.legend(loc="best")

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]):
            """Display acquired fractions, their shot SEM and scan progress."""
            totals = data.sum(axis=-1)
            progress.set_text(
                f"MIST probe spectrum · qubit · classified populations\n"
                f"{np.count_nonzero(totals > 0)}/{totals.size} points acquired"
            )
            covariance = class_probability_covariance(data)
            sem = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
            return (progress, *viewer.set(freqs, (data / totals[..., None]).T, sem.T))

        figure.show()
        buffer = SignalBuffer(
            (len(freqs), POPULATION_COUNT),
            dtype=np.float64,
            on_update=update,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            freq_sweep = cfg.freq

            # Repeat reset/preparation/probe/readout at each hardware sweep point.
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            freq = builder.add_sweep("freq", freq_sweep)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(Pulse("probe_pulse", cfg.probe_pulse, freq=freq))
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                soc,
                reducer=shot_sums(prog, [read_axis]),
                rounds=cfg.rounds,
            )

        # 4. Keep the unnormalized G/E/Other counts in the saved Result.
        counts = buffer.array

        return FreqResult(freqs=freqs, counts=counts)

    def analyze(
        self,
        source: RunRecord[FreqCfg, FreqResult],
        options: FreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot the run's [g,e,other] fractions with optional confusion correction.

        source supplies the saved sweep grid and three-column counts;
        normalize each point before confusion correction.
        cfg may be None. options controls correction and any axis conversion.
        Publish the fit figure to plots. Legacy G/E data raise ValueError.
        """
        # 1. Normalize G/E/Other counts and apply optional confusion correction.
        result = source.result
        freqs, populations = result.freqs, result.counts
        populations = populations / populations.sum(axis=-1, keepdims=True)
        if populations.shape[-1] != POPULATION_COUNT:
            raise ValueError(
                "Expected [g,e,other] populations; legacy G/E files are unsupported. "
                "Acquire a new run."
            )

        series = circle_readout_series(
            populations, result.counts, options.confusion_matrix
        )

        # 2. Show all three populations versus probe frequency, without a spectral fit.
        _, ax = plots.subplots(
            num=CLASSIFIED_FIGURE if series.correction_unavailable else "fit",
            figsize=(7, 5),
            layout="constrained",
        )
        for index, (label, color) in enumerate(
            zip(
                ("P(g)", "P(e)", "P(other)"),
                ("tab:blue", "tab:red", "tab:green"),
                strict=True,
            )
        ):
            ax.errorbar(
                freqs,
                series.values[:, index],
                yerr=series.sem[:, index],
                fmt="o-",
                markersize=3,
                label=label,
                color=color,
            )
        quantity = (
            "classified populations"
            if options.confusion_matrix is None
            else "readout-corrected populations"
        )
        if series.correction_unavailable:
            quantity = f"classified fractions\n{UNAVAILABLE_NOTICE}"
        ax.set(
            title=f"MIST probe spectrum · qubit · {quantity}",
            xlabel="MIST drive RF frequency (MHz)",
            ylabel="Classified fraction"
            if series.correction_unavailable
            else "Outcome probability",
        )
        ax.legend(loc="best")
