from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.shot_uncertainty import (
    class_probability_covariance,
    corrected_population_sem,
)
from zcu_tools.analysis.singleshot import correct_populations
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

POPULATION_STATES = (0, 1, 2)
POPULATION_COUNT = len(POPULATION_STATES)


@dataclass(frozen=True)
class PreFreqResult:
    """Rounded preparation freqs in MHz and class counts (Nfreq, 3).

    The final state axis is [g,e,other], with fixed coordinates [0,1,2].
    Counts sum over shots and completed rounds, without normalization.
    Unmeasured points remain NaN after cancellation.
    """

    freqs: NDArray[np.float64]
    counts: NDArray[np.float64]


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class PreFreqAnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None


class PreFreqCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select the reset module applied before preparation, or leave empty to skip reset.",
        ),
    ] = None
    init_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Preparation pulse",
            description="Enter the qubit pulse with the RF frequency swept before the MIST tone.",
        ),
    ]
    pi_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Optional π pulse",
            description="Select a calibrated g↔e π pulse after preparation, or leave empty to skip it.",
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
            label="Preparation RF frequency",
            unit="MHz",
            description="Set the preparation pulse RF sweep in MHz before the fixed MIST tone.",
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


class PreFreqExp(Experiment[PreFreqResult, PreFreqCfg]):
    Options: ClassVar[type[PreFreqAnalyzeOptions]] = PreFreqAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[PreFreqResult, PreFreqCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                values=POPULATION_STATES,
                label="Population",
                unit="None",
                dtype=np.int64,
            ),
            AxesSpec.Axis(
                "freqs",
                label="PrePulse frequency",
                unit="Hz",
                scale=1e6,
                dtype=np.float64,
            ),
        ),
        z=AxesSpec.Z("counts", label="Counts", unit="shots", dtype=np.float64),
        result_type=PreFreqResult,
        cfg_type=PreFreqCfg,
        data_variable=DataVariable("counts"),
        cfg_schema_version="4.0",
        tag="singleshot/mist/pre_freq",
    )

    def run(self, cfg: PreFreqCfg, *, context: RunContext) -> PreFreqResult:
        """Measure MIST populations versus the preparation-pulse frequency.

        After optional reset, sweep init_pulse frequency on the board, apply
        the optional pi pulse, then the fixed probe pulse and readout.
        Return rounded preparation frequencies (MHz) and classified G/E/Other
        counts shaped (Nfreq, 3), summed over shots and completed rounds.

        cfg.readout supplies a circle classifier per ADC with a positive radius.
        Other counts unclassified shots, not calibrated leakage.
        Optional confusion correction belongs to subsequent analysis.
        """

        # 1. Sweep the preparation tone, not the later MIST probe.
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
            {"soccfg": soccfg, "gen_ch": cfg.init_pulse.ch},
        )

        # 3. Show the measured G/E/Other fractions in classifier order.

        figure, ax = context.plots.subplots(num="measurement", layout="constrained")
        progress = figure.suptitle(
            "MIST preparation spectrum · qubit · classified populations"
        )
        ax.set(xlabel="Preparation RF frequency (MHz)", ylabel="Outcome probability")
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
                f"MIST preparation spectrum · qubit · classified populations\n"
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

            # Vary initialization before the optional pi pulse and fixed probe.
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            freq = builder.add_sweep("freq", freq_sweep)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("init_pulse", cfg.init_pulse, freq=freq))
            builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.emit(Pulse("probe_pulse", cfg.probe_pulse))
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

        return PreFreqResult(freqs=freqs, counts=counts)

    def analyze(
        self,
        source: RunRecord[PreFreqCfg, PreFreqResult],
        options: PreFreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot G/E/Other populations versus preparation frequency in MHz.

        Normalize the explicit run's G/E/Other counts and apply the optional
        confusion matrix. Publish a named fit figure without estimating scalars.
        Three-column data can be analyzed without cfg. Legacy G/E data raise
        ValueError.
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

        populations = correct_populations(
            populations, options.confusion_matrix, clip=False
        )
        sem = corrected_population_sem(result.counts, options.confusion_matrix)

        # 2. Isolate preparation-frequency dependence with the later probe held fixed.
        _, ax = plots.subplots(num="fit", figsize=(7, 5), layout="constrained")
        for index, (label, color) in enumerate(
            zip(
                ("P(g)", "P(e)", "P(other)"),
                ("tab:blue", "tab:red", "tab:green"),
                strict=True,
            )
        ):
            ax.errorbar(
                freqs,
                populations[:, index],
                yerr=sem[:, index],
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
        ax.set(
            title=f"MIST preparation spectrum · qubit · {quantity}",
            xlabel="Preparation RF frequency (MHz)",
            ylabel="Outcome probability",
        )
        ax.legend(loc="best")
