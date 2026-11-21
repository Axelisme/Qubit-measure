from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass
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

from zcu_lab.v2.support import live_line

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
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Preparation pulse")]
    pi_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Pi pulse")] = None
    probe_pulse: Annotated[Pulse.Cfg, FieldMeta(label="MIST probe pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    freq: Annotated[SweepCfg, FieldMeta(label="Preparation freq", unit="MHz")]
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

        figure, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Preparation pulse frequency (MHz)", ylabel="State probability")
        viewer = live_line(ax, num_lines=POPULATION_COUNT)
        ax.set_ylim(0.0, 1.0)
        for line, label in zip(ax.lines, ("Ground", "Excited", "Other"), strict=True):
            line.set_label(label)
        ax.legend()

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]):
            """Draw acquired shot counts on the measurement axes; return changed artists.

            Read scan-shaped samples without changing them. Update the live
            artists over the prepared coordinates; plot errors propagate.
            """
            return viewer.set(freqs, (data / data.sum(axis=-1, keepdims=True)).T)

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

        populations = correct_populations(populations, options.confusion_matrix)

        # 2. Isolate preparation-frequency dependence with the later probe held fixed.
        _, ax = plots.subplots(num="fit", figsize=(6, 6))

        ax.plot(
            freqs,
            populations[:, 0],
            color="blue",
            label="Ground",
            ls="-",
            marker="o",
            markersize=1,
        )
        ax.plot(
            freqs,
            populations[:, 1],
            color="red",
            label="Excited",
            ls="-",
            marker="o",
            markersize=1,
        )
        ax.plot(
            freqs,
            populations[:, 2],
            color="green",
            label="Other",
            ls="-",
            marker="o",
            markersize=1,
        )
        ax.set_xlabel("Frequency (MHz)", fontsize=14)
        ax.set_ylabel("State probability", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_ylim(0, 1)
