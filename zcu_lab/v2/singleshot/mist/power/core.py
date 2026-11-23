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
class PowerResult:
    """Rounded gains in a.u. and class counts (Ngain, 3).

    The final state axis is [g,e,other], with fixed coordinates [0,1,2].
    Counts sum over shots and completed rounds, without normalization.
    Unmeasured points remain NaN after cancellation.
    """

    gains: NDArray[np.float64]
    counts: NDArray[np.float64]


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class PowerAnalyzeOptions:
    ac_coeff: float | None = None
    log_scale: bool = False
    confusion_matrix: NDArray[np.float64] | None = None


class PowerCfg(ConfigBase):
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
    gain: Annotated[
        SweepCfg,
        FieldMeta(
            label="MIST drive gain",
            unit="a.u.",
            description="Set the MIST tone gain sweep in a.u. while holding its RF frequency fixed.",
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


class PowerExp(Experiment[PowerResult, PowerCfg]):
    Options: ClassVar[type[PowerAnalyzeOptions]] = PowerAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[PowerResult, PowerCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                values=POPULATION_STATES,
                label="Population",
                unit="None",
                dtype=np.int64,
            ),
            AxesSpec.Axis("gains", label="Drive gain", unit="a.u.", dtype=np.float64),
        ),
        z=AxesSpec.Z("counts", label="Counts", unit="shots", dtype=np.float64),
        result_type=PowerResult,
        cfg_type=PowerCfg,
        data_variable=DataVariable("counts"),
        cfg_schema_version="4.0",
        tag="singleshot/mist/power",
    )

    def run(self, cfg: PowerCfg, *, context: RunContext) -> PowerResult:
        """Measure MIST populations versus probe gain.

        Apply optional reset and init, sweep probe gain on the board, then
        read out. Return rounded gains (a.u.) and classified G/E/Other counts
        shaped (Ngain, 3), summed over shots and completed rounds. Gain is not
        calibrated power; analysis can convert it to photons with ac_coeff.

        cfg.readout supplies a circle classifier per ADC with a positive radius.
        Other counts unclassified shots, not calibrated leakage.
        Confusion correction and photon/log plotting are analysis choices.
        """

        # 1. Prepare the hardware gain grid before collecting populations.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        # 2. Keep the G/E/Other contract for every declared ADC.
        require_circle_classifiers(cfg.readout)
        read_axis = Classify(
            "readout",
        )

        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.probe_pulse.ch},
        )

        # 3. Show the measured G/E/Other fractions in classifier order.

        figure, ax = context.plots.subplots(num="measurement", layout="constrained")
        figure.suptitle("MIST gain scan · qubit · classified populations")
        ax.set(xlabel="MIST drive gain (a.u.)", ylabel="Outcome probability")
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
        figure.legend(loc="outside lower center", ncols=3)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]):
            """Display class fractions and their shot SEM."""
            totals = data.sum(axis=-1)
            covariance = class_probability_covariance(data)
            sem = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
            return viewer.set(gains, (data / totals[..., None]).T, sem.T)

        figure.show()
        buffer = SignalBuffer(
            (len(gains), POPULATION_COUNT),
            dtype=np.float64,
            on_update=update,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            gain_sweep = cfg.gain

            # Repeat reset/preparation/probe/readout at each hardware sweep point.
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            gain = builder.add_sweep("gain", gain_sweep)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(Pulse("probe_pulse", cfg.probe_pulse, gain=gain))
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

        return PowerResult(gains=gains, counts=counts)

    def analyze(
        self,
        source: RunRecord[PowerCfg, PowerResult],
        options: PowerAnalyzeOptions,
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
        gains, populations = result.gains, result.counts
        populations = populations / populations.sum(axis=-1, keepdims=True)
        if populations.shape[-1] != POPULATION_COUNT:
            raise ValueError(
                "Expected [g,e,other] populations; legacy G/E files are unsupported. "
                "Acquire a new run."
            )

        classified = populations
        populations = correct_populations(
            classified, options.confusion_matrix, clip=False, fallback_on_nonfinite=True
        )
        confusion_matrix = (
            None if populations is classified else options.confusion_matrix
        )
        sem = corrected_population_sem(result.counts, confusion_matrix)

        # 2. Use gain coordinates unless a photon-number calibration is supplied.
        if options.ac_coeff is None:
            xs = gains
            xlabel = "MIST drive gain (a.u.)"
        else:
            xs = options.ac_coeff * gains**2
            xlabel = "Mean cavity photon number (count)"

        # 3. Show probe-induced populations with optional logarithmic gain/photon axis.
        _, ax = plots.subplots(
            num="fit",
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
                xs,
                populations[:, index],
                yerr=sem[:, index],
                fmt="o-",
                markersize=3,
                capsize=2,
                label=label,
                color=color,
            )
        quantity = (
            "classified populations"
            if confusion_matrix is None
            else "readout-corrected populations"
        )
        ax.set(
            title=f"MIST gain scan · qubit · {quantity}",
            xlabel=xlabel,
            ylabel="Outcome probability",
        )
        ax.legend(loc="best", title="Data ± shot SEM")
        if options.log_scale:
            ax.set_xscale("log")
