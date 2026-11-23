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

from zcu_lab.v2.support import live_heatmap

POPULATION_STATES = (0, 1, 2)
POPULATION_COUNT = len(POPULATION_STATES)


@dataclass(frozen=True)
class FreqPowerResult:
    """Measured class counts (Ngain, Nfreq, 3) on rounded gain/frequency grids.

    Gains are in a.u.; frequencies are in MHz.

    The final state axis is [g,e,other], with fixed coordinates [0,1,2].
    Counts sum over shots and completed rounds, without normalization.
    Unmeasured points remain NaN after cancellation.
    """

    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    counts: NDArray[np.float64]


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class FreqPowerAnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None


class FreqPowerCfg(ConfigBase):
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
            description="Set the MIST tone gain sweep in a.u.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="MIST drive RF frequency",
            unit="MHz",
            description="Set the MIST tone RF sweep in MHz.",
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


class FreqPowerExp(Experiment[FreqPowerResult, FreqPowerCfg]):
    Options: ClassVar[type[FreqPowerAnalyzeOptions]] = FreqPowerAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[FreqPowerResult, FreqPowerCfg]] = AxesSpec(
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
            AxesSpec.Axis("gains", label="Drive gain", unit="a.u.", dtype=np.float64),
        ),
        z=AxesSpec.Z("counts", label="Counts", unit="shots", dtype=np.float64),
        result_type=FreqPowerResult,
        cfg_type=FreqPowerCfg,
        data_variable=DataVariable("counts"),
        cfg_schema_version="4.0",
        tag="singleshot/mist/power_freq",
    )

    def run(
        self,
        cfg: FreqPowerCfg,
        *,
        context: RunContext,
    ) -> FreqPowerResult:
        """Map MIST populations versus probe gain and frequency.

        At each host-side probe gain, apply optional reset and init, play the
        frequency-swept probe, then read out. Return rounded gains (a.u.),
        frequencies (MHz), and classified G/E/Other counts shaped (Ngain, Nfreq, 3),
        summed over shots and completed rounds.

        cfg.readout supplies a circle classifier per ADC with a positive radius.
        Other counts unclassified shots, not calibrated leakage.
        Gain is a drive amplitude, not a photon number.
        Optional confusion correction is analysis-only.
        """

        # 1. Prepare host gain points and the board frequency grid.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        # 2. Keep the G/E/Other contract for every declared ADC.
        require_circle_classifiers(cfg.readout)
        read_axis = Classify(
            "readout",
        )

        freq_sweep = cfg.freq
        gain_sweep = cfg.gain

        gains = sweep2array(
            gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.probe_pulse.ch},
            allow_array=True,
        )
        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": cfg.probe_pulse.ch},
        )

        # 3. Use the same probability scale for all outcomes.
        figure, _ = context.plots.subplots(
            ncols=3,
            num="measurement",
            figsize=(11, 4),
            layout="constrained",
        )
        figure.suptitle("MIST gain–frequency map · qubit · classified populations")
        viewers = []
        for state, ax in zip(("g", "e", "other"), figure.axes, strict=True):
            ax.set(
                xlabel="MIST drive gain (a.u.)",
                ylabel="MIST drive RF frequency (MHz)",
                title=f"P({state})",
            )
            ax.grid(False)
            ax.set_facecolor("0.9")
            viewers.append(live_heatmap(ax, uniform=False))
            figure.colorbar(ax.images[0], ax=ax)

        @context.plots.on_plot_thread
        def plot_fn(counts: NDArray[np.float64]):
            """Render the class fraction maps."""
            totals = counts.sum(axis=-1)
            populations = counts / totals[..., None]
            return tuple(
                artist
                for i, viewer in enumerate(viewers)
                for artist in viewer.set(gains, freqs, populations[..., i])
            )

        figure.show()

        buffer = SignalBuffer(
            (len(gains), len(freqs), POPULATION_COUNT),
            dtype=np.float64,
            on_update=plot_fn,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            for gain, step in sched.scan("gain", gains.tolist()):
                # Fix gain for this row and sweep probe frequency after preparation.
                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                )
                freq = builder.add_sweep("freq", cfg.freq)
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.emit(Pulse("init_pulse", cfg.init_pulse))
                builder.emit(
                    Pulse("probe_pulse", cfg.probe_pulse, gain=gain, freq=freq)
                )
                builder.emit(Readout("readout", cfg.readout))
                prog = builder.build()
                step.run_acquire_summed(
                    prog,
                    soc,
                    reducer=shot_sums(prog, [read_axis]),
                    rounds=cfg.rounds,
                )

        # 4. Preserve gain/frequency/state order with explicit G/E/Other columns.
        counts = buffer.array

        return FreqPowerResult(gains=gains, freqs=freqs, counts=counts)

    def analyze(
        self,
        source: RunRecord[FreqPowerCfg, FreqPowerResult],
        options: FreqPowerAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot the run's [g,e,other] fractions with optional confusion correction.

        source supplies the saved sweep grid and three-column counts;
        normalize each point before confusion correction.
        cfg may be None. options controls correction and any axis conversion.
        Publish the fit figure to plots. Legacy G/E data raise ValueError.
        """
        # 1. Normalize each gain/frequency point and optionally correct G/E/Other fractions.
        result = source.result
        gains, freqs, populations = result.gains, result.freqs, result.counts
        populations = populations / populations.sum(axis=-1, keepdims=True)
        if populations.shape[-1] != POPULATION_COUNT:
            raise ValueError(
                "Expected [g,e,other] populations; legacy G/E files are unsupported. "
                "Acquire a new run."
            )

        corrected = correct_populations(
            populations, options.confusion_matrix, fallback_on_nonfinite=True
        )

        fig, _ = plots.subplots(
            num="fit", ncols=3, figsize=(11, 4), layout="constrained"
        )
        quantity = (
            "classified populations"
            if corrected is populations
            else "readout-corrected populations"
        )
        fig.suptitle(f"MIST gain–frequency map · qubit · {quantity}")
        for index, (state, ax) in enumerate(
            zip(("g", "e", "other"), fig.axes, strict=True)
        ):
            ax.set(
                xlabel="MIST drive gain (a.u.)",
                ylabel="MIST drive RF frequency (MHz)",
                title=f"P({state})",
            )
            ax.grid(False)
            ax.set_facecolor("0.9")
            (image,) = live_heatmap(ax, uniform=False).set(
                gains, freqs, corrected[..., index]
            )
            fig.colorbar(image, ax=ax)
