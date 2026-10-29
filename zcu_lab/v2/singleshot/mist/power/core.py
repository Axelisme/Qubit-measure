from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from zcu_tools.analysis.singleshot import correct_populations
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import GainSweep, Pulse, Readout, Reset, sweep2param
from zcu_tools.program.v2.modules.readout import classification_processors

from zcu_lab.v2.support import live_line


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1, 2], dtype=np.int64)


@dataclass(frozen=True)
class PowerResult:
    """Rounded gains in a.u. and measured signals (Ngain, 3).

    The final state axis is [g,e,other], labeled by population_states=[0,1,2].
    Unmeasured points remain NaN after cancellation.
    """

    gains: NDArray[np.float64]
    signals: NDArray[np.float64]
    population_states: NDArray[np.int64] = field(
        default_factory=_default_population_states
    )


class PowerCfg(ConfigBase):
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    probe_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Probe Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[GainSweep, FieldMeta(label="Probe gain", unit="a.u.")]


@dataclass(frozen=True)
class PowerAnalyzeOptions:
    ac_coeff: float | None = None
    log_scale: bool = False
    confusion_matrix: NDArray[np.float64] | None = None


class PowerExp(Experiment[PowerResult, PowerCfg]):
    Options: ClassVar[type[PowerAnalyzeOptions]] = PowerAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[PowerResult, PowerCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("population_states", "Population", "None", dtype=np.int64),
            AxesSpec.Axis("gains", "Drive gain", "a.u.", dtype=np.float64),
        ),
        z=AxesSpec.Z("signals", "Population", "a.u.", dtype=np.float64),
        result_type=PowerResult,
        cfg_type=PowerCfg,
        cfg_schema_version="3.0",
        tag="singleshot/mist/power",
    )

    def run(self, cfg: PowerCfg, *, context: RunContext) -> PowerResult:
        """Measure MIST populations versus probe gain.

        Apply optional reset and init, sweep probe gain on the board, then
        read out. Return rounded gains (a.u.) and classified G/E/Other fractions
        shaped (Ngain, 3), averaged over shots and rounds. Gain is not
        calibrated power; analysis can convert it to photons with ac_coeff.

        cfg.readout supplies a circle classifier per ADC with a positive radius.
        Other counts unclassified shots, not calibrated leakage.
        Confusion correction and photon/log plotting are analysis choices.
        """

        # Prepare the hardware gain grid before collecting populations.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        processors = classification_processors(cfg.readout, require_other=True)

        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.probe_pulse.ch},
        )

        # Show the measured G/E/Other fractions in classifier order.
        def configure_axes(ax: Axes) -> None:
            ax.set_ylim(0.0, 1.0)
            for line, label in zip(
                ax.lines, ("Ground", "Excited", "Other"), strict=True
            ):
                line.set_label(label)
            ax.legend()

        viewer = live_line(
            context.plots,
            "measurement",
            "Pulse gain",
            "MIST",
            num_lines=3,
            configure_axes=configure_axes,
        )
        buffer = SignalBuffer(
            (len(gains), 3),
            dtype=np.float64,
            on_update=lambda data: viewer.update(gains, data.T),
        )
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            run_cfg = sched.cfg
            gain_sweep = run_cfg.gain
            run_cfg.probe_pulse.set_param("gain", sweep2param("gain", gain_sweep))

            # Repeat reset/preparation/probe/readout at each hardware sweep point.
            _ = (
                sched.prog_builder(
                    soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
                )
                .add_reset("reset", run_cfg.reset)
                .add_pulse("init_pulse", run_cfg.init_pulse)
                .add_pulse("probe_pulse", run_cfg.probe_pulse)
                .add_readout("readout", run_cfg.readout)
                .declare_sweep("gain", gain_sweep)
                .build_and_acquire_processed(
                    # First ADC/read, with all G/E/Other columns.
                    processed2signal_fn=lambda processed: processed[0][0],
                    processors=processors,
                    rounds=cfg.rounds,
                )
            )

        # Keep the measured G/E/Other columns in the saved Result.
        signals = buffer.array

        return PowerResult(gains=gains, signals=signals)

    def analyze(
        self,
        source: RunRecord[PowerCfg, PowerResult],
        options: PowerAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot the run's [g,e,other] fractions with optional confusion correction.

        source supplies the saved sweep grid and three-column populations;
        cfg may be None. options controls correction and any axis conversion.
        Publish the fit figure to plots. Legacy G/E data raise ValueError.
        """
        result = source.result
        gains, populations = result.gains, result.signals
        if populations.shape[-1] != 3:
            raise ValueError(
                "Expected [g,e,other] populations; legacy G/E files are unsupported. "
                "Acquire a new run."
            )

        populations = correct_populations(populations, options.confusion_matrix)

        if options.ac_coeff is None:
            xs = gains
            xlabel = "probe gain (a.u.)"
        else:
            xs = options.ac_coeff * gains**2
            xlabel = r"$\bar n$"

        _, ax = plots.subplots("fit", figsize=(6, 6))

        ax.plot(
            xs,
            populations[:, 0],
            color="blue",
            label="Ground",
            ls="-",
            marker="o",
            markersize=1,
        )
        ax.plot(
            xs,
            populations[:, 1],
            color="red",
            label="Excited",
            ls="-",
            marker="o",
            markersize=1,
        )
        ax.plot(
            xs,
            populations[:, 2],
            color="green",
            label="Other",
            ls="-",
            marker="o",
            markersize=1,
        )
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Population", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_ylim(0, 1)
        if options.log_scale:
            ax.set_xscale("log")
