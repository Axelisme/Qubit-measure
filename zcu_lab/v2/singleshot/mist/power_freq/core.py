from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
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
    GainSweep,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    shot_sums,
    sweep2param,
)
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg, circle_classifier

from zcu_lab.v2.support import live_heatmap


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1, 2], dtype=np.int64)


@dataclass(frozen=True)
class FreqPowerResult:
    """Measured class counts (Ngain, Nfreq, 3) on rounded gain/frequency grids.

    Gains are in a.u.; frequencies are in MHz.

    The final state axis is [g,e,other], labeled by population_states=[0,1,2].
    Counts sum over shots and completed rounds, without normalization.
    Unmeasured points remain NaN after cancellation.
    """

    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    counts: NDArray[np.float64]
    population_states: NDArray[np.int64] = field(
        default_factory=_default_population_states
    )


class FreqPowerCfg(ConfigBase):
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
    freq: Annotated[SweepCfg, FieldMeta(label="Probe freq", unit="MHz")]
    gain: Annotated[GainSweep, FieldMeta(label="Probe gain", unit="a.u.")]


@dataclass(frozen=True)
class FreqPowerAnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None


class FreqPowerExp(Experiment[FreqPowerResult, FreqPowerCfg]):
    Options: ClassVar[type[FreqPowerAnalyzeOptions]] = FreqPowerAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[FreqPowerResult, FreqPowerCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("population_states", "Population", "None", dtype=np.int64),
            AxesSpec.Axis("freqs", "Drive Freq", "Hz", scale=1e6, dtype=np.float64),
            AxesSpec.Axis("gains", "Drive gain", "a.u.", dtype=np.float64),
        ),
        z=AxesSpec.Z("counts", "Counts", "shots", dtype=np.float64),
        result_type=FreqPowerResult,
        cfg_type=FreqPowerCfg,
        data_variable=DataVariable("counts"),
        cfg_schema_version="3.0",
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

        # Prepare host gain points and the board frequency grid.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        # Keep the G/E/Other contract for every declared ADC.
        readout_channels = (
            [cfg.readout.ro_ch]
            if isinstance(cfg.readout, PulseReadoutCfg)
            else [tone.ro_ch for tone in cfg.readout.tones]
        )
        for ro_ch in readout_channels:
            if circle_classifier(cfg.readout, ro_ch=ro_ch).radius is None:
                raise ValueError(f"ADC {ro_ch}: radius missing; run singleshot GE Post")
        read_axis = Classify(
            "readout", tone=None if isinstance(cfg.readout, PulseReadoutCfg) else 0
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

        # Display the measured G/E/Other fractions in classifier order.
        figure, _ = context.plots.subplots(
            "measurement", nrows=3, ncols=1, figsize=(12, 6)
        )
        viewers = tuple(
            live_heatmap(
                context.plots,
                "measurement",
                "gain (a.u.)",
                "freq (MHz)",
                title=state.capitalize(),
                uniform=False,
                axes=ax,
            )
            for state, ax in zip(
                ("ground", "excited", "other"), figure.axes, strict=True
            )
        )

        def plot_fn(counts: NDArray[np.float64]) -> None:
            populations = counts / counts.sum(axis=-1, keepdims=True)
            for i, viewer in enumerate(viewers):
                viewer.update(gains, freqs, populations[..., i])

        buffer = SignalBuffer(
            (len(gains), len(freqs), 3),
            dtype=np.float64,
            on_update=plot_fn,
        )
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            sched.cfg.probe_pulse.set_param("freq", sweep2param("freq", sched.cfg.freq))
            for gain, step in sched.scan("gain", gains.tolist()):
                step.cfg.probe_pulse.set_param("gain", gain)

                # Fix gain for this row and sweep probe frequency after preparation.
                builder = (
                    step.prog_builder(
                        soc,
                        soccfg,
                        reps=step.cfg.reps,
                        relax_delay=step.cfg.relax_delay,
                    )
                    .add_reset("reset", step.cfg.reset)
                    .add_pulse("init_pulse", step.cfg.init_pulse)
                    .add_pulse("probe_pulse", step.cfg.probe_pulse)
                    .add_readout("readout", step.cfg.readout)
                    .declare_sweep("freq", step.cfg.freq)
                )
                prog = builder.build()
                builder.run_program_summed(
                    prog,
                    processor=shot_sums(prog, [read_axis]),
                    sums2signal_fn=lambda sums: sums,
                    rounds=cfg.rounds,
                )

        # Preserve gain/frequency/state order with explicit G/E/Other columns.
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
        result = source.result
        gains, freqs, populations = result.gains, result.freqs, result.counts
        populations = populations / populations.sum(axis=-1, keepdims=True)
        if populations.shape[-1] != 3:
            raise ValueError(
                "Expected [g,e,other] populations; legacy G/E files are unsupported. "
                "Acquire a new run."
            )

        populations = correct_populations(populations, options.confusion_matrix)

        fig, _ = plots.subplots(
            "fit", nrows=3, ncols=1, figsize=(8, 10), layout="constrained"
        )
        ax_g, ax_e, ax_o = fig.axes

        # Stored populations are (gain, frequency, state); display frequency rows.
        im_g = ax_g.pcolormesh(gains, freqs, populations[..., 0].T, shading="nearest")
        ax_g.set_title("Ground State Population")
        ax_g.set_xlabel("Drive Gain (a.u.)")
        ax_g.set_ylabel("Drive Frequency (MHz)")
        fig.colorbar(im_g, ax=ax_g, label="Population (a.u.)")

        im_e = ax_e.pcolormesh(gains, freqs, populations[..., 1].T, shading="nearest")
        ax_e.set_title("Excited State Population")
        ax_e.set_xlabel("Drive Gain (a.u.)")
        ax_e.set_ylabel("Drive Frequency (MHz)")
        fig.colorbar(im_e, ax=ax_e, label="Population (a.u.)")

        im_o = ax_o.pcolormesh(gains, freqs, populations[..., 2].T, shading="nearest")
        ax_o.set_title("Other States Population")
        ax_o.set_xlabel("Drive Gain (a.u.)")
        ax_o.set_ylabel("Drive Frequency (MHz)")
        fig.colorbar(im_o, ax=ax_o, label="Population (a.u.)")
