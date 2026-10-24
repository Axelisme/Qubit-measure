from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import field_serializer
from zcu_tools.analysis.singleshot import (
    calc_populations,
    correct_populations,
    raw_population_signal,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting import live_heatmap
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, sweep2param


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


@dataclass(frozen=True)
class FreqPowerResult:
    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.float64]
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
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset", group="Modules")] = None
    init_pulse: Annotated[
        Pulse.Cfg | None, FieldMeta(label="Init Pulse", group="Modules")
    ] = None
    probe_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Probe Pulse", group="Modules")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout", group="Modules")]
    freq: Annotated[SweepCfg, FieldMeta(label="Probe freq", unit="MHz", group="Sweep")]
    gain: Annotated[SweepCfg, FieldMeta(label="Probe gain", unit="a.u.", group="Sweep")]
    g_center: Annotated[complex, FieldMeta(label="Ground center")]
    e_center: Annotated[complex, FieldMeta(label="Excited center")]
    radius: Annotated[float, FieldMeta(label="Classification radius")]

    @field_serializer("g_center", "e_center")
    def serialize_center(self, value: complex) -> str:
        return str(value)


@dataclass(frozen=True)
class FreqPowerAnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None


class FreqPowerExp(Experiment[FreqPowerResult, FreqPowerCfg]):
    Options: ClassVar[type[FreqPowerAnalyzeOptions]] = FreqPowerAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[FreqPowerResult, FreqPowerCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("population_states", "GE Population", "None", dtype=np.int64),
            AxesSpec.Axis("freqs", "Drive Freq", "Hz", scale=1e6, dtype=np.float64),
            AxesSpec.Axis("gains", "Drive gain", "a.u.", dtype=np.float64),
        ),
        z=AxesSpec.Z("signals", "Population", "a.u.", dtype=np.float64),
        result_type=FreqPowerResult,
        cfg_type=FreqPowerCfg,
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
        frequencies (MHz), and classified G/E fractions shaped (Ngain, Nfreq, 2),
        averaged over shots and rounds.

        cfg supplies distinct finite IQ centers and a positive finite radius.
        Other is 1-G-E, not calibrated leakage. Gain remains a drive amplitude,
        not a photon number; optional confusion correction is analysis-only.
        """

        # Prepare host gain points and the board frequency grid.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        modules = cfg

        freq_sweep = cfg.freq
        gain_sweep = cfg.gain

        gains = sweep2array(
            gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
            allow_array=True,
        )
        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
        )

        # Display all classification regions while storing only G/E fractions.
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

        def plot_fn(data: NDArray[np.float64]) -> None:
            populations = calc_populations(data)
            for i, viewer in enumerate(viewers):
                viewer.update(gains, freqs, populations[..., i])

        buffer = SignalBuffer(
            (len(gains), len(freqs), 2),
            dtype=np.float64,
            on_update=plot_fn,
        )
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            sched.cfg.probe_pulse.set_param("freq", sweep2param("freq", sched.cfg.freq))
            for gain, step in sched.scan("gain", gains.tolist()):
                modules = step.cfg
                modules.probe_pulse.set_param("gain", gain)

                # Fix gain for this row and sweep probe frequency after preparation.
                _ = (
                    step.prog_builder(
                        soc,
                        soccfg,
                        reps=step.cfg.reps,
                        relax_delay=step.cfg.relax_delay,
                    )
                    .add_reset("reset", modules.reset)
                    .add_pulse("init_pulse", modules.init_pulse)
                    .add_pulse("probe_pulse", modules.probe_pulse)
                    .add_readout("readout", modules.readout)
                    .declare_sweep("freq", step.cfg.freq)
                    .build_and_acquire(
                        # Classify individual shots before averaging populations.
                        raw2signal_fn=raw_population_signal,
                        g_center=cfg.g_center,
                        e_center=cfg.e_center,
                        ge_radius=cfg.radius,
                        rounds=cfg.rounds,
                    )
                )

        # Preserve gain/frequency/G-E axis order; Other is derived on demand.
        signals = buffer.array

        return FreqPowerResult(gains=gains, freqs=freqs, signals=signals)

    def analyze(
        self,
        source: RunRecord[FreqPowerCfg, FreqPowerResult],
        options: FreqPowerAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        result = source.result
        gains, freqs, populations = result.gains, result.freqs, result.signals

        populations = calc_populations(populations)

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
