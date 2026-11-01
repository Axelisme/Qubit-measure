from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import NonUniformImage
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
    GainSweep,
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
from zcu_tools.utils.process import minus_background

from zcu_lab.v2.support import live_heatmap, live_line
from zcu_lab.v2.support.resonance import get_resonance_freq


@dataclass(frozen=True)
class AcStarkResult:
    """Measured class counts (Ngain, Nfreq, 3) on rounded gain/frequency grids.

    Gains are in a.u.; frequencies are in MHz.

    The final state axis is [g,e,other], with fixed coordinates [0,1,2].
    Counts sum over shots and completed rounds, without normalization.
    Unmeasured points remain NaN after cancellation.
    """

    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    counts: NDArray[np.float64]


class AcStarkCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    stark_pulse1: Annotated[
        Pulse.Cfg, FieldMeta(label="Cavity Stark Tone (gain sweep)")
    ]
    stark_pulse2: Annotated[Pulse.Cfg, FieldMeta(label="Qubit Probe Tone (freq sweep)")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[GainSweep, FieldMeta(label="Gain", unit="a.u.")]
    freq: Annotated[SweepCfg, FieldMeta(label="Frequency", unit="MHz")]
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


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class AcStarkAnalyzeOptions:
    chi: float
    kappa: float
    confusion_matrix: NDArray[np.float64] | None = None
    cutoff: float | None = None


@dataclass(frozen=True)
class AcStarkAnalysis:
    ac_stark_coeff: float


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class AcStarkPlotOptions:
    ac_coeff: float
    confusion_matrix: NDArray[np.float64] | None = None
    cutoff: float | None = None


class AcStarkExp(Experiment[AcStarkResult, AcStarkCfg]):
    Options: ClassVar[type[AcStarkAnalyzeOptions]] = AcStarkAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[AcStarkResult, AcStarkCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                values=(0, 1, 2), label="Population", unit="None", dtype=np.int64
            ),
            AxesSpec.Axis(
                "freqs", label="Frequency", unit="Hz", scale=1e6, dtype=np.float64
            ),
            AxesSpec.Axis(
                "gains", label="Stark Pulse Gain", unit="a.u.", dtype=np.float64
            ),
        ),
        z=AxesSpec.Z("counts", label="Counts", unit="shots", dtype=np.float64),
        result_type=AcStarkResult,
        cfg_type=AcStarkCfg,
        data_variable=DataVariable("counts"),
        cfg_schema_version="4.0",
        tag="singleshot/ac_stark",
    )

    def run(
        self,
        cfg: AcStarkCfg,
        *,
        context: RunContext,
    ) -> AcStarkResult:
        """Measure the qubit resonance under a variable cavity Stark tone.

        After optional reset and initialization, launch stark_pulse1 without
        waiting for it to finish, play the frequency-swept stark_pulse2 qubit
        probe, then read out. Pulse lengths and delays in cfg set the overlap.
        Scan cavity gain on the host, uniformly in gain squared before
        hardware rounding, and probe frequency on the board.

        Return gains (a.u.), probe frequencies (MHz), and classified G/E/Other
        counts shaped (Ngain, Nfreq, 3). cfg.readout supplies circle
        classifiers with positive radii for all declared ADCs. Other counts
        unclassified shots, not calibrated leakage.
        Subsequent analysis extracts the Stark shift.
        """

        # Prepare the probe grid and a gain grid spaced in drive intensity.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        # Keep the G/E/Other contract for every declared ADC.
        require_circle_classifiers(cfg.readout)
        read_axis = Classify(
            "readout",
        )

        gain_sweep = cfg.gain

        freqs = sweep2array(
            cfg.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": cfg.stark_pulse2.ch},
        )
        gains = np.sqrt(
            np.linspace(gain_sweep.start**2, gain_sweep.stop**2, gain_sweep.expts)
        )
        gains = sweep2array(
            gains,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.stark_pulse1.ch},
            allow_array=True,
        )

        # Show measured G/E/Other fractions alongside the current trace.
        def configure_axes(ax: Axes) -> None:
            ax.set_ylim(0.0, 1.0)
            for line, label in zip(
                ax.lines, ("Ground", "Excited", "Other"), strict=True
            ):
                line.set_label(label)
            ax.legend()

        figure, _ = context.plots.subplots(
            nrows=2, ncols=2, num="measurement", figsize=(8, 6)
        )
        ax_ground, ax_excited, ax_other, ax_current = figure.axes
        heatmaps = []
        for state, ax in (
            ("ground", ax_ground),
            ("excited", ax_excited),
            ("other", ax_other),
        ):
            ax.set(
                xlabel="Stark Pulse Gain (a.u.)",
                ylabel="Probe Frequency (MHz)",
                title=state.capitalize(),
            )
            heatmaps.append(live_heatmap(ax, uniform=False, clim=(0.0, 1.0)))
        ax_current.set(xlabel="Probe Frequency (MHz)", ylabel="Population")
        cur_1d = live_line(ax_current, num_lines=3)
        configure_axes(ax_current)
        current_index = 0

        @context.plots.on_plot_thread
        def plot_fn(counts: NDArray[np.float64]):
            populations = counts / counts.sum(axis=-1, keepdims=True)
            artists = [
                artist
                for i, heatmap in enumerate(heatmaps)
                for artist in heatmap.set(gains, freqs, populations[..., i])
            ]
            return (*artists, *cur_1d.set(freqs, populations[current_index, :].T))

        figure.show()

        buffer = SignalBuffer(
            (len(gains), len(freqs), 3),
            dtype=np.float64,
            on_update=plot_fn,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            for gain_idx, (gain, step) in enumerate(
                sched.scan("resonator gain", gains.tolist())
            ):
                current_index = gain_idx

                # Start the cavity tone nonblocking so the qubit probe can overlap.
                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                )
                freq = builder.add_sweep("freq", cfg.freq)
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.emit(Pulse("init_pulse", cfg.init_pulse))
                builder.fork("default", into="stark")
                builder.emit(
                    Pulse("stark_pulse1", cfg.stark_pulse1, gain=gain), timeline="stark"
                )
                builder.emit(Pulse("stark_pulse2", cfg.stark_pulse2, freq=freq))
                builder.emit(Readout("readout", cfg.readout))
                prog = builder.build()
                step.run_acquire_summed(
                    prog,
                    soc,
                    processor=shot_sums(prog, [read_axis]),
                    rounds=cfg.rounds,
                )

        # Preserve unnormalized G/E/Other counts in the Result.
        counts = buffer.array
        return AcStarkResult(gains=gains, freqs=freqs, counts=counts)

    def analyze(
        self,
        source: RunRecord[AcStarkCfg, AcStarkResult],
        options: AcStarkAnalyzeOptions,
        *,
        plots: Plots,
    ) -> AcStarkAnalysis:
        """Fit the saved three-state map and return photons per gain squared.

        source supplies gains in a.u., frequencies in MHz and [g,e,other] counts;
        normalize each point before confusion correction.
        cfg may be None. options supplies chi/kappa in MHz, optional confusion
        correction and an exclusive gain cutoff. Publish the fit to plots.
        Legacy G/E data raise ValueError before correction or fitting.
        """
        result = source.result
        chi, kappa = options.chi, options.kappa
        confusion_matrix, cutoff = options.confusion_matrix, options.cutoff

        gains, freqs, populations = result.gains, result.freqs, result.counts
        populations = populations / populations.sum(axis=-1, keepdims=True)
        if populations.shape[-1] != 3:
            raise ValueError(
                "Expected [g,e,other] populations; legacy G/E files are unsupported. "
                "Acquire a new run."
            )

        # apply cutoff if provided
        if cutoff is not None:
            valid_indices = np.where(gains < cutoff)[0]
            gains = gains[valid_indices]
            populations = populations[valid_indices]

        populations = correct_populations(populations, confusion_matrix)

        # merge two populations into one
        populations = (
            np.abs(minus_background(populations[..., 0]))
            + np.abs(minus_background(populations[..., 1]))
        ) / 2

        s_gains, s_freqs = get_resonance_freq(gains, freqs, populations)

        gains2 = gains**2
        s_gains2 = s_gains**2

        # fitting max_freqs with ax2 + bx + c
        x2_fit = np.linspace(min(gains2), max(gains2), 100)
        b, c = np.polyfit(s_gains2, s_freqs, 1)
        y_fit = b * x2_fit + c

        # Calculate the Stark shift. eta accounts for the finite resonator
        # linewidth kappa relative to the dispersive shift chi (matches the
        # twotone AcStarkExp.analyze formula).
        eta = kappa**2 / (kappa**2 + chi**2)
        ac_coeff = abs(b) / (2 * eta * chi)

        # plot the data and the fitted polynomial
        avg_n = ac_coeff * gains2

        fig, ax1 = plots.subplots(num="fit")

        # Use NonUniformImage for better visualization with gain^2 as x-axis
        im = NonUniformImage(ax1, cmap="RdBu_r", interpolation="nearest")
        im.set_data(avg_n, freqs, populations.T)
        im.set_extent((avg_n[0], avg_n[-1], freqs[0], freqs[-1]))
        ax1.add_image(im)

        # Set proper limits for the plot
        ax1.set_xlim(avg_n[0], avg_n[-1])
        ax1.set_ylim(freqs[0], freqs[-1])

        # Plot the resonance frequencies and fitted curve
        ax1.plot(ac_coeff * s_gains2, s_freqs, ".", c="k")

        # Fit curve in terms of gain^2
        label = r"$\bar n$" + f" = {ac_coeff:.2g} " + r"$gain^2$"
        gain_fit = ac_coeff * x2_fit
        ax1.plot(gain_fit, y_fit, "-", label=label, color="y")

        # Create secondary x-axis for gain^2 (Readout Gain²)
        ax2 = ax1.twiny()

        # The secondary axis converts average photon number back to drive gain.
        ax1.set_xticks(ax1.get_xticks())
        ax1.set_xlabel(r"Average Photon Number ($\bar n$)", fontsize=14)

        # 上方次 x 軸顯示 gain
        avgn_ticks = ax1.get_xticks()
        gain_ticks = np.sqrt(avgn_ticks / ac_coeff)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(avgn_ticks)
        ax2.set_xticklabels([f"{gain:.2g}" for gain in gain_ticks])
        ax2.set_xlabel("Readout Gain (a.u.)", fontsize=14)

        ax1.set_ylabel("Qubit Frequency (MHz)", fontsize=14)
        ax1.legend(fontsize="x-large")
        ax1.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

        return AcStarkAnalysis(ac_stark_coeff=float(ac_coeff))

    def plot(
        self,
        source: RunRecord[AcStarkCfg, AcStarkResult],
        options: AcStarkPlotOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot the saved [g,e,other] map against photon number and MHz.

        Normalize the saved counts per point. cfg may be None. options supplies
        photons per gain squared, optional confusion correction and an exclusive
        gain cutoff. Publish the populations figure to plots.
        Legacy G/E data raise ValueError before correction or plotting.
        """
        result = source.result
        ac_coeff = options.ac_coeff
        confusion_matrix, cutoff = options.confusion_matrix, options.cutoff

        gains, freqs, populations = result.gains, result.freqs, result.counts
        populations = populations / populations.sum(axis=-1, keepdims=True)
        if populations.shape[-1] != 3:
            raise ValueError(
                "Expected [g,e,other] populations; legacy G/E files are unsupported. "
                "Acquire a new run."
            )

        # apply cutoff if provided
        if cutoff is not None:
            valid_indices = np.where(gains < cutoff)[0]
            gains = gains[valid_indices]
            populations = populations[valid_indices]

        populations = correct_populations(populations, confusion_matrix)

        gains2 = gains**2

        # plot the data and the fitted polynomial
        photons = ac_coeff * gains2

        fig, _ = plots.subplots(
            num="populations", nrows=1, ncols=3, figsize=(12, 4), sharey=True
        )
        ax1, ax2, ax3 = fig.axes

        max_p = np.max(populations).item()

        # Use NonUniformImage for better visualization with gain^2 as x-axis
        im = NonUniformImage(ax1, cmap="RdBu_r", interpolation="nearest")
        im.set_data(photons, freqs, populations[..., 0].T)
        im.set_extent((photons[0], photons[-1], freqs[0], freqs[-1]))
        im.set_clim(0.0, max_p)
        ax1.add_image(im)
        ax1.set_xlim(photons[0], photons[-1])
        ax1.set_ylim(freqs[0], freqs[-1])
        ax1.set_aspect("auto")
        ax1.set_title("Ground")
        ax1.set_xlabel(r"$\bar n$", fontsize=14)
        ax1.set_ylabel("Frequency (MHz)", fontsize=14)

        im = NonUniformImage(ax2, cmap="RdBu_r", interpolation="nearest")
        im.set_data(photons, freqs, populations[..., 1].T)
        im.set_extent((photons[0], photons[-1], freqs[0], freqs[-1]))
        im.set_clim(0.0, max_p)
        ax2.add_image(im)
        ax2.set_xlim(photons[0], photons[-1])
        ax2.set_ylim(freqs[0], freqs[-1])
        ax2.set_aspect("auto")
        ax2.set_title("Excited")
        ax2.set_xlabel(r"$\bar n$", fontsize=14)

        im = NonUniformImage(ax3, cmap="RdBu_r", interpolation="nearest")
        im.set_data(photons, freqs, populations[..., 2].T)
        im.set_extent((photons[0], photons[-1], freqs[0], freqs[-1]))
        im.set_clim(0.0, max_p)
        ax3.add_image(im)
        ax3.set_xlim(photons[0], photons[-1])
        ax3.set_ylim(freqs[0], freqs[-1])
        ax3.set_aspect("auto")
        ax3.set_title("Other")
        ax3.set_xlabel(r"$\bar n$", fontsize=14)

        fig.tight_layout()
