from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import NonUniformImage
from numpy.typing import NDArray
from pydantic import field_serializer
from zcu_tools.analysis.fitting import fitlor
from zcu_tools.analysis.singleshot import (
    calc_populations,
    correct_populations,
    raw_population_signal,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, ProgramExpCfg
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, sweep2param
from zcu_tools.utils.process import minus_background


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


@dataclass(frozen=True)
class AcStarkResult:
    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    populations: NDArray[np.float64]
    population_states: NDArray[np.int64] = field(
        default_factory=_default_population_states
    )


def get_resonance_freq(
    xs: NDArray[np.float64],
    freqs: NDArray[np.float64],
    populations: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    s_xs = []
    s_freqs = []

    prev_freq = np.nan
    for x, pop in zip(xs, populations, strict=False):
        if np.any(np.isnan(pop)):
            continue

        param, _ = fitlor(freqs, pop)
        curr_freq = param[3]

        if abs(curr_freq - prev_freq) > 0.1 * (freqs[-1] - freqs[0]):
            continue

        prev_freq = curr_freq

        s_xs.append(x)
        s_freqs.append(curr_freq)

    return np.array(s_xs), np.array(s_freqs)


class AcStarkModuleCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    stark_pulse1: Annotated[
        Pulse.Cfg, FieldMeta(label="Cavity Stark Tone (gain sweep)")
    ]
    stark_pulse2: Annotated[Pulse.Cfg, FieldMeta(label="Qubit Probe Tone (freq sweep)")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]


class AcStarkSweepCfg(ConfigBase):
    gain: Annotated[SweepCfg, FieldMeta(label="Gain", unit="a.u.")]
    freq: Annotated[SweepCfg, FieldMeta(label="Frequency", unit="MHz")]


class AcStarkCfg(ProgramExpCfg):
    modules: AcStarkModuleCfg
    sweep: AcStarkSweepCfg
    g_center: Annotated[complex, FieldMeta(label="Ground center")]
    e_center: Annotated[complex, FieldMeta(label="Excited center")]
    radius: Annotated[float, FieldMeta(label="Classification radius")]

    @field_serializer("g_center", "e_center")
    def serialize_center(self, value: complex) -> str:
        return str(value)


@dataclass(frozen=True)
class AcStarkAnalyzeOptions:
    chi: float
    kappa: float
    confusion_matrix: NDArray[np.float64] | None = None
    cutoff: float | None = None


@dataclass(frozen=True)
class AcStarkAnalysis:
    ac_stark_coeff: float


@dataclass(frozen=True)
class AcStarkPlotOptions:
    ac_coeff: float
    confusion_matrix: NDArray[np.float64] | None = None
    cutoff: float | None = None


class AcStarkExp(Experiment[AcStarkResult, AcStarkCfg]):
    Options: ClassVar[type[AcStarkAnalyzeOptions]] = AcStarkAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[AcStarkResult, AcStarkCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("population_states", "GE Population", "None", dtype=np.int64),
            AxesSpec.Axis("freqs", "Frequency", "Hz", scale=1e6, dtype=np.float64),
            AxesSpec.Axis("gains", "Stark Pulse Gain", "a.u.", dtype=np.float64),
        ),
        z=AxesSpec.Z("populations", "Population", "a.u.", dtype=np.float64),
        result_type=AcStarkResult,
        cfg_type=AcStarkCfg,
        cfg_schema_version="2.0",
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

        Return gains (a.u.), probe frequencies (MHz), and classified G/E
        fractions shaped (Ngain, Nfreq, 2). cfg supplies distinct finite IQ
        centers and a positive finite radius. Other is 1-G-E, not a calibrated
        leakage population. Subsequent analysis extracts the Stark shift.
        """

        # Prepare the probe grid and a gain grid spaced in drive intensity.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        modules = cfg.modules

        gain_sweep = cfg.sweep.gain

        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.stark_pulse2.ch},
        )
        gains = np.sqrt(
            np.linspace(gain_sweep.start**2, gain_sweep.stop**2, gain_sweep.expts)
        )
        gains = sweep2array(
            gains,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.stark_pulse1.ch},
            allow_array=True,
        )

        # Show G/E and the unclassified remainder alongside the current trace.
        def configure_axes(ax: Axes) -> None:
            ax.set_ylim(0.0, 1.0)
            for line, label in zip(
                ax.lines, ("Ground", "Excited", "Other"), strict=True
            ):
                line.set_label(label)
            ax.legend()

        figure, _ = context.plots.subplots(
            "measurement", nrows=2, ncols=2, figsize=(8, 6)
        )
        ax_ground, ax_excited, ax_other, ax_current = figure.axes
        g_2d, e_2d, o_2d = (
            context.plots.liveplot_2d(
                "measurement",
                "Stark Pulse Gain (a.u.)",
                "Probe Frequency (MHz)",
                title=state.capitalize(),
                uniform=False,
                clim=(0.0, 1.0),
                axes=ax,
            )
            for state, ax in (
                ("ground", ax_ground),
                ("excited", ax_excited),
                ("other", ax_other),
            )
        )
        cur_1d = context.plots.liveplot_1d(
            "measurement",
            "Probe Frequency (MHz)",
            "Population",
            num_lines=3,
            configure_axes=configure_axes,
            axes=ax_current,
        )
        current_index = 0

        def plot_fn(data: NDArray[np.float64]) -> None:
            i = current_index

            populations = calc_populations(data)

            g_2d.update(gains, freqs, populations[..., 0])
            e_2d.update(gains, freqs, populations[..., 1])
            o_2d.update(gains, freqs, populations[..., 2])
            cur_1d.update(freqs, populations[i, :].T)

        buffer = SignalBuffer(
            (len(gains), len(freqs), 2),
            dtype=np.float64,
            on_update=plot_fn,
        )
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            sched.cfg.modules.stark_pulse2.set_param(
                "freq", sweep2param("freq", sched.cfg.sweep.freq)
            )
            for gain_idx, (gain, step) in enumerate(
                sched.scan("resonator gain", gains.tolist())
            ):
                modules = step.cfg.modules
                modules.stark_pulse1.set_param("gain", gain)
                current_index = gain_idx

                # Start the cavity tone nonblocking so the qubit probe can overlap.
                _ = (
                    step.prog_builder(soc, soccfg)
                    .add_reset("reset", modules.reset)
                    .add_pulse("init_pulse", modules.init_pulse)
                    .add_pulse(
                        "stark_pulse1",
                        modules.stark_pulse1,
                        block_mode=False,
                    )
                    .add_pulse("stark_pulse2", modules.stark_pulse2)
                    .add_readout("readout", modules.readout)
                    .declare_sweep("freq", step.cfg.sweep.freq)
                    .build_and_acquire(
                        # Classify shots before averaging; do not classify mean IQ.
                        raw2signal_fn=raw_population_signal,
                        g_center=cfg.g_center,
                        e_center=cfg.e_center,
                        ge_radius=cfg.radius,
                    )
                )

        # Preserve measured G/E fractions; Other remains a derived quantity.
        signals = buffer.array
        return AcStarkResult(gains=gains, freqs=freqs, populations=signals)

    def analyze(
        self,
        source: RunRecord[AcStarkCfg, AcStarkResult],
        options: AcStarkAnalyzeOptions,
        *,
        plots: Plots,
    ) -> AcStarkAnalysis:
        result = source.result
        chi, kappa = options.chi, options.kappa
        confusion_matrix, cutoff = options.confusion_matrix, options.cutoff

        gains, freqs, populations = result.gains, result.freqs, result.populations

        # apply cutoff if provided
        if cutoff is not None:
            valid_indices = np.where(gains < cutoff)[0]
            gains = gains[valid_indices]
            populations = populations[valid_indices]

        populations = calc_populations(populations)  # (xs, 2, Ts, 3)

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

        fig, ax1 = plots.subplots("fit")

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
        result = source.result
        ac_coeff = options.ac_coeff
        confusion_matrix, cutoff = options.confusion_matrix, options.cutoff

        gains, freqs, populations = result.gains, result.freqs, result.populations

        # apply cutoff if provided
        if cutoff is not None:
            valid_indices = np.where(gains < cutoff)[0]
            gains = gains[valid_indices]
            populations = populations[valid_indices]

        populations = calc_populations(populations)  # (xs, 2, Ts, 3)

        populations = correct_populations(populations, confusion_matrix)

        gains2 = gains**2

        # plot the data and the fitted polynomial
        photons = ac_coeff * gains2

        fig, _ = plots.subplots(
            "populations", nrows=1, ncols=3, figsize=(12, 4), sharey=True
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
