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
from zcu_tools.utils.process import minus_background

from zcu_lab.v2.support import live_heatmap
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


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class AcStarkAnalyzeOptions:
    chi: float
    kappa: float
    confusion_matrix: NDArray[np.float64] | None = None
    cutoff: float | None = None


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class AcStarkPlotOptions:
    ac_coeff: float
    confusion_matrix: NDArray[np.float64] | None = None
    cutoff: float | None = None


@dataclass(frozen=True)
class AcStarkAnalysis:
    ac_stark_coeff: float


class AcStarkCfg(ConfigBase):
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
    stark_pulse1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Cavity Stark tone",
            description="Enter the cavity drive that overlaps qubit spectroscopy, with its gain swept.",
        ),
    ]
    stark_pulse2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit probe tone",
            description="Enter the qubit spectroscopy pulse whose RF frequency is swept.",
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
            label="Cavity Stark gain",
            unit="a.u.",
            description="Set the Stark-tone gain sweep in a.u.; points are equally spaced in gain².",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Qubit probe RF frequency",
            unit="MHz",
            description="Set the qubit spectroscopy RF sweep in MHz during the cavity Stark tone.",
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

        # 1. Prepare the classified probe grid and intensity-spaced gains.
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

        # 2. Use the same probability scale for the three readout outcomes.
        figure, _ = context.plots.subplots(
            ncols=3,
            num="measurement",
            figsize=(11, 4),
            layout="constrained",
        )
        progress = figure.suptitle("AC Stark scan · qubit · classified populations")
        heatmaps = []
        for state, ax in zip(("g", "e", "other"), figure.axes, strict=True):
            ax.set(
                xlabel="Cavity Stark gain (a.u.)",
                ylabel="Qubit probe RF frequency (MHz)",
                title=f"P({state})",
            )
            ax.grid(False)
            heatmaps.append(live_heatmap(ax, uniform=False, clim=(0.0, 1.0)))
        figure.colorbar(
            figure.axes[0].images[0], ax=figure.axes, label="Probability (0–1)"
        )

        @context.plots.on_plot_thread
        def plot_fn(counts: NDArray[np.float64]):
            """Display the classified probability maps and completed cells."""
            totals = counts.sum(axis=-1)
            populations = counts / totals[..., None]
            progress.set_text(
                "AC Stark scan · qubit · classified populations\n"
                f"{np.count_nonzero(totals > 0)}/{totals.size} gain–frequency points acquired"
            )
            return (
                progress,
                *(
                    artist
                    for i, heatmap in enumerate(heatmaps)
                    for artist in heatmap.set(gains, freqs, populations[..., i])
                ),
            )

        figure.show()

        buffer = SignalBuffer(
            (len(gains), len(freqs), 3),
            dtype=np.float64,
            on_update=plot_fn,
        )
        # 3. Acquire overlapping cavity and qubit probes across the gain grid.
        with Schedule(buffer, scope=context.scope) as sched:
            for gain, step in sched.scan("resonator gain", gains.tolist()):
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
                    reducer=shot_sums(prog, [read_axis]),
                    rounds=cfg.rounds,
                )

        # 4. Preserve measured counts and their gain/frequency coordinates.
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
        # 1. Normalize the saved three-state counts and apply analysis controls.
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

        # 2. Subtract the off-peak baseline and extract one resonance per gain.
        contrast = (
            np.abs(minus_background(populations[..., 0]))
            + np.abs(minus_background(populations[..., 1]))
        ) / 2
        s_gains, s_freqs = get_resonance_freq(gains, freqs, contrast)
        gains2 = gains**2
        dense_gain2 = np.linspace(float(gains2.min()), float(gains2.max()), 250)
        slope, intercept = np.polyfit(s_gains**2, s_freqs, 1)
        deviations = s_freqs - s_freqs.mean()
        total = float(deviations @ deviations)
        residuals = s_freqs - (slope * s_gains**2 + intercept)
        r2 = (
            "unavailable"
            if total == 0.0
            else f"{1 - (residuals @ residuals) / total:.3g}"
        )
        eta = kappa**2 / (kappa**2 + chi**2)
        ac_coeff = abs(slope) / (2 * eta * chi)

        # 3. Preserve the live coordinates, then show the line used for calibration.
        _plot_populations(
            plots,
            gains,
            freqs,
            populations,
            xlabel="Cavity Stark gain (a.u.)",
            corrected=confusion_matrix is not None,
        )
        fig, ax = plots.subplots(num="fit", layout="constrained")
        heatmap = live_heatmap(ax, uniform=False)
        (image,) = heatmap.set(gains2, freqs, contrast)
        ax.grid(False)
        fig.colorbar(image, ax=ax, label="Baseline-subtracted g/e probability contrast")
        ax.plot(
            s_gains**2,
            s_freqs,
            "o",
            markersize=4,
            color="black",
            label="Extracted resonances",
        )
        ax.plot(
            dense_gain2,
            slope * dense_gain2 + intercept,
            color="tab:orange",
            label="Linear Stark fit",
        )
        ax.set(
            title=f"AC Stark scan · qubit · resonance shift\n"
            f"Photon calibration = {ac_coeff:.3g} photons / gain² · R² = {r2}",
            xlabel="Squared cavity Stark gain (a.u.²)",
            ylabel="Qubit probe RF frequency (MHz)",
            xlim=(float(gains2.min()), float(gains2.max())),
            ylim=(float(freqs.min()), float(freqs.max())),
        )
        ax.legend(loc="best")

        # 4. Return the photon-number calibration for later analysis.
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

        _plot_populations(
            plots,
            ac_coeff * gains**2,
            freqs,
            populations,
            xlabel="Mean cavity photon number (count)",
            corrected=confusion_matrix is not None,
        )


def _plot_populations(
    plots: Plots,
    coordinates: NDArray[np.float64],
    freqs: NDArray[np.float64],
    populations: NDArray[np.float64],
    *,
    xlabel: str,
    corrected: bool,
) -> None:
    """Show comparable G/E/Other maps against drive gain or calibrated photon number."""
    fig, _ = plots.subplots(
        num="populations",
        ncols=3,
        figsize=(11, 4),
        layout="constrained",
    )
    quantity = (
        "readout-corrected populations" if corrected else "classified populations"
    )
    fig.suptitle(f"AC Stark scan · qubit · {quantity}")
    for index, (ax, state) in enumerate(
        zip(fig.axes, ("g", "e", "other"), strict=True)
    ):
        ax.set(
            title=f"P({state})", xlabel=xlabel, ylabel="Qubit probe RF frequency (MHz)"
        )
        ax.grid(False)
        live_heatmap(ax, uniform=False, clim=(0.0, 1.0)).set(
            coordinates, freqs, populations[..., index]
        )
    fig.colorbar(fig.axes[0].images[0], ax=fig.axes, label="Probability (0–1)")
