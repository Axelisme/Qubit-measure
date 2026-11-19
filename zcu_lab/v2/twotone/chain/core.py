"""Two-tone readout-chain sweep: qubit lines as distance from background.

A dwell is one shot at one drive frequency: reset, optional init pulse, then
READS_PER_DWELL (drive pulse, readout) cycles without reset in between. Each
dwell is reduced to 11 numbers on the host and only running sums are kept, so
the saved record does not grow with the number of sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.analysis.fitting import PeakCandidate
from zcu_tools.analysis.fitting.spectrum import SpectrumFitOptions, fit_spectrum
from zcu_tools.analysis.readout_chain import SweepSums, sweep_curve
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
    config,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    PulseReadout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    chain_sweep_sums,
)

from zcu_lab.v2.support import live_line

# (drive, readout) cycles per dwell, fixed by the experiment design: 16 to 96
# give the same line heights at fixed total reads, and 48 keeps the tail that
# slow states leave along the scan direction near one point.
READS_PER_DWELL = 48


@validated_dataclass(frozen=True)
class ChainAnalyzeOptions:
    """Select lor/sinc and a candidate near expected_center_mhz (None: strongest).

    min_prominence is relative to projected signal ptp; min_distance_mhz is the
    minimum candidate separation. min_snr gates prominence and height above
    the median in first-difference MAD noise units. min_width_samples gates
    observable half-prominence width in sample intervals; 0 disables this gate.
    Raise it only for fine scans. peaks reports height SNR, not prominence SNR.
    The fitted overlay is always drawn.
    """

    model_type: Annotated[Literal["lor", "sinc"], FieldMeta(label="Model type")] = "lor"
    expected_center_mhz: Annotated[
        float | None,
        FieldMeta(label="Expected center", unit="MHz"),
        Field(allow_inf_nan=False),
    ] = None
    min_prominence: Annotated[
        float, FieldMeta(label="Minimum prominence / signal range"), Field(ge=0, le=1)
    ] = 0.2
    min_snr: Annotated[
        float,
        FieldMeta(label="Minimum height above median / noise sigma"),
        Field(ge=0, allow_inf_nan=False),
    ] = 5.0
    min_width_samples: Annotated[
        float,
        FieldMeta(label="Minimum half-prominence width", unit="samples"),
        Field(ge=0, allow_inf_nan=False),
    ] = 0.0
    min_distance_mhz: Annotated[
        float,
        FieldMeta(label="Minimum peak separation", unit="MHz"),
        Field(ge=0, allow_inf_nan=False),
    ] = 0.0


@dataclass(frozen=True)
class ChainAnalysis:
    """Peak fitted on the sigma curve.

    freq and freq_err are the fitted centre and its error in MHz. fwhm and
    fwhm_err are the full width at half maximum of the sigma peak in MHz; this
    width only approximates the line width. level is the dashed 5% family
    false-mark level in sigma. peaks lists every candidate on the sigma curve,
    most prominent first.
    """

    freq: float
    freq_err: float
    fwhm: float
    fwhm_err: float
    level: float
    peaks: tuple[PeakCandidate, ...] = ()


class ChainCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Init Pulse",
            description="Optional fixed-frequency pulse before each dwell's readout chain.",
        ),
    ] = None
    qub_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Probe Pulse",
            description="Qubit drive played before every readout of the chain; its frequency is set by the sweep.",
        ),
    ]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Frequency",
            unit="MHz",
            description="Qubit drive frequency sweep.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            description="Fast sweeps in each software round.",
        ),
    ] = Field(ge=12)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            description="Software rounds, each followed by a live update.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Wait after each dwell's chain, before the next dwell.",
        ),
    ]


def restore_sums(data: GroupedLoadData[ChainCfg]) -> SweepSums:
    """Rebuild the saved running sums from the grouped record."""
    dwell_count = data.variable("dwell_count")
    return SweepSums(
        axis=dwell_count.axes[0],
        dwell_count=dwell_count.z,
        feature_sums=data.variable("feature_sums").z,
        product_sum=data.variable("product_sum").z,
    )


# Frequencies are MHz in memory and Hz on disk; features are the 11 numbers.
FREQUENCY_AXIS = VariableAxisSpec(
    label="Frequency", unit="Hz", field_name="axis", scale=1e6
)
FEATURES = tuple(range(11))


class ChainExp(Experiment[SweepSums, ChainCfg]):
    Options: ClassVar[type[ChainAnalyzeOptions]] = ChainAnalyzeOptions

    # The record is the SweepSums of readout_chain: dwell counts and 11 feature
    # sums per frequency, and one 11 x 11 product sum pooled over all points.
    AXES_SPEC: ClassVar[GroupedAxesSpec[SweepSums, ChainCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                "dwell_count",
                (FREQUENCY_AXIS,),
                VariableZSpec("dwell_count", label="Dwells", unit=""),
            ),
            VariableSpec(
                "feature_sums",
                (
                    VariableAxisSpec(label="Feature", unit="", values=FEATURES),
                    FREQUENCY_AXIS,
                ),
                VariableZSpec("feature_sums", label="Feature sum", unit="a.u."),
            ),
            VariableSpec(
                "product_sum",
                (
                    VariableAxisSpec(label="Column feature", unit="", values=FEATURES),
                    VariableAxisSpec(label="Row feature", unit="", values=FEATURES),
                ),
                VariableZSpec("product_sum", label="Product sum", unit="a.u."),
            ),
        ),
        result_type=SweepSums,
        cfg_type=ChainCfg,
        tag="twotone/chain",
        cfg_schema_version="3.0",
        result_builder=restore_sums,
        labber_layout="independent",
    )

    def run(self, cfg: ChainCfg, *, context: RunContext) -> SweepSums:
        """Sweep the drive frequency with a readout chain at every point.

        Each dwell applies reset, optional init_pulse, then READS_PER_DWELL
        cycles of the frequency-swept probe pulse and a readout, with no reset
        between cycles. One round makes cfg.reps fast sweeps over all
        frequencies. After each round the live plot shows the sigma curve of
        readout_chain.sweep_curve with its dashed 5% level. Return the running
        sums of all completed rounds as SweepSums; frequencies are in MHz.
        A stop keeps completed rounds. A stop before the first round finishes
        raises RuntimeError because there are no sums to keep.
        """
        # 1. Quantize drive frequencies for repeated-readout significance spectroscopy.
        freqs = sweep2array(
            cfg.freq, "freq", {"soccfg": context.soccfg, "gen_ch": cfg.qub_pulse.ch}
        )

        # 2. Monitor spectral significance relative to the false-mark threshold.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Frequency (MHz)", ylabel="Distance from background (sigma)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid()
        curve_line = live_line(ax)
        (level_line,) = ax.plot(
            [], [], color="gray", linestyle="--", label="5% false-mark level"
        )
        ax.legend()

        @context.plots.on_plot_thread
        def update(augmented_sums: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Refresh the sweep significance curve and its reference level; propagate invalid-sum errors.

            augmented_sums contains float64 moment sums shaped (frequency, 12, 12).
            Return the changed artists without modifying the input array.
            """
            sums = SweepSums.from_augmented(freqs, augmented_sums)
            curve = sweep_curve(sums)
            level_line.set_data([freqs[0], freqs[-1]], [curve.level, curve.level])
            # Set the level first: the curve update rescales the axes to both lines.
            return (level_line, *curve_line.set(freqs, curve.sigma))

        fig.show()
        augmented_sums = SignalBuffer(
            (len(freqs), 12, 12), dtype=np.float64, on_update=update
        )

        # 3. Accumulate drive-readout chain moments at each frequency after initialization.
        with Schedule(augmented_sums, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            freq = builder.add_sweep("freq", cfg.freq)

            if cfg.reset is not None:
                builder.emit(cfg.reset.build("reset"))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            with builder.repeat("chain", READS_PER_DWELL):
                builder.emit(Pulse("qub_pulse", cfg.qub_pulse, freq=freq))
                builder.emit(PulseReadout("readout", cfg.readout))
                # The next drive starts after this readout's ADC window closes.
                builder.delay_auto()

            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                processor=chain_sweep_sums(prog, "readout"),
                rounds=cfg.rounds,
            )

        # 4. Summed acquisition writes whole rounds, so NaN means no round finished.
        if np.isnan(augmented_sums.array).any():
            raise RuntimeError(
                "Stopped before the first round finished; no sums to keep"
            )
        return SweepSums.from_augmented(freqs, augmented_sums.array)

    def analyze(
        self,
        source: RunRecord[ChainCfg, SweepSums],
        options: ChainAnalyzeOptions,
        *,
        plots: Plots,
    ) -> ChainAnalysis:
        """Fit one peak of the sigma curve with the numerical spectrum fitter.

        Recompute the sigma curve from the saved sums. Select the candidate
        nearest options.expected_center_mhz, or the most prominent one, and fit
        it with the selected model (lor or sinc). Publish the curve, the dashed 5%
        level, the fit and the other candidates as ``fit``. Return centre and
        FWHM in MHz with fit errors; the FWHM of a sigma peak only approximates
        the line width. The source cfg may be None. No candidate passing the
        options' thresholds raises AnalysisFailed; invalid sums raise ValueError.
        """
        # 1. Locate spectral peaks in the chain significance using the chosen peak policy.
        sums = source.result
        freqs = sums.axis
        curve = sweep_curve(sums)
        fit = fit_spectrum(
            freqs,
            curve.sigma,
            SpectrumFitOptions(
                model_type=options.model_type,
                expected_center_mhz=options.expected_center_mhz,
                min_prominence=options.min_prominence,
                min_snr=options.min_snr,
                min_width_samples=options.min_width_samples,
                min_distance_mhz=options.min_distance_mhz,
            ),
        )

        # 2. Assess the selected resonance against competing peaks and the false-mark level.
        fig, ax = plots.subplots(num="fit", figsize=config.figsize)
        ax.plot(freqs, curve.sigma, label="sigma", marker="o", markersize=3)
        ax.axhline(curve.level, color="gray", ls="--", label="5% false-mark level")
        for peak in fit.peaks:
            if peak != fit.selected_peak:
                ax.axvline(
                    peak.center,
                    color="gray",
                    ls=":",
                    label=f"candidate {peak.center:.5g} MHz",
                )
        ax.plot(fit.freqs, fit.fitted, label=f"fit, FWHM≈{fit.fwhm:.2g} MHz")
        label = f"center = {fit.freq:.5g} ± {fit.freq_err:.1g} MHz"
        ax.axvline(fit.freq, color="r", ls="--", label=label)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Distance from background (sigma)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return ChainAnalysis(
            freq=fit.freq,
            freq_err=fit.freq_err,
            fwhm=fit.fwhm,
            fwhm_err=fit.fwhm_err,
            level=curve.level,
            peaks=fit.peaks,
        )
