"""Single-carrier, two-photon g/f scans retaining classified shot counts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
from zcu_tools.analysis.singleshot import (
    Centroid3IQAnalysis,
    Centroid3Summary,
    analyze_centroid3_populations,
    summarize_centroid3,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import (
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import round_zcu_freq, round_zcu_gain
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    PulseReadout,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules.readout import centroid3_classifier

from zcu_lab.v2.support import live_errorbar, live_heatmap

STATE_LABELS = ("g", "e", "f")


@dataclass(frozen=True)
class RamanGFResult:
    """Classified shot counts, with no simulator state information.

    counts: (points,3) g/e/f counts in detuning-major, gain-minor order.
    detunings: (points,) actual two-photon detuning in MHz, positive above center.
    gains: (points,) actual DAC gains in full-scale fraction (a.u.).
    Partial/unacquired rows remain NaN. No calibration or state truth is acquired.
    """

    counts: NDArray[np.float64]
    detunings: NDArray[np.float64]
    gains: NDArray[np.float64]


class RamanGFCfg(ConfigBase):
    """One charge-drive carrier with any supported pulse waveform.

    carrier_mhz is operator-supplied (f03-f32)/2, not a model frequency.
    detuning = 2*f_drive - 2*carrier_mhz, in MHz (cycles/us, not angular units).
    Frequency mode uses gain.start; gain mode uses detuning.start; grid uses both.
    Gain is a DAC full-scale fraction, not a calibrated rotation angle.
    The drive's freq/gain are supplied by the scan; its channel, NQZ, waveform,
    phase and delays remain operator inputs. Calibration belongs to saved readout.
    """

    drive: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Raman drive",
            description="Fill in the Raman pulse; the scan supplies its RF frequency and gain.",
        ),
    ]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]

    carrier_mhz: Annotated[
        float,
        FieldMeta(
            label="Carrier center",
            unit="MHz",
            description="Fill in the measured Raman RF carrier (MHz), half the difference between the 0–3 and 3–2 transition frequencies.",
        ),
    ] = Field(gt=0, allow_inf_nan=False)
    scan_mode: Annotated[
        Literal["frequency", "gain", "frequency_gain"],
        FieldMeta(
            label="Scan mode",
            description="Choose frequency at the starting gain, gain at the starting detuning, or a grid of both.",
        ),
    ] = "frequency_gain"
    detuning: Annotated[SweepCfg, FieldMeta(label="Two-photon detuning", unit="MHz")]
    gain: Annotated[SweepCfg, FieldMeta(label="Drive gain", unit="a.u.")]
    shots: Annotated[int, FieldMeta(label="Shots per point", unit="count")] = Field(
        default=1024, ge=6
    )
    relax_delay: Annotated[float, FieldMeta(label="Relax delay", unit="µs")] = Field(
        ge=0
    )
    initial_delay: Annotated[float, FieldMeta(label="Initial delay", unit="µs")] = (
        Field(default=1, ge=0)
    )


def restore_result(data: GroupedLoadData[RamanGFCfg]) -> RamanGFResult:
    """Restore g/e/f counts and actual coordinates; missing counts raises ValueError."""
    return RamanGFResult(
        counts=np.asarray(data.variable("counts").z, dtype=np.float64),
        detunings=np.asarray(data.variable("detuning").z, dtype=np.float64),
        gains=np.asarray(data.variable("gain").z, dtype=np.float64),
    )


class RamanGFExp(Experiment[RamanGFResult, RamanGFCfg]):
    """Scan a single-tone two-photon g/f drive and retain g/e/f counts."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[RamanGFResult, RamanGFCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                "counts",
                (
                    VariableAxisSpec.generated_arange(label="State", unit="None"),
                    VariableAxisSpec.generated_arange(label="Point", unit="None"),
                ),
                VariableZSpec(
                    "counts", label="Shot count", unit="None", dtype=np.float64
                ),
            ),
            *(
                VariableSpec(
                    name,
                    (VariableAxisSpec.generated_arange(label="Point", unit="None"),),
                    VariableZSpec(field, label=label, unit=unit, dtype=np.float64),
                )
                for name, field, label, unit in (
                    ("detuning", "detunings", "Two-photon detuning", "MHz"),
                    ("gain", "gains", "Drive gain", "a.u."),
                )
            ),
        ),
        result_type=RamanGFResult,
        cfg_type=RamanGFCfg,
        result_builder=restore_result,
        tag="twotone/raman_gf",
        cfg_schema_version="4.0",
        labber_layout="independent",
    )

    def run(self, cfg: RamanGFCfg, *, context: RunContext) -> RamanGFResult:
        """Acquire one Raman pulse and readout per scan point, without references.

        Positive two-photon detuning raises the carrier by detuning/2 MHz.
        Each point starts after relaxation; there is no Lambda pair or reset.
        Classify cfg.shots repetitions in one round into g/e/f counts per point.
        Require a centroid3 readout; any supported waveform can drive the scan.
        Unfinished points retain NaN counts. Gains are actual DAC settings,
        not a calibrated rotation angle.
        """
        # 1. Require calibrated g/e/f classification before planning the Raman scan.
        centroid3_classifier(cfg.readout, ro_ch=cfg.readout.ro_ch)
        cfg = deepcopy(cfg)
        detunings = np.linspace(
            cfg.detuning.start, cfg.detuning.stop, cfg.detuning.expts
        )
        gains = np.linspace(cfg.gain.start, cfg.gain.stop, cfg.gain.expts)
        # 2. Pair each programmed gain with its carrier in scan order.
        if cfg.scan_mode == "gain":
            detunings = detunings[:1]
        elif cfg.scan_mode == "frequency":
            gains = gains[:1]
        frequencies = round_zcu_freq(
            cfg.carrier_mhz + detunings / 2,
            context.soccfg,
            cfg.drive.ch,
            mixer_freq=cfg.drive.mixer_freq,
        )
        gains = round_zcu_gain(gains, context.soccfg, cfg.drive.ch)
        point_frequencies = np.repeat(frequencies, len(gains))
        point_gains = np.tile(gains, len(frequencies))
        actual_detunings = 2 * (frequencies - cfg.carrier_mhz)
        counts = self._live_counts(context.plots, cfg, actual_detunings, gains)

        # 3. Measure Raman preparation labels for each carrier-gain pair with a fixed envelope.
        with Schedule(counts, scope=context.scope) as sched:
            for point, step in sched.scan("point", range(len(point_gains))):
                builder = TimelineBuilder(
                    context.soccfg,
                    reps=cfg.shots,
                    initial_delay=cfg.initial_delay,
                    relax_delay=cfg.relax_delay,
                )
                builder.emit(
                    Pulse(
                        "drive",
                        cfg.drive,
                        freq=float(point_frequencies[point]),
                        gain=float(point_gains[point]),
                    )
                )
                builder.emit(PulseReadout("readout", cfg.readout))
                program = builder.build()
                step.run_acquire_summed(
                    program,
                    context.soc,
                    reducer=shot_sums(program, [Classify("readout")]),
                    options=RunOptions(
                        converter=lambda sums: sums.reshape(len(STATE_LABELS))
                    ),
                )

        return RamanGFResult(
            counts=counts.array,
            detunings=np.repeat(actual_detunings, len(gains)),
            gains=point_gains,
        )

    def _live_counts(
        self,
        plots: Plots,
        cfg: RamanGFCfg,
        detunings: NDArray[np.float64],
        gains: NDArray[np.float64],
    ) -> SignalBuffer:
        """Attach measured class fractions to actual coordinates as points arrive."""
        grid = cfg.scan_mode == "frequency_gain"
        figure, _ = plots.subplots(
            num="measurement",
            ncols=3 if grid else 1,
            figsize=(11, 4.8) if grid else (7, 5),
            layout="constrained",
        )
        title = "Raman spectroscopy · g → f · measured fractions"
        headline = figure.suptitle(title)
        shape = (len(detunings), len(gains))
        if grid:
            maps = [
                live_heatmap(ax, uniform=False, clim=(0, 1), cmap="viridis")
                for ax in figure.axes
            ]
            for ax, heatmap, state in zip(figure.axes, maps, STATE_LABELS, strict=True):
                ax.set(
                    xlabel="Drive gain (a.u.)",
                    ylabel="Two-photon detuning (MHz)",
                    title=f"{state} label",
                )
                ax.grid(False)
                (image,) = heatmap.set(gains, detunings, np.full(shape[::-1], np.nan))
                figure.colorbar(
                    image, ax=ax, label="Measured label fraction", ticks=(0, 0.5, 1)
                )

            @plots.on_plot_thread
            def update_grid(data: NDArray[np.float64]) -> tuple[Artist, ...]:
                totals = data.sum(axis=-1, keepdims=True)
                fractions = np.divide(
                    data, totals, out=np.full_like(data, np.nan), where=totals > 0
                )
                headline.set_text(
                    f"{title}\nRetained shots: {int(np.nansum(totals)):,} count"
                )
                return (
                    headline,
                    *(
                        artist
                        for index, heatmap in enumerate(maps)
                        for artist in heatmap.set(
                            gains, detunings, fractions[:, index].reshape(shape).T
                        )
                    ),
                )

            update = update_grid
        else:
            ax = figure.axes[0]
            frequency = cfg.scan_mode == "frequency"
            xs = detunings if frequency else gains
            ax.set(
                xlabel="Two-photon detuning (MHz)"
                if frequency
                else "Drive gain (a.u.)",
                ylabel="Measured label fraction",
            )
            lines = live_errorbar(ax, num_lines=len(STATE_LABELS))
            for line, state in zip(ax.lines, STATE_LABELS, strict=True):
                line.set(
                    marker="o", markersize=3, linestyle="none", label=f"{state} label"
                )
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=3)

            @plots.on_plot_thread
            def update_trace(data: NDArray[np.float64]) -> tuple[Artist, ...]:
                totals = data.sum(axis=-1, keepdims=True)
                fractions = np.divide(
                    data, totals, out=np.full_like(data, np.nan), where=totals > 0
                )
                covariance = class_probability_covariance(data)
                sem = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
                headline.set_text(
                    f"{title}\nRetained shots: {int(np.nansum(totals)):,} count"
                )
                return (headline, *lines.set(xs, fractions.T, sem.T))

            update = update_trace
        figure.show()
        return SignalBuffer(
            (shape[0] * shape[1], len(STATE_LABELS)), dtype=np.float64, on_update=update
        )

    def analyze(
        self,
        source: RunRecord[RamanGFCfg, RamanGFResult],
        options: None,
        *,
        plots: Plots,
    ) -> Centroid3Summary:
        """Normalize acquired counts and correct with the saved readout confusion.

        Require source.cfg and its centroid3 ADC classifier. Missing cfg,
        missing classifier or circle/linear mode raises ValueError. No live
        Form, library, reference acquisition or simulator truth is consulted.
        Plot raw/unclipped corrected g/e/f preparation-label estimates and
        return only training confusion and its quality. The e label estimates leakage, not a pure
        energy-state population or gate fidelity. Incomplete/nonfinite or empty
        counts and singular confusion fail explicitly. Plot actual coordinates.
        """
        # 1. Recover g/e/f populations from counts using the saved readout confusion calibration.
        del options
        cfg = source.cfg
        if cfg is None:
            raise ValueError("Raman analysis requires the saved run readout cfg")
        calibration = centroid3_classifier(cfg.readout, ro_ch=cfg.readout.ro_ch)
        counts = source.result.counts
        if not np.isfinite(counts).all() or np.any(counts.sum(axis=-1) <= 0):
            raise ValueError(
                "Raman analysis requires finite, nonempty counts at every point"
            )
        raw = counts / counts.sum(axis=-1, keepdims=True)
        result = analyze_centroid3_populations(raw, calibration)
        grid = cfg.scan_mode == "frequency_gain"
        # 2. Compare raw and corrected populations along the scanned gain or detuning axes.
        figure, _ = plots.subplots(
            num="fit",
            nrows=2 if grid else 1,
            ncols=len(STATE_LABELS) if grid else 2,
            figsize=(11, 7) if grid else (10, 5),
            layout="constrained",
        )

        if grid:
            self._draw_maps(figure, source.result, cfg, result)
        else:
            self._draw_traces(figure, source.result, cfg, result)
        figure.suptitle(
            "Raman spectroscopy · g → f · preparation-label fractions\n"
            f"Readout condition number: {result.condition_number:.3g} · "
            f"Minimum singular value: {result.minimum_singular_value:.3g}"
        )
        figure.show()
        return summarize_centroid3(result)

    def _draw_maps(
        self,
        figure: Figure,
        data: RamanGFResult,
        cfg: RamanGFCfg,
        result: Centroid3IQAnalysis,
    ) -> None:
        """Draw measured and corrected label fractions on the same scan axes."""
        shape = (cfg.detuning.expts, cfg.gain.expts)
        detunings = data.detunings.reshape(shape)[:, 0]
        gains = data.gains.reshape(shape)[0, :]
        corrected = result.corrected_populations
        for row, (values, title, limits) in enumerate(
            (
                (result.raw_populations, "Measured", (0.0, 1.0)),
                (
                    corrected,
                    "Corrected",
                    (
                        min(0.0, float(corrected.min())),
                        max(1.0, float(corrected.max())),
                    ),
                ),
            )
        ):
            for label, state in enumerate(STATE_LABELS):
                ax = figure.axes[row * len(STATE_LABELS) + label]
                ax.grid(False)
                image = ax.pcolormesh(
                    gains,
                    detunings,
                    values[:, label].reshape(shape),
                    shading="nearest",
                    cmap="viridis",
                    vmin=limits[0],
                    vmax=limits[1],
                )
                ax.set(
                    xlabel="Drive gain (a.u.)",
                    ylabel="Two-photon detuning (MHz)",
                    title=f"{title} {state} label",
                )
                figure.colorbar(image, ax=ax, label=f"{title} label fraction")

    def _draw_traces(
        self,
        figure: Figure,
        data: RamanGFResult,
        cfg: RamanGFCfg,
        result: Centroid3IQAnalysis,
    ) -> None:
        """Draw one-dimensional scans with shot SEM before and after correction."""
        frequency = cfg.scan_mode == "frequency"
        xs = data.detunings if frequency else data.gains
        xlabel = "Two-photon detuning (MHz)" if frequency else "Drive gain (a.u.)"
        measured_cov = class_probability_covariance(data.counts)
        correction = np.linalg.inv(result.confusion_matrix.T)
        corrected_cov = correction @ measured_cov @ correction.T
        for ax, values, covariance, label in zip(
            figure.axes,
            (result.raw_populations, result.corrected_populations),
            (measured_cov, corrected_cov),
            ("Measured", "Corrected"),
            strict=True,
        ):
            sem = np.sqrt(np.maximum(np.diagonal(covariance, axis1=-2, axis2=-1), 0.0))
            for index, state in enumerate(STATE_LABELS):
                ax.errorbar(
                    xs,
                    values[:, index],
                    yerr=sem[:, index],
                    fmt="o",
                    color=f"C{index}",
                    markersize=3,
                    label=f"{state} label",
                )
            ax.set(xlabel=xlabel, ylabel=f"{label} label fraction")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3)
