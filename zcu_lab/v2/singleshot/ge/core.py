from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.analysis.singleshot import (
    GECalibration,
    GEConfusionResult,
    calc_confusion_matrix,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, RunRecord
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils.single_shot.ge import (
    GEFitOptions,
    GENumericalError,
    plot_ge_iq_density,
    singleshot_ge_analysis,
)
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, PulseReadout, Reset, TimelineBuilder

from zcu_lab.v2.support.singleshot.classification_plot import (
    plot_ge_classification_boundary,
    share_ge_preparation_limits,
)
from zcu_lab.v2.support.singleshot.util import ge_signals_by_state, raw_shots_to_signal


@dataclass(frozen=True)
class GE_Result:
    """Raw IQ rows in fixed probe-off/on acquisition order."""

    signals: NDArray[np.complex128]
    shot_indices: NDArray[np.int64]


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class GEAnalyzeOptions:
    initial_state: Literal["ground", "excited"] = "ground"
    backend: Literal["pca", "center"] = "pca"
    logscale: bool = False
    align_t1: bool = True
    length_ratio: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    angle: float | None = Field(default=None, allow_inf_nan=False)


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class GEPostAnalyzeOptions:
    radius: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    consider_other: bool = False


@dataclass(frozen=True)
class GEAnalysis:
    """Prepared-label assignment and model-based GE calibration estimates.

    initial_state maps probe off/on to predominantly g/e preparation rows, not
    pure states. fidelity is the empirical equal-weight prepared-label assignment
    score at the contrast-maximizing histogram edge, not latent pure-state readout
    fidelity. theta rotates IQ by exp(1j*theta), in radians; threshold is the
    selected edge in rotated ADC units. ge_s and g_center/e_center are fitted
    width and original-frame complex centers in normalized ADC units.
    init_pops is the fitted (2,2) g/e population matrix for the preparation rows.
    length_ratio is readout length / T1 (dimensionless), or None when the data
    cannot identify it.
    fit_quality describes those histogram fits, or None when unavailable.
    Poor histogram residuals make populations and the population-corrected post
    confusion matrix unreliable even when empirical assignment is high.
    """

    initial_state: Literal["ground", "excited"]
    fidelity: float
    theta: float
    threshold: float
    ge_s: float
    g_center: complex
    e_center: complex
    init_pops: NDArray[np.float64]
    length_ratio: float | None
    fit_quality: Mapping[str, FitQuality] | None = None

    def validate_calibration(self) -> None:
        """Reject invalid physical GE calibration geometry or preparation populations."""
        populations = np.asarray(self.init_pops, dtype=np.float64)
        scalars = [
            self.fidelity,
            self.theta,
            self.threshold,
            self.ge_s,
            self.g_center,
            self.e_center,
        ]
        if (
            not np.isfinite(scalars).all()
            or not 0.0 <= self.fidelity <= 1.0
            or self.ge_s <= 0
            or np.isclose(self.g_center, self.e_center)
            or populations.shape != (2, 2)
            or not np.isfinite(populations).all()
            or np.any(populations < 0)
            or np.any(populations.sum(axis=1) > 1 + 1e-12)
        ):
            raise ValueError(
                "Invalid GE calibration: check centers, width and populations"
            )


@dataclass(frozen=True)
class GEPostAnalysis:
    confusion: GEConfusionResult


def _plot_post(
    primary: GEAnalysis,
    confusion: GEConfusionResult,
    signals: tuple[NDArray[np.complex128], NDArray[np.complex128]],
    plots: Plots,
) -> None:
    """Draw ordered g/e IQ, the fitted classifier and both Post matrices."""
    fig, _ = plots.subplots(
        num="post",
        nrows=2,
        ncols=2,
        figsize=(8, 8),
        gridspec_kw={"width_ratios": (1, 1.32)},
        layout="constrained",
    )
    g_iq, populations, e_iq, classifier = fig.axes
    empty = np.empty(0, dtype=np.complex128)
    plot_ge_iq_density(g_iq, signals[0], empty)
    plot_ge_iq_density(e_iq, empty, signals[1])
    for ax, label, color, (g, e, other) in (
        (g_iq, "g preparation", "blue", confusion.g_classification),
        (e_iq, "e preparation", "red", confusion.e_classification),
    ):
        plot_ge_classification_boundary(
            ax, primary.g_center, primary.e_center, confusion.radius
        )
        ax.set_title(f"Observed: g {g:.1%}, e {e:.1%}, Other {other:.1%}", fontsize=10)
        ax.set_xlabel("I (a.u.)", fontsize=10)
        ax.set_ylabel("Q (a.u.)", fontsize=10)
        ax.tick_params(labelsize=10)
        ax.grid(False)
        ax.legend(
            handles=[Patch(color=color, label=label)],
            loc="lower left",
            bbox_to_anchor=(1, 0),
            frameon=False,
            fontsize=10,
        )

    share_ge_preparation_limits(g_iq, e_iq)
    _plot_post_matrix(populations, confusion.init_matrix)
    populations.set_title("Initial Populations\nPrepared → actual", fontsize=10)
    _plot_post_matrix(classifier, confusion.matrix)
    classifier.set_title(
        f"Classifier confusion\nActual → measured\n"
        f"Cond. number\n{confusion.condition_number:.3g}",
        fontsize=10,
    )


def _plot_post_matrix(ax: Axes, matrix: NDArray[np.float64]) -> None:
    """Draw a 3×3 g/e/Other fraction matrix as labeled percentages on ax."""
    ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.grid(False)
    states = ("g", "e", "Other")
    ax.set_xticks([0, 1, 2], labels=states, fontsize=10)
    ax.set_yticks([0, 1, 2], labels=states, fontsize=10)
    # Leave the cell width for percentages rather than a colorbar.
    for (i, j), value in np.ndenumerate(matrix):
        percent = f"{100 * value:.1f}".removesuffix(".0")
        ax.text(
            j,
            i,
            f"{percent}%",
            fontsize=10,
            ha="center",
            va="center",
            color="white" if value > 0.5 else "black",
        )


class GE_Cfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    probe_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Probe Pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    shots: Annotated[int, FieldMeta(label="Shots", unit="count")]
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay", unit="µs", description="Delay before the first shot."
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Delay after each shot.",
        ),
    ]


class GE_Exp(Experiment[GE_Result, GE_Cfg]):
    Options: ClassVar[type[GEAnalyzeOptions]] = GEAnalyzeOptions
    PostOptions: ClassVar[type[GEPostAnalyzeOptions]] = GEPostAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[GE_Result, GE_Cfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                "shot_indices", label="Shot Index", unit="None", dtype=np.int64
            ),
            AxesSpec.Axis(
                values=(0, 1), label="Prepared State", unit="None", dtype=np.int64
            ),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u.", dtype=np.complex128),
        result_type=GE_Result,
        cfg_type=GE_Cfg,
        cfg_schema_version="3.0",
        tag="singleshot/ge",
    )

    def run(self, config: GE_Cfg, *, context: RunContext) -> GE_Result:
        """Collect probe-off/on IQ clouds for ground/excited readout calibration.

        Each shot applies optional reset and init, omits or applies the probe
        pulse, then reads out. Return complex IQ shaped (2, cfg.shots), shot
        indices, and prepared_states=[0, 1] denoting probe off/on, not guaranteed
        pure ground/excited states. Primary analysis maps these rows to physical
        G/E using its initial_state option; post-analysis derives the radius
        and confusion matrix from that fit.

        Use one round and cfg.shots reps on a copied cfg. Raw-shot conversion
        requires exactly one readout channel.
        """

        # 1. Configure raw-shot acquisition rather than averaging the IQ clouds.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(config)

        figure, axes = context.plots.subplots(num="measurement")
        probe_labels = ("Probe off", "Probe on")
        total_shots = 2 * cfg.shots
        axes.set(
            xlabel="I (a.u.)",
            ylabel="Q (a.u.)",
            title=f"GE IQ\nAcquired shots: 0 / {total_shots:,} · no branches yet",
        )
        axes.grid(False)
        axes.legend(
            handles=[
                Patch(color=color, label=label)
                for color, label in zip(("blue", "red"), probe_labels, strict=True)
            ]
        )
        empty = np.empty(0, dtype=np.complex128)
        image = plot_ge_iq_density(axes, empty, empty)

        @context.plots.on_plot_thread
        def update(clouds: NDArray[np.complex128]):
            density = plot_ge_iq_density(axes, clouds[0], clouds[1], image=image)
            counts = np.count_nonzero(np.isfinite(clouds), axis=1)
            acquired = ", ".join(
                label
                for label, count in zip(probe_labels, counts, strict=True)
                if count
            )
            title = axes.set_title(
                f"GE IQ\nAcquired shots: {counts.sum():,} / {total_shots:,}"
                f" · {acquired or 'no branches yet'}"
            )
            return density, title

        figure.show()

        signals_buffer = SignalBuffer((2, cfg.shots), on_update=update)
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for with_probe, step in sched.scan("w/o probe pulse", [False, True]):
                probe_cfg = cfg.probe_pulse if with_probe else None

                # Compare identical preparations with the probe omitted/applied.
                builder = TimelineBuilder(
                    soccfg,
                    reps=cfg.shots,
                    initial_delay=cfg.initial_delay,
                    relax_delay=cfg.relax_delay,
                )
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.emit(Pulse("init_pulse", cfg.init_pulse))
                builder.emit(Pulse("probe_pulse", probe_cfg))
                builder.emit(PulseReadout("readout", cfg.readout))
                program = builder.build()

                # Store each cloud independently, preserving every complex shot.
                shots = step.run_acquire_shots(
                    program, soc, options=RunOptions(progress=True)
                )
                if shots is None:
                    break
                signals_buffer[step].set(raw_shots_to_signal(shots, program)[0])
        signals = signals_buffer.array

        # 2. Keep acquisition order; physical G/E mapping belongs to analysis.
        return GE_Result(
            signals=signals,
            shot_indices=np.arange(signals.shape[1], dtype=np.int64),
        )

    def analyze(
        self,
        source: RunRecord[GE_Cfg, GE_Result],
        options: GEAnalyzeOptions,
        *,
        plots: Plots,
    ) -> GEAnalysis:
        """Fit readout separation and preparation populations from raw IQ clouds.

        source contains probe-off/on IQ shaped (2, N), in normalized ADC units.
        options.initial_state maps the rows to physical G/E. Return fitted
        centers, width, model populations and empirical prepared-label assignment
        fidelity, which includes preparation mixing and is not pure-state readout
        fidelity. Poor histogram fit quality makes model populations and post
        population correction unreliable. Publish raw g/e IQ scatter to plots
        as "fit", including on numerical failure, and the fitted histogram
        diagnostic as "diagnostic" on success. If either name already exists,
        raise ValueError without fitting or drawing. cfg=None is supported.

        Rotation/classification, histogram fitting, fitted population/center
        validation and fidelity calculations wrap ValueError, RuntimeError,
        ArithmeticError and numpy.linalg.LinAlgError in GENumericalError with
        the original cause. Plotting, invalid backend/initial_state and final
        calibration validation errors propagate unchanged.
        """
        if collisions := plots.keys() & {"fit", "diagnostic"}:
            raise ValueError(
                f"GE figure names already exist: {', '.join(sorted(collisions))}"
            )

        # 1. Acquisition order is always probe off/on; fit in physical g/e order.
        signals = ge_signals_by_state(source.result.signals, options.initial_state)
        _, axes = plots.subplots(num="fit", layout="constrained")
        plot_ge_iq_density(axes, signals[0], signals[1])
        axes.set(xlabel="I (a.u.)", ylabel="Q (a.u.)")
        # Reserve space between the IQ box and its title even for narrow I ranges.
        axes.set_title("GE IQ", pad=36)
        axes.grid(False)
        axes.legend(
            handles=[
                Patch(color="blue", label="g preparation"),
                Patch(color="red", label="e preparation"),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 1),
            borderaxespad=1,
            ncols=2,
            fontsize=10,
        )
        # 2. Fit readout separation and preparation mixing in physical g/e order.
        fidelity, init_pops, fit = singleshot_ge_analysis(
            signals,
            angle=options.angle,
            backend=options.backend,
            options=GEFitOptions(
                logscale=options.logscale,
                align_t1=options.align_t1,
                length_ratio=options.length_ratio,
                raw_iq_diagnostic=True,
            ),
            plots=plots,
            figure_name="diagnostic",
        )
        # 3. Validate the calibration; label the fit independently of the reporting limit.
        analysis = GEAnalysis(
            initial_state=options.initial_state,
            fidelity=fidelity,
            theta=fit["theta"],
            threshold=fit["threshold"],
            ge_s=fit["s"],
            g_center=fit["g_center"],
            e_center=fit["e_center"],
            init_pops=init_pops,
            length_ratio=fit["length_ratio"],
            fit_quality=fit["fit_quality"],
        )
        analysis.validate_calibration()
        error = fit["length_ratio_standard_error"]
        if options.length_ratio is not None:
            label = f"length ratio = {options.length_ratio:.3g}"
        elif error is not None:
            label = f"length ratio = {fit['ge_params'][-1]:.3g} ± {error:.3g}"
        else:
            label = "length ratio: unavailable"
        axes.title.set_text(f"GE IQ ({label})")
        return analysis

    def post_analyze(
        self,
        source: RunRecord[GE_Cfg, GE_Result],
        primary: GEAnalysis,
        options: GEPostAnalyzeOptions,
        *,
        plots: Plots,
    ) -> GEPostAnalysis:
        """Derive classification radius and confusion matrix without refitting IQ.

        source contains probe-off/on IQ shaped (2, N), in normalized ADC units.
        primary supplies physical G/E mapping and fitted populations/centers.
        options sets the radius policy, in the same IQ units. Return the
        classification diagnostic and publish it as "post" in plots.
        Classification uses nearest-center regions and radius, not the primary
        histogram threshold. Solving the fitted preparation matrix removes its
        inferred mixing; poor primary histogram fits make this correction
        unreliable. It is not the empirical prepared-label assignment score.
        cfg=None is supported.

        Radius optimization, classification and confusion-matrix solving wrap
        ValueError, RuntimeError, ArithmeticError and numpy.linalg.LinAlgError
        in GENumericalError with the original cause. Primary validation, row
        mapping and plotting errors, including name conflicts, propagate unchanged.
        """
        # 1. Select the measured data and analysis controls.
        primary.validate_calibration()
        g_signals, e_signals = ge_signals_by_state(
            source.result.signals, primary.initial_state
        )
        signals = (g_signals, e_signals)
        try:
            confusion = calc_confusion_matrix(
                signals,
                GECalibration(
                    g_center=primary.g_center,
                    e_center=primary.e_center,
                    init_pops=primary.init_pops,
                    sigma=primary.ge_s,
                ),
                radius=options.radius,
                consider_other=options.consider_other,
            )
        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ) as exc:
            raise GENumericalError(str(exc)) from exc
        # 2. Keep preparation color separate from observed and corrected classification.
        _plot_post(primary, confusion, signals, plots)
        return GEPostAnalysis(confusion=confusion)
