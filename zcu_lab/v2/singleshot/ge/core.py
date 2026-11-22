from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.analysis.singleshot import (
    GECalibration,
    GEConfusionResult,
    calc_confusion_matrix,
    classify_result,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, RunRecord
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils.single_shot.ge import (
    GEFitOptions,
    GENumericalError,
    singleshot_ge_analysis,
)
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, PulseReadout, Reset, TimelineBuilder

from zcu_lab.v2.support.singleshot.center_quality import ge_centers_identifiable
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


def _plot_classified_iq(
    ax: Axes,
    signals: NDArray[np.complex128],
    g_center: complex,
    e_center: complex,
    radius: float,
    max_point: int = 5000,
) -> None:
    # Classify shots
    """Draw classified IQ points, centers and clipped circle boundaries on ax.

    signals is a nonempty 1D complex shot vector. g_center/e_center and radius
    follow analysis.singleshot.classify_result; max_point limits downsampling
    and must be positive. Mutates ax, adds labels/legend/equal aspect, and
    returns None. Invalid classification geometry raises ValueError; Matplotlib
    errors propagate. The caller owns the axes and figure lifecycle."""
    mask_g, mask_e, mask_o = classify_result(signals, g_center, e_center, radius)

    colors = np.full(signals.shape, "gray", dtype=object)
    colors[mask_g] = "blue"
    colors[mask_e] = "red"
    colors[mask_o] = "green"

    # plot shots
    num_sample = signals.shape[0]
    downsample_num = min(max_point, num_sample)
    downsample_mask = np.arange(0, num_sample, max(1, num_sample // downsample_num))
    downsample_signals = signals[downsample_mask]
    ax.scatter(
        downsample_signals.real,
        downsample_signals.imag,
        c=colors[downsample_mask].tolist(),
        s=1,
        alpha=0.5,
    )

    # plot centers with circle
    ax.plot(
        g_center.real,
        g_center.imag,
        markerfacecolor="b",
        label="Ground",
        linestyle=":",
        color="k",
        marker="o",
        markersize=5,
    )
    ax.plot(
        e_center.real,
        e_center.imag,
        markerfacecolor="r",
        label="Excited",
        linestyle=":",
        color="k",
        marker="o",
        markersize=5,
    )
    separation = abs(e_center - g_center)
    axis = (e_center - g_center) / separation
    cut_angle = np.arccos(separation / (2 * radius)) if radius > separation / 2 else 0.0
    angles = np.linspace(cut_angle, 2 * np.pi - cut_angle, 361)
    for center, direction, color in ((g_center, axis, "b"), (e_center, -axis, "r")):
        boundary = center + direction * radius * np.exp(1j * angles)
        ax.add_patch(
            Polygon(
                np.column_stack((boundary.real, boundary.imag)),
                closed=True,
                color=color,
                fill=False,
                linestyle="--",
            )
        )

    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlabel("I value (a.u.)")
    ax.set_ylabel("Q value (a.u.)")


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

        @context.plots.on_plot_thread
        def update(clouds: NDArray[np.complex128]):
            """Draw acquired raw complex IQ on the measurement axes; return changed artists.

            Read scan-shaped samples without changing them. Update the live
            artists over the prepared coordinates; plot errors propagate.
            """
            axes.clear()
            for cloud, label, color in zip(
                clouds,
                ("g preparation (probe off)", "e preparation (probe on)"),
                ("tab:blue", "tab:red"),
                strict=True,
            ):
                axes.scatter(
                    cloud.real, cloud.imag, s=8, alpha=0.5, color=color, label=label
                )
            axes.set(xlabel="I (a.u.)", ylabel="Q (a.u.)", title="GE IQ")
            axes.set_aspect("equal", adjustable="datalim")
            axes.grid()
            axes.legend()
            return (figure,)

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
        _, axes = plots.subplots(num="fit")
        for cloud, label, color in zip(
            signals,
            ("g preparation", "e preparation"),
            ("tab:blue", "tab:red"),
            strict=True,
        ):
            axes.scatter(
                cloud.real, cloud.imag, s=8, alpha=0.5, color=color, label=label
            )
        axes.set(xlabel="I (a.u.)", ylabel="Q (a.u.)", title="GE IQ")
        axes.set_aspect("equal", adjustable="datalim")
        axes.grid()
        axes.legend()
        # 2. Fit readout separation and preparation mixing in physical g/e order.
        fidelity, init_pops, fit = singleshot_ge_analysis(
            signals,
            angle=options.angle,
            backend=options.backend,
            options=GEFitOptions(
                logscale=options.logscale,
                align_t1=options.align_t1,
                length_ratio=options.length_ratio,
                centers_identifiable=ge_centers_identifiable,
            ),
            plots=plots,
            figure_name="diagnostic",
        )
        # 3. Validate the calibration and label an unidentifiable length ratio.
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
        axes.set_title(
            "GE IQ (length ratio: not identifiable)"
            if analysis.length_ratio is None
            else f"GE IQ (length ratio = {analysis.length_ratio:.2f})"
        )
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
        # 2. Display preparation mixing alongside classified IQ and confusion.
        g_signals, e_signals = signals
        g_center, e_center = primary.g_center, primary.e_center
        fig, _ = plots.subplots(
            num="post", nrows=2, ncols=2, figsize=(8, 8), layout="constrained"
        )
        ax1, ax4, ax2, ax3 = fig.axes

        g_label = r"$|0\rangle$"
        e_label = r"$|1\rangle$"
        l_label = r"$|L\rangle$"
        _plot_classified_iq(ax1, g_signals, g_center, e_center, confusion.radius)
        _plot_classified_iq(ax2, e_signals, g_center, e_center, confusion.radius)

        n_gg, n_ge, n_go = confusion.g_classification
        n_eg, n_ee, n_eo = confusion.e_classification
        ax1.set_title(
            f"{g_label}: {n_gg:.1%}, {e_label}: {n_ge:.1%}, {l_label}: {n_go:.1%}"
        )
        ax2.set_title(
            f"{g_label}: {n_eg:.1%}, {e_label}: {n_ee:.1%}, {l_label}: {n_eo:.1%}"
        )
        ax1.set_xlabel("")

        im = ax4.imshow(confusion.init_matrix, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, cax=ax4.inset_axes((1.04, 0, 0.06, 1)))
        for i in range(confusion.init_matrix.shape[0]):
            for j in range(confusion.init_matrix.shape[1]):
                val = confusion.init_matrix[i, j]
                ax4.text(
                    j,
                    i,
                    f"{val:.2%}",
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="white" if val > 0.5 else "black",
                )

        ax4.set_xticks([0, 1, 2])
        ax4.set_yticks([0, 1, 2])
        ax4.set_xticklabels([g_label, e_label, l_label])
        ax4.set_yticklabels([g_label, e_label, l_label])
        ax4.set_xlabel("Actual State")
        ax4.set_ylabel("Prepared State")
        ax4.set_title("Initial Populations")

        im = ax3.imshow(confusion.matrix, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, cax=ax3.inset_axes((1.04, 0, 0.06, 1)))
        for i in range(confusion.matrix.shape[0]):
            for j in range(confusion.matrix.shape[1]):
                val = confusion.matrix[i, j]
                ax3.text(
                    j,
                    i,
                    f"{val:.2%}",
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="white" if val > 0.5 else "black",
                )

        ax3.set_xticks([0, 1, 2])
        ax3.set_yticks([0, 1, 2])
        ax3.set_xticklabels([g_label, e_label, l_label])
        ax3.set_yticklabels([g_label, e_label, l_label])
        ax3.set_xlabel("Measured State")
        ax3.set_ylabel("Actual State")
        ax3.set_title(f"Confusion Matrix (cond: {confusion.condition_number:.1f})")
        return GEPostAnalysis(confusion=confusion)
