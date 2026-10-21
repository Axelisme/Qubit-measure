from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal, NamedTuple, cast

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.analysis.singleshot import classify_result
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    RunRecord,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils.single_shot.ge import (
    GEFitOptions,
    GENumericalError,
    singleshot_ge_analysis,
)
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.plotting.singleshot import plot_with_classified
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
)
from zcu_tools.utils.shot_classification import gaussian_region_probability

from zcu_lab.v2._support.singleshot.plot import draw_ge_scatter
from zcu_lab.v2._support.singleshot.util import raw_shots_to_signal

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------


def make_init_matrix(init_pops: NDArray[np.float64]) -> NDArray[np.float64]:
    # Each acquisition has its own refined preparation populations. Retaining
    # both rows makes the diagnostic equivariant under g/e relabeling.
    if (
        init_pops.shape != (2, 2)
        or not np.isfinite(init_pops).all()
        or np.any(init_pops < 0)
        or np.any(init_pops.sum(axis=1) > 1 + 1e-12)
    ):
        raise ValueError("GE initial populations must be a valid 2x2 probability array")
    p_gg_init, p_ge_init = init_pops[0]
    p_eg_init, p_ee_init = init_pops[1]
    p_go_init = max(0.0, 1.0 - p_gg_init - p_ge_init)
    p_eo_init = max(0.0, 1.0 - p_eg_init - p_ee_init)

    return np.array(
        [
            [p_gg_init, p_ge_init, p_go_init],
            [p_eg_init, p_ee_init, p_eo_init],
            [0.0, 0.0, 1.0],  # assume all initial to O
        ]
    )


class GEClassification(NamedTuple):
    """One prepared state's fractions classified as ground, excited and other."""

    ground: float
    excited: float
    other: float


ISOLATED_OTHER_CLASSIFICATION = GEClassification(0.0, 0.0, 1.0)


def make_result_matrix(
    ground: GEClassification,
    excited: GEClassification,
    other: GEClassification = ISOLATED_OTHER_CLASSIFICATION,
) -> NDArray[np.float64]:
    """Stack ground/excited/other preparation rows into a 3x3 result matrix.

    Columns follow GEClassification order. The default other-state row models
    perfectly isolated classification. Values are preserved without normalization.
    """
    return np.array([ground, excited, other])


def solve_confusion_matrix(
    A_init: NDArray[np.float64],
    Q: NDArray[np.float64],
) -> NDArray[np.float64]:
    confusion_matrix = np.linalg.solve(A_init, Q)
    confusion_matrix = np.clip(confusion_matrix, 0.0, None)
    confusion_matrix /= confusion_matrix.sum(axis=1, keepdims=True)

    return confusion_matrix


# tolerated: One caller supplies shot arrays, fit calibration and the other-state penalty switch.
def optimize_ge_radius(  # noqa: PLR0913
    g_signals: NDArray[np.complex128],
    e_signals: NDArray[np.complex128],
    g_center: complex,
    e_center: complex,
    init_pops: NDArray[np.float64],
    sigma: float,
    *,
    consider_other: bool = True,
) -> float:
    from scipy.optimize import OptimizeResult, minimize_scalar

    # Validate geometry even before the optimizer evaluates its first candidate.
    classify_result(np.empty(0, dtype=np.complex128), g_center, e_center, 0.0)
    ge_dist = abs(g_center - e_center)
    A_init = make_init_matrix(init_pops)

    def loss_fn(radius: float) -> float:

        gg_mask, ge_mask, go_mask = classify_result(
            g_signals, g_center, e_center, radius
        )
        n_gg = gg_mask.sum() / gg_mask.shape[0]
        n_ge = ge_mask.sum() / ge_mask.shape[0]
        n_go = go_mask.sum() / go_mask.shape[0]

        eg_mask, ee_mask, eo_mask = classify_result(
            e_signals, g_center, e_center, radius
        )
        n_eg = eg_mask.sum() / eg_mask.shape[0]
        n_ee = ee_mask.sum() / ee_mask.shape[0]
        n_eo = eo_mask.sum() / eo_mask.shape[0]

        # Radius-selection penalty only: model other at the midpoint.
        # The returned calibration retains the isolated-other row.
        n_og = (
            float(
                gaussian_region_probability(
                    np.array([ge_dist / 2]), sigma, radius, ge_dist
                )[0]
            )
            if consider_other
            else 0.0
        )
        n_oe = n_og
        n_oo = 1.0 - n_og - n_oe

        Q = make_result_matrix(
            GEClassification(n_gg, n_ge, n_go),
            GEClassification(n_eg, n_ee, n_eo),
            GEClassification(n_og, n_oe, n_oo),
        )
        confusion_matrix = solve_confusion_matrix(A_init, Q)

        # calculate condision number of confusion matrix as loss
        condition = float(np.linalg.cond(confusion_matrix))
        return condition if np.isfinite(condition) else 1e12

    result = minimize_scalar(loss_fn, bounds=(0.0, ge_dist), method="bounded")
    if not isinstance(result, OptimizeResult):
        raise TypeError("classification-radius optimizer must return OptimizeResult")
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError("classification-radius optimization failed")
    # The optimum can be the upper bound; bounded minimization excludes endpoints.
    return min((float(result.x), ge_dist), key=loss_fn)


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------


def _default_prepared_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


def ge_signals_by_state(
    signals: NDArray[np.complex128],
    initial_state: Literal["ground", "excited"],
) -> NDArray[np.complex128]:
    """Map acquisition rows (probe off/on) to predominantly ground/excited."""
    if initial_state == "ground":
        return signals
    if initial_state == "excited":
        return signals[::-1]
    raise ValueError(f"Unknown initial state: {initial_state!r}")


@dataclass(frozen=True)
class GE_Result:
    """Raw probe-off/on rows; prepared_states=[0, 1] records that acquisition order."""

    signals: NDArray[np.complex128]
    shot_indices: NDArray[np.int64]
    prepared_states: NDArray[np.int64]


@dataclass(frozen=True)
class GEAnalyzeOptions:
    initial_state: Literal["ground", "excited"] = "ground"
    backend: Literal["pca", "center"] = "pca"
    logscale: bool = False
    align_t1: bool = True
    length_ratio: float | None = None
    angle: float | None = None


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
    fit_quality: Mapping[str, FitQuality] | None = None

    def validate_calibration(self) -> None:
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
class GEPostAnalyzeOptions:
    radius: float | None = None
    consider_other: bool = False


@dataclass(frozen=True)
class GEConfusionResult:
    radius: float
    matrix: NDArray[np.float64]
    init_matrix: NDArray[np.float64]
    g_classification: GEClassification
    e_classification: GEClassification
    condition_number: float


@dataclass(frozen=True)
class GEPostAnalysis:
    confusion: GEConfusionResult


class GEModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[PulseCfg | None, FieldMeta(label="Init Pulse")] = None
    probe_pulse: Annotated[PulseCfg, FieldMeta(label="Probe Pulse")]
    readout: Annotated[PulseReadoutCfg, FieldMeta(label="Readout")]


class GE_Cfg(ExpCfgModel):
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay", unit="us", description="Delay before the first shot."
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Delay after each shot.",
        ),
    ] = 1.0
    modules: GEModuleCfg
    shots: Annotated[int, FieldMeta(label="Shots")]


class GE_Exp(PersistableExperiment[GE_Result, GE_Cfg]):
    Options: ClassVar[type[GEAnalyzeOptions]] = GEAnalyzeOptions
    PostOptions: ClassVar[type[GEPostAnalyzeOptions]] = GEPostAnalyzeOptions

    AXES_SPEC = AxesSpec(
        axes=(
            Axis(
                "shot_indices",
                "Shot Index",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
            Axis(
                "prepared_states",
                "Prepared State",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
        ),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
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

        # Configure raw-shot acquisition rather than averaging the IQ clouds.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(config)
        program_cfg = ProgramV2Cfg(
            reps=cfg.shots,
            rounds=1,
            initial_delay=cfg.initial_delay,
            relax_delay=cfg.relax_delay,
        )

        signals_buffer = SignalBuffer(
            (2, cfg.shots),
            on_update=lambda data: draw_ge_scatter(context.plots, "measurement", data),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for with_probe, step in sched.scan("w/o probe pulse", [False, True]):
                modules = step.cfg.modules
                probe_cfg = modules.probe_pulse if with_probe else None

                # Compare identical preparations with the probe omitted/applied.
                program = (
                    step.prog_builder(soc, soccfg, cfg=program_cfg)
                    .add(
                        Reset("reset", modules.reset),
                        Pulse("init_pulse", modules.init_pulse),
                        Pulse("probe_pulse", probe_cfg),
                        PulseReadout("readout", modules.readout),
                    )
                    .build()
                )

                # Store each cloud independently, preserving every complex shot.
                try:
                    program.acquire(soc, progress=True, cancel_flag=step.stop)
                except StoppedPartialAcquireError:
                    step.set_stop()
                    break
                signals_buffer[step].set(raw_shots_to_signal(program))
        signals_buffer.trigger_update(flush=True)
        signals = signals_buffer.array

        # Keep acquisition order; physical G/E mapping belongs to analysis.
        return GE_Result(
            signals=signals,
            shot_indices=np.arange(signals.shape[1], dtype=np.int64),
            prepared_states=_default_prepared_states(),
        )

    def analyze(
        self,
        source: RunRecord[GE_Cfg, GE_Result],
        options: GEAnalyzeOptions,
        *,
        plots: Plots,
        figure_name: str = "fit",
    ) -> GEAnalysis:
        """Fit readout separation and preparation populations from raw IQ clouds.

        source contains probe-off/on IQ shaped (2, N), in normalized ADC units.
        options.initial_state maps the rows to physical G/E. Return fitted
        centers, width, model populations and empirical prepared-label assignment
        fidelity, which includes preparation mixing and is not pure-state readout
        fidelity. Poor histogram fit quality makes model populations and post
        population correction unreliable. Publish raw g/e IQ scatter to plots
        as figure_name, including on numerical failure. Histogram fitting stays
        internal. cfg=None is supported.

        Rotation/classification, histogram fitting, fitted population/center
        validation and fidelity calculations wrap ValueError, RuntimeError,
        ArithmeticError and numpy.linalg.LinAlgError in GENumericalError with
        the original cause. Plotting, invalid backend/initial_state and final
        calibration validation errors propagate unchanged.
        """
        # Acquisition order is always probe off/on; fit in physical g/e order.
        signals = ge_signals_by_state(source.result.signals, options.initial_state)
        if options.backend not in ("pca", "center"):
            raise ValueError(f"Unknown backend: {options.backend}")
        if figure_name in plots:
            raise ValueError(f"Figure already exists: {figure_name}")
        draw_ge_scatter(
            plots, figure_name, signals, labels=("g preparation", "e preparation")
        )
        fidelity, init_pops, fit = singleshot_ge_analysis(
            signals,
            angle=options.angle,
            backend=options.backend,
            options=GEFitOptions(
                logscale=options.logscale,
                align_t1=options.align_t1,
                length_ratio=options.length_ratio,
            ),
            plots=None,
        )
        analysis = GEAnalysis(
            initial_state=options.initial_state,
            fidelity=fidelity,
            theta=fit["theta"],
            threshold=fit["threshold"],
            ge_s=fit["s"],
            g_center=fit["g_center"],
            e_center=fit["e_center"],
            init_pops=init_pops,
            fit_quality=fit["fit_quality"],
        )
        analysis.validate_calibration()
        return analysis

    def post_analyze(
        self,
        source: RunRecord[GE_Cfg, GE_Result],
        primary: GEAnalysis,
        options: GEPostAnalyzeOptions,
        *,
        plots: Plots,
        figure_name: str = "post",
    ) -> GEPostAnalysis:
        """Derive classification radius and confusion matrix without refitting IQ.

        source contains probe-off/on IQ shaped (2, N), in normalized ADC units.
        primary supplies physical G/E mapping and fitted populations/centers.
        options sets the radius policy, in the same IQ units. Return the
        classification diagnostic and publish it as figure_name in plots.
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
        primary.validate_calibration()
        g_signals, e_signals = ge_signals_by_state(
            source.result.signals, primary.initial_state
        )
        signals = (g_signals, e_signals)
        try:
            confusion = self._calc_confusion_matrix(signals, primary, options)
        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ) as exc:
            raise GENumericalError(str(exc)) from exc
        self._plot_confusion_matrix(confusion, signals, primary, plots, figure_name)
        return GEPostAnalysis(confusion=confusion)

    def _calc_confusion_matrix(
        self,
        signals: tuple[NDArray[np.complex128], NDArray[np.complex128]],
        primary: GEAnalysis,
        options: GEPostAnalyzeOptions,
    ) -> GEConfusionResult:
        g_signals, e_signals = signals
        g_center, e_center = primary.g_center, primary.e_center
        init_matrix = make_init_matrix(primary.init_pops)

        radius = options.radius
        if radius is None:
            radius = optimize_ge_radius(
                g_signals,
                e_signals,
                g_center,
                e_center,
                primary.init_pops,
                primary.ge_s,
                consider_other=options.consider_other,
            )

        gg_mask, ge_mask, go_mask = classify_result(
            g_signals, g_center, e_center, radius
        )
        g_classification = GEClassification(
            float(gg_mask.mean()),
            float(ge_mask.mean()),
            float(go_mask.mean()),
        )

        eg_mask, ee_mask, eo_mask = classify_result(
            e_signals, g_center, e_center, radius
        )
        e_classification = GEClassification(
            float(eg_mask.mean()),
            float(ee_mask.mean()),
            float(eo_mask.mean()),
        )

        matrix = solve_confusion_matrix(
            init_matrix,
            make_result_matrix(g_classification, e_classification),
        )
        return GEConfusionResult(
            radius=radius,
            matrix=matrix,
            init_matrix=init_matrix,
            g_classification=g_classification,
            e_classification=e_classification,
            condition_number=float(np.linalg.cond(matrix)),
        )

    def _plot_confusion_matrix(
        self,
        confusion: GEConfusionResult,
        signals: tuple[NDArray[np.complex128], NDArray[np.complex128]],
        primary: GEAnalysis,
        plots: Plots,
        figure_name: str,
    ) -> None:
        g_signals, e_signals = signals
        g_center, e_center = primary.g_center, primary.e_center
        fig, raw_axes = plots.subplots(figure_name, nrows=2, ncols=2, figsize=(8, 8))
        # tolerated(cast): Native 2x2 subplots return Axes | object ndarray;
        # the array type encodes neither this shape nor its Axes elements.
        ((ax1, ax4), (ax2, ax3)) = cast(
            "tuple[tuple[Axes, Axes], tuple[Axes, Axes]]", raw_axes
        )

        g_label = r"$|0\rangle$"
        e_label = r"$|1\rangle$"
        l_label = r"$|L\rangle$"
        plot_with_classified(ax1, g_signals, g_center, e_center, confusion.radius)
        plot_with_classified(ax2, e_signals, g_center, e_center, confusion.radius)

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
