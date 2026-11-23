"""Numerical shot classification and population conversion."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Annotated, NamedTuple

import numpy as np
from numpy.typing import NDArray
from pydantic import PlainValidator

from zcu_tools.cfg_model import Centroid3Classifier
from zcu_tools.utils.shot_classification import (
    classify_shots,
    gaussian_region_probability,
    nearest_centroid3,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GECalibration:
    """Fitted physical g/e geometry and preparation mixing, without lab settings.

    g_center/e_center are complex centers in the original IQ frame, in normalized
    ADC units. init_pops is the (2,2) preparation-row / physical-g/e-column
    probability matrix. sigma is the fitted width in normalized ADC units.
    """

    g_center: complex
    e_center: complex
    init_pops: NDArray[np.float64]
    sigma: float


def make_init_matrix(init_pops: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extend (2,2) g/e preparation populations to a 3x3 g/e/other matrix.

    The other preparation is isolated. Invalid probabilities raise ValueError.
    """
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
    """Solve preparation mixing A_init @ confusion = Q for 3x3 matrices.

    Clip negative entries and normalize rows. Singular A_init raises LinAlgError.
    """
    confusion_matrix = np.linalg.solve(A_init, Q)
    confusion_matrix = np.clip(confusion_matrix, 0.0, None)
    confusion_matrix /= confusion_matrix.sum(axis=1, keepdims=True)

    return confusion_matrix


def optimize_ge_radius(
    g_signals: NDArray[np.complex128],
    e_signals: NDArray[np.complex128],
    calibration: GECalibration,
    *,
    consider_other: bool,
) -> float:
    """Select a radius in normalized IQ units minimizing matrix condition number.

    calibration supplies fitted centers, sigma and (2,2) preparation populations.
    consider_other penalizes midpoint leakage only during radius selection.
    Invalid geometry and unsuccessful optimization raise.
    """
    from scipy.optimize import OptimizeResult, minimize_scalar

    g_center, e_center = calibration.g_center, calibration.e_center
    init_pops, sigma = calibration.init_pops, calibration.sigma

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


@dataclass(frozen=True)
class GEConfusionResult:
    """Radius in normalized IQ units and population-corrected g/e/other matrix.

    matrix and init_matrix are (3,3), with g/e/other rows and columns.
    g_classification/e_classification are observed fractions before correction.
    condition_number describes matrix invertibility, not calibration approval.
    """

    radius: float
    matrix: NDArray[np.float64]
    init_matrix: NDArray[np.float64]
    g_classification: GEClassification
    e_classification: GEClassification
    condition_number: float


def calc_confusion_matrix(
    signals: tuple[NDArray[np.complex128], NDArray[np.complex128]],
    calibration: GECalibration,
    *,
    radius: float | None,
    consider_other: bool,
) -> GEConfusionResult:
    """Classify physical g/e shot arrays and correct fitted preparation mixing.

    calibration supplies centers, sigma and (2,2) preparation populations.
    signals contains two one-dimensional raw IQ arrays in normalized ADC units.
    radius is nonnegative in those units, or None for automatic optimization.
    consider_other selects the midpoint leakage penalty during optimization only.
    Return a 3x3 g/e/other diagnostic; numerical failures propagate.
    """
    g_signals, e_signals = signals
    g_center, e_center = calibration.g_center, calibration.e_center
    init_matrix = make_init_matrix(calibration.init_pops)

    if radius is None:
        radius = optimize_ge_radius(
            g_signals,
            e_signals,
            calibration,
            consider_other=consider_other,
        )

    gg_mask, ge_mask, go_mask = classify_result(g_signals, g_center, e_center, radius)
    g_classification = GEClassification(
        float(gg_mask.mean()),
        float(ge_mask.mean()),
        float(go_mask.mean()),
    )

    eg_mask, ee_mask, eo_mask = classify_result(e_signals, g_center, e_center, radius)
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


def raw_population_signal(
    raw: Sequence[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Return the first readout's first population array from acquisition raw data.

    raw has the program's readout/round nesting; indexing errors propagate and
    the returned float array is borrowed, not copied."""
    return raw[0][0]


def correct_populations(
    populations: NDArray[np.float64],
    confusion_matrix: NDArray[np.float64] | None,
    *,
    clip: bool = True,
    fallback_on_nonfinite: bool = False,
) -> NDArray[np.float64]:
    """Correct final-axis state fractions with the readout confusion matrix.

    populations has arbitrary leading dimensions and a final state axis.
    confusion_matrix holds true-state rows and measured-state columns; None
    returns populations unchanged. Otherwise solve its transpose, returning
    float64 with the same shape. clip=True limits estimates to [0, 1];
    clip=False preserves negative and above-one estimates. Singular/non-square or
    incompatible matrices raise NumPy linear-algebra/value errors.

    With fallback_on_nonfinite=True, require a finite matrix. If any fully
    finite input point solves to nonfinite values before clipping, log once
    and return the original populations object unchanged, just as for None.
    Callers can use object identity to tell whether a matrix was applied.
    Already missing input points do not trigger this fallback.
    """
    if confusion_matrix is None:
        return populations

    matrix = np.asarray(confusion_matrix, dtype=np.float64)
    if fallback_on_nonfinite and not np.isfinite(matrix).all():
        raise ValueError("confusion matrix must be finite")
    state_count = populations.shape[-1]
    flat_populations = populations.reshape(-1, state_count)
    corrected = np.linalg.solve(matrix.T, flat_populations.T).T
    corrected = corrected.reshape(populations.shape)
    if fallback_on_nonfinite:
        failed = np.isfinite(populations).all(axis=-1) & ~np.isfinite(corrected).all(
            axis=-1
        )
        if failed.any():
            logger.warning(
                "Readout correction: skipping correction; nonfinite corrected values for %d finite classified points",
                np.count_nonzero(failed),
            )
            return populations
    return np.asarray(
        np.clip(corrected, 0.0, 1.0) if clip else corrected, dtype=np.float64
    )


@dataclass(frozen=True)
class ReferenceIQAnalysis:
    """Three-label nearest-centroid calibration and held-out diagnostics.

    All state axes use label order (0, 1, 2), not input row order.
    centroids: three complex means in input IQ units, trained on the first half.
    confusion_matrix: (3,3) training fractions, preparation rows / prediction columns.
    raw_populations: (3,3) evaluation fractions with the same axis convention.
    corrected_populations: evaluation fractions solved in preparation-label
        coordinates, including negative and above-one estimates. These are not
        latent energy-state populations when preparations are contaminated.
    label_errors: three evaluation misassignment fractions, including preparation error.
    macro_error: equally weighted mean label error, not pure-state readout infidelity.
    minimum_separation: closest pair of training centroids' Euclidean distance
        divided by the shared isotropic training sigma (dimensionless).
    condition_number: training confusion matrix 2-norm condition number.
    minimum_singular_value: training confusion matrix minimum singular value.
    training_shots/evaluation_shots: disjoint per-label sample counts.
    """

    centroids: NDArray[np.complex128]
    confusion_matrix: NDArray[np.float64]
    raw_populations: NDArray[np.float64]
    corrected_populations: NDArray[np.float64]
    label_errors: NDArray[np.float64]
    macro_error: float
    minimum_separation: float
    condition_number: float
    minimum_singular_value: float
    training_shots: int
    evaluation_shots: int


def analyze_reference_iq(
    signals: NDArray[np.complex128], labels: NDArray[np.int64]
) -> ReferenceIQAnalysis:
    """Train and evaluate a three-label classifier using only acquired IQ.

    signals is finite complex IQ of shape (3,N), in any consistent ADC units;
    labels is a permutation of integers (0,1,2) naming the corresponding rows.
    N must be >=6. First floor(N/2) shots per row train centroids and confusion;
    the remaining shots evaluate assignment and unclipped correction. Exact
    nearest-centroid ties choose the lowest label. Output axes are label-sorted.
    The shared isotropic sigma is the square root of half the trace of the
    equally pooled unbiased within-training IQ covariance. Coincident centers,
    zero or nonfinite sigma, invalid labels, and incomplete/nonfinite data raise
    ValueError. Singular confusion raises LinAlgError.
    No state-purity inference or clipping is performed.
    """
    if signals.ndim != 2 or signals.shape[0] != 3 or signals.shape[1] < 6:
        raise ValueError("Reference IQ must have shape (3,N) with N >= 6")
    if labels.shape != (3,) or not np.array_equal(np.sort(labels), [0, 1, 2]):
        raise ValueError("Reference labels must be a permutation of (0,1,2)")
    if not np.isfinite(signals).all():
        raise ValueError("Reference IQ must contain only finite shots")
    ordered = signals[np.argsort(labels)]
    split = ordered.shape[1] // 2
    training, evaluation = ordered[:, :split], ordered[:, split:]
    centroids = training.mean(axis=1)
    train_predictions = nearest_centroid3(training, *centroids)
    evaluation_predictions = nearest_centroid3(evaluation, *centroids)
    confusion = np.stack(
        [(train_predictions == i).mean(axis=1) for i in range(3)], axis=1
    )
    raw = np.stack(
        [(evaluation_predictions == i).mean(axis=1) for i in range(3)], axis=1
    )
    # Complex sample variance already sums the I and Q variances.
    sigma = np.sqrt(np.mean(np.var(training, axis=1, ddof=1)) / 2)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Reference shared IQ sigma must be finite and positive")
    minimum_distance = min(
        abs(first - second) for first, second in combinations(centroids, 2)
    )
    errors = 1 - np.diag(raw)
    return ReferenceIQAnalysis(
        centroids=centroids,
        confusion_matrix=confusion,
        raw_populations=raw,
        corrected_populations=correct_populations(raw, confusion, clip=False),
        label_errors=errors,
        macro_error=float(errors.mean()),
        minimum_separation=float(minimum_distance / sigma),
        condition_number=float(np.linalg.cond(confusion)),
        minimum_singular_value=float(np.linalg.svd(confusion, compute_uv=False).min()),
        training_shots=split,
        evaluation_shots=ordered.shape[1] - split,
    )


def classify_reference_iq(
    signals: NDArray[np.complex128], centroids: NDArray[np.complex128]
) -> NDArray[np.float64]:
    """Classify acquired shots using three fixed reference IQ centers.

    signals has arbitrary leading scan axes and a nonempty final shot axis.
    centroids has shape (3,) in label order (g,e,f), in the same ADC units as
    signals, e.g. analyze_reference_iq(...).centroids. Centers must be distinct.
    Return scan axes plus a final three-label fraction axis. Exact nearest-center
    ties choose the lowest label. No retraining, correction or clipping occurs.
    Empty/nonfinite shots or invalid centers raise ValueError. Preparation-label
    fractions are not latent energy-state populations.
    """
    if signals.ndim < 1 or signals.shape[-1] == 0 or not np.isfinite(signals).all():
        raise ValueError("Acquired IQ must have a nonempty shot axis and finite shots")
    if centroids.shape != (3,) or not np.isfinite(centroids).all():
        raise ValueError("Reference centroids must have shape (3,) and be finite")
    predictions = nearest_centroid3(signals, *centroids)
    return np.stack(
        [(predictions == label).mean(axis=-1) for label in range(3)], axis=-1
    )


def _centroid3_populations(value: object) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except TypeError as error:
        raise ValueError("Expected float64-convertible population values") from error
    if array.ndim < 1 or array.shape[-1] != 3:
        raise ValueError("Population shape must be scan axes followed by g/e/f")
    if not np.isfinite(array).all():
        raise ValueError("Population values must be finite numbers")
    return array


def _centroid3_confusion(value: object) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except TypeError as error:
        raise ValueError("Expected float64-convertible confusion values") from error
    if array.shape != (3, 3):
        raise ValueError("Confusion matrix shape must be (3, 3)")
    if not np.isfinite(array).all():
        raise ValueError("Confusion matrix values must be finite numbers")
    return array


@dataclass(frozen=True)
class Centroid3IQAnalysis:
    """Fixed-calibration IQ fractions in preparation-label coordinates.

    raw_populations/corrected_populations preserve all scan axes and append
    g/e/f. Corrected fractions may be negative or above one, without clipping.
    confusion_matrix copies calibration rows prepared g/e/f, columns measured
    g/e/f. condition_number is its 2-norm condition number;
    minimum_singular_value is its smallest singular value. No purity is inferred.
    TypeAdapter decodes native summary numeric lists to finite float64 arrays:
    populations require a final axis of three, confusion requires shape (3, 3).
    Invalid dtype/shape or null/nonfinite array values raise validation errors.
    Decoding preserves scan axes and unclipped corrected fractions.
    """

    raw_populations: Annotated[
        NDArray[np.float64], PlainValidator(_centroid3_populations)
    ]
    corrected_populations: Annotated[
        NDArray[np.float64], PlainValidator(_centroid3_populations)
    ]
    confusion_matrix: Annotated[
        NDArray[np.float64], PlainValidator(_centroid3_confusion)
    ]
    condition_number: float
    minimum_singular_value: float


@dataclass(frozen=True)
class Centroid3Summary:
    """Fixed-size saved-readout diagnostics in preparation-label coordinates.

    The fields describe the matrix actually applied. All three are None when
    no correction was applied. Rows are prepared g/e/f, columns measured g/e/f.
    TypeAdapter decodes finite numeric lists to a float64 (3, 3) matrix.
    condition_number is the 2-norm condition number (None if it exceeds
    float64 range); minimum_singular_value is the smallest singular value
    (None if it exceeds float64 range). No purity or fidelity is inferred.
    """

    confusion_matrix: (
        Annotated[NDArray[np.float64], PlainValidator(_centroid3_confusion)] | None
    )
    condition_number: float | None
    minimum_singular_value: float | None

    @staticmethod
    def from_applied_matrix(matrix: NDArray[np.float64] | None) -> Centroid3Summary:
        """Describe an applied g/e/f correction, or report no correction for None."""
        if matrix is None:
            return Centroid3Summary(None, None, None)
        matrix = _centroid3_confusion(matrix)
        condition, minimum = _confusion_quality(matrix)
        return Centroid3Summary(
            matrix.copy(),
            condition if np.isfinite(condition) else None,
            minimum if np.isfinite(minimum) else None,
        )


def _confusion_quality(matrix: NDArray[np.float64]) -> tuple[float, float]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        condition = singular_values[0] / singular_values[-1]
    return float(condition), float(singular_values[-1])


def summarize_centroid3(result: Centroid3IQAnalysis) -> Centroid3Summary:
    """Retain saved-readout diagnostics without scan-sized fractions.

    Copy the finite training confusion matrix and retain both computed
    diagnostics, using None for values above float64 range. Malformed
    calibration or negative/NaN diagnostics raise ValueError. The complete
    result remains available to the caller for local plotting.
    """
    matrix = _centroid3_confusion(result.confusion_matrix)
    condition, minimum = result.condition_number, result.minimum_singular_value
    if not (condition >= 0 and minimum >= 0):
        raise ValueError("Confusion diagnostics must be nonnegative and not NaN")
    return Centroid3Summary(
        confusion_matrix=matrix.copy(),
        condition_number=None if condition == np.inf else condition,
        minimum_singular_value=None if minimum == np.inf else minimum,
    )


def analyze_centroid3_iq(
    signals: NDArray[np.complex128], calibration: Centroid3Classifier
) -> Centroid3IQAnalysis:
    """Classify raw acquired IQ and correct with one captured readout calibration.

    signals has arbitrary scan axes and a nonempty final shot axis, in the same
    mean-sample ADC units as calibration centers. calibration must be the
    caller's saved-run centroid3 calibration, not a new fit or live lookup.
    Return raw/unclipped corrected g/e/f fractions and training-matrix quality.
    Nonfinite/incomplete shots raise ValueError; a singular confusion matrix
    raises LinAlgError. Inputs are unchanged. No library or hardware is read.
    """
    centers = np.asarray(
        [calibration.g_center, calibration.e_center, calibration.f_center],
        dtype=np.complex128,
    )
    raw = classify_reference_iq(signals, centers)
    return analyze_centroid3_populations(raw, calibration)


def analyze_centroid3_populations(
    populations: NDArray[np.float64], calibration: Centroid3Classifier
) -> Centroid3IQAnalysis:
    """Correct measured g/e/f fractions with one saved readout calibration.

    populations contains normalized measured fractions, with arbitrary scan axes
    and a final g/e/f axis of length three. The caller normalizes shot counts
    before calling. calibration is the saved-run centroid3 classifier; its
    confusion rows are prepared g/e/f and columns are measured g/e/f.
    Return raw fractions (borrowed), unclipped corrected preparation-label
    estimates, a detached training matrix and its 2-norm condition number and
    minimum singular value. Negative and above-one estimates are retained;
    no purity is inferred. Wrong-shaped/nonfinite fractions raise ValueError;
    singular confusion raises LinAlgError. Inputs are unchanged. No library,
    program, hardware, classification or retraining is used.
    """
    raw = _centroid3_populations(populations)
    matrix = np.asarray(calibration.confusion_matrix, dtype=np.float64)
    corrected = correct_populations(raw, matrix, clip=False)
    condition, minimum = _confusion_quality(matrix)
    return Centroid3IQAnalysis(
        raw_populations=raw,
        corrected_populations=corrected,
        confusion_matrix=matrix,
        condition_number=condition,
        minimum_singular_value=minimum,
    )


def classify_result(
    signals: NDArray[np.complex128],
    g_center: complex,
    e_center: complex,
    radius: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Classify complex IQ shots into ground, excited and other boolean masks.

    g_center/e_center are distinct complex centers; radius is their positive
    finite classification radius. Output masks preserve signals.shape, use
    nearest-center regions and assign exact ties/outside circles to other.
    Invalid centers/radius raise ValueError via the classification kernel."""
    return classify_shots(signals, g_center, e_center, radius)


def classify_rabi_iq(
    signals: NDArray[np.complex128],
    g_center: complex,
    e_center: complex,
    radius: float,
) -> NDArray[np.float64]:
    """Compute ground/excited fractions for raw IQ of shape (sweep, shot).

    g_center/e_center and radius follow classify_result. Returns float64 of
    shape (sweep, 2); non-2D signals raise ValueError and invalid classification
    geometry propagates. No readout correction or fit is performed."""
    if signals.ndim != 2:
        raise ValueError("Rabi raw IQ must have shape (sweep, shot)")
    g_mask, e_mask, _ = classify_result(signals, g_center, e_center, radius)
    return np.stack((g_mask.mean(axis=1), e_mask.mean(axis=1)), axis=1)
