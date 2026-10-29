"""Numerical shot classification and population conversion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import PlainValidator

from zcu_tools.cfg_model import Centroid3Classifier
from zcu_tools.utils.shot_classification import classify_shots, nearest_centroid3


def raw_population_signal(
    raw: Sequence[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Return the first readout's first population array from acquisition raw data.

    raw has the program's readout/round nesting; indexing errors propagate and
    the returned float array is borrowed, not copied."""
    return raw[0][0]


def calc_populations(signals: NDArray[np.float64]) -> NDArray[np.float64]:
    """Append other-state probability 1-g-e to final-axis ground/excited fractions.

    signals has final axis containing at least g/e; returns a new float array
    with final axis (g, e, other). Does not clamp or renormalize inputs."""
    g_pop, e_pop = signals[..., 0], signals[..., 1]
    return np.stack([g_pop, e_pop, 1 - g_pop - e_pop], axis=-1)


def correct_populations(
    populations: NDArray[np.float64],
    confusion_matrix: NDArray[np.float64] | None,
    *,
    clip: bool = True,
) -> NDArray[np.float64]:
    # Apply readout confusion-matrix correction. None -> returned unchanged
    # (deliberate no-op contract). A singular/non-square matrix raises via
    # np.linalg.solve (Fast-Fail preserved).
    """Correct final-axis state fractions with the readout confusion matrix.

    populations has arbitrary leading dimensions and a final state axis.
    confusion_matrix holds true-state rows and measured-state columns; None
    returns populations unchanged. Otherwise solve its transpose, returning
    float64 with the same shape. clip=True limits estimates to [0, 1];
    clip=False preserves negative and above-one estimates. Singular/non-square or
    incompatible matrices raise NumPy linear-algebra/value errors."""
    if confusion_matrix is None:
        return populations

    matrix = np.asarray(confusion_matrix, dtype=np.float64)
    state_count = populations.shape[-1]
    flat_populations = populations.reshape(-1, state_count)
    corrected = np.linalg.solve(matrix.T, flat_populations.T).T
    corrected = corrected.reshape(populations.shape)
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
    minimum_separation: minimum pairwise pooled Mahalanobis distance from training.
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
    Pooled covariance is the equally weighted mean of unbiased within-training
    IQ covariances. Degenerate centers/covariance, singular confusion, invalid
    labels or incomplete/nonfinite data raise ValueError or LinAlgError.
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
    if any(first == second for first, second in combinations(centroids, 2)):
        raise ValueError("Reference centroids must be distinct")
    train_predictions = nearest_centroid3(training, *centroids)
    evaluation_predictions = nearest_centroid3(evaluation, *centroids)
    confusion = np.stack(
        [(train_predictions == i).mean(axis=1) for i in range(3)], axis=1
    )
    raw = np.stack(
        [(evaluation_predictions == i).mean(axis=1) for i in range(3)], axis=1
    )
    covariance = np.mean(
        [np.cov(np.stack((row.real, row.imag))) for row in training], axis=0
    )
    if np.linalg.matrix_rank(covariance) != 2:
        raise ValueError("Reference pooled IQ covariance must be nonsingular")
    separations = []
    for first, second in combinations(centroids, 2):
        difference = np.array([(first - second).real, (first - second).imag])
        separations.append(
            float(np.sqrt(difference @ np.linalg.solve(covariance, difference)))
        )
    errors = 1 - np.diag(raw)
    return ReferenceIQAnalysis(
        centroids=centroids,
        confusion_matrix=confusion,
        raw_populations=raw,
        corrected_populations=correct_populations(raw, confusion, clip=False),
        label_errors=errors,
        macro_error=float(errors.mean()),
        minimum_separation=min(separations),
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
    if len(np.unique(centroids)) != 3:
        raise ValueError("Reference centroids must be distinct")
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
    matrix = np.asarray(calibration.confusion_matrix, dtype=np.float64)
    raw = classify_reference_iq(signals, centers)
    return Centroid3IQAnalysis(
        raw_populations=raw,
        corrected_populations=correct_populations(raw, matrix, clip=False),
        confusion_matrix=matrix,
        condition_number=float(np.linalg.cond(matrix)),
        minimum_singular_value=float(np.linalg.svd(matrix, compute_uv=False).min()),
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
