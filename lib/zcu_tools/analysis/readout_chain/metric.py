"""Readout information from saved 32-read block moments."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._estimator import fitted_pair_result, one_step_fit


@dataclass(frozen=True)
class ReadoutSums:
    """Saved readout moments for F settings in acquisition order.

    axis: Shape (F,), setting coordinates in caller-defined units.
    block_count: Shape (F,), integer number of contiguous 32-read blocks, >= 2.
    feature_sums: Shape (F,113), sums of the per-block averages. Columns:
        I,Q; I²,IQ,Q²; lag 1..15 II,IQ,QI,QQ; lag 1..4
        (I²,IQ,Q²)_t (I,Q)_{t+k}; then lag 1..4
        (I,Q)_t (I²,IQ,Q²)_{t+k}. Each lag uses 32-k pairs.
        Units are ADC, ADC², or ADC³ according to the moment degree.

    Construction rejects empty/mismatched shapes, non-finite arrays and counts
    that are non-integer or less than two with ValueError.
    """

    axis: NDArray[np.float64]
    block_count: NDArray[np.float64]
    feature_sums: NDArray[np.float64]

    def __post_init__(self) -> None:
        _validate_sums(self)

    @classmethod
    def from_processor(
        cls, axis: NDArray[np.float64], sums: NDArray[np.float64]
    ) -> "ReadoutSums":
        """Copy (F,114) processor output: count followed by 113 moment sums.

        axis has shape (F,) and caller-defined units. Raise ValueError for
        malformed/non-finite output or counts below two or non-integer.
        """
        if axis.ndim != 1 or sums.shape != (len(axis), 114):
            raise ValueError("processor sums must have shape (settings, 114)")
        return cls(axis.copy(), sums[:, 0].copy(), sums[:, 1:].copy())


@dataclass(frozen=True)
class ReadoutMetric:
    """Results for F settings, all dimensionless and in acquisition order.

    metric: Shape (F,), read-drive-read information for exchange of the two
        most populated fitted states, evaluated at equal pair populations
        and uniform landing within that pair. Larger is better.
    standard_error: Shape (F,), one-sigma delta-method error of metric using
        the start-model covariance and the one-step estimator, including
        its fixed weak priors. Asymptotic in block count; not a fit-bias bound.
    separation: Shape (F,), dominant-pair centre distance divided by the
        fitted per-I/Q Gaussian noise standard deviation (d/sigma).
    retention: Shape (F,2), read retention λ for the two dominant states,
        in descending stationary-population order; 0.01 <= λ <= 0.995.
        λ is the retain-versus-reset probability, not a diagonal transition.
        Individual λ may be unidentifiable when the third state is unseen.
    third_population: Shape (F,), stationary population of the remaining
        fitted state (the least populated), between zero and one third.
    ranking: Shape (F,), integer setting indices, metric descending.
        Exact ties preserve acquisition order. It is not a reordered axis.
    """

    metric: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    separation: NDArray[np.float64]
    retention: NDArray[np.float64]
    third_population: NDArray[np.float64]
    ranking: NDArray[np.int64]


def readout_metric(sums: ReadoutSums) -> ReadoutMetric:
    """Fit three states with a closed-form start and exactly one GN step.

    sums holds the 113 block moments defined on ReadoutSums. Assume a stationary
    contiguous stream, isotropic independent Gaussian read noise, and transition
    R_ij = λ_i δ_ij + (1-λ_i) ρ_j. Use exact contiguous 32-read block covariance
    at the start and the fixed weak third-state priors. Ignore stream-end edges.
    No hardware, file access or input mutation occurs. The metric discards the
    least populated state only when evaluating the fitted dominant pair.
    Raise ValueError for malformed/non-finite sums, fewer than two blocks,
    unidentifiable main-axis signal or a non-finite one-step result.
    """
    _validate_sums(sums)
    results = np.empty((len(sums.axis), 6))
    for i, blocks in enumerate(sums.block_count):
        features = sums.feature_sums[i] / blocks
        fitted, covariance = one_step_fit(features, float(blocks))
        results[i] = fitted_pair_result(fitted, covariance)
    if not np.isfinite(results).all():
        raise ValueError("non-finite one-step readout result")
    return ReadoutMetric(
        metric=results[:, 0],
        standard_error=results[:, 1],
        separation=results[:, 2],
        retention=results[:, 3:5],
        third_population=results[:, 5],
        ranking=np.argsort(-results[:, 0], kind="stable"),
    )


def _validate_sums(sums: ReadoutSums) -> None:
    if sums.axis.ndim != 1 or sums.axis.size == 0:
        raise ValueError("axis must have nonempty shape (settings,)")
    points = len(sums.axis)
    for name, values, shape in (
        ("axis", sums.axis, (points,)),
        ("block_count", sums.block_count, (points,)),
        ("feature_sums", sums.feature_sums, (points, 113)),
    ):
        if values.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
    if np.any(sums.block_count < 2):
        raise ValueError("each setting must contain at least two blocks")
    if np.any(sums.block_count != np.floor(sums.block_count)):
        raise ValueError("block counts must be integer-valued")
