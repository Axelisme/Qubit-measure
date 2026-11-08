"""Saved cycle records and live-plot projections."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._statistics import whiten


@dataclass(frozen=True)
class CycleSums:
    """Running sums for F cycle sweep points in acquisition order.

    axis: Shape (F,), sweep coordinates in caller-defined units, unchanged.
    cycle_count: Shape (F,), number of cycles at each point, >= 2.
    feature_sums: Shape (F,65), sums of per-cycle features. Columns 0:32
        contain I,Q of 16 reads in read order, in ADC units. Columns 32:41
        contain same-read I²,IQ,Q² summed over read 0, even reads 2..14,
        and odd reads 1..15. Columns 41:65 contain II,IQ,QI,QQ pair sums
        for six groups of drive count d and read gap g: (0,1), (1,1),
        (1,2..3), (2,2..3), (2,>=4), (>=3,any). Products use ADC units squared.
    product_sum: Shape (65,65), feature outer products summed over all cycles
        and points, not per point. Entry (i,j) has feature i times feature j units.

    All arrays must be finite. Construction raises ValueError for incompatible
    shapes, an empty axis, non-finite values or fewer than 2 cycles at a point.
    Covariance is checked by cycle_curves, not by this record constructor.
    Fields are frozen; their NumPy arrays remain mutable.
    """

    axis: NDArray[np.float64]
    cycle_count: NDArray[np.float64]
    feature_sums: NDArray[np.float64]
    product_sum: NDArray[np.float64]

    def __post_init__(self) -> None:
        _validate_sums(self)

    @classmethod
    def from_augmented(
        cls, axis: NDArray[np.float64], augmented: NDArray[np.float64]
    ) -> "CycleSums":
        """Pool processor sums of shape (F,66,66) into a saved record.

        axis has the meaning given on CycleSums. At each point augmented is
        sum z zᵀ for z=(1, feature_0, ..., feature_64): [0,0] is the count,
        [0,1:] the feature sums and [1:,1:] the feature product sums.
        Copy axis, counts and feature sums; add the product block over points.
        Raise ValueError for incompatible shapes, non-finite values, an empty
        axis or fewer than 2 cycles at any point. Inputs are not changed.
        """
        if axis.ndim != 1 or augmented.shape != (len(axis), 66, 66):
            raise ValueError("augmented sums must have shape (points, 66, 66)")
        if not np.isfinite(augmented).all():
            raise ValueError("augmented sums must contain only finite values")
        return cls(
            axis=axis.copy(),
            cycle_count=augmented[:, 0, 0].copy(),
            feature_sums=augmented[:, 0, 1:].copy(),
            product_sum=augmented[:, 1:, 1:].sum(axis=0),
        )


def _validate_sums(sums: CycleSums) -> None:
    if sums.axis.ndim != 1 or sums.axis.size == 0:
        raise ValueError("axis must have nonempty shape (points,)")
    points = len(sums.axis)
    arrays = (
        ("axis", sums.axis, (points,)),
        ("cycle_count", sums.cycle_count, (points,)),
        ("feature_sums", sums.feature_sums, (points, 65)),
        ("product_sum", sums.product_sum, (65, 65)),
    )
    for name, array, shape in arrays:
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
    if np.any(sums.cycle_count < 2):
        raise ValueError("each point must contain at least 2 cycles")


@dataclass(frozen=True)
class CycleCurves:
    """Live-plot results in acquisition order for F points.

    main_sigma: Shape (F,), signed projection of all 65 whitened mean features
        onto their main direction, in standard errors of the point means.
        Its sign aligns with first_read_sigma when their correlation is nonzero.
    first_read_sigma: Shape (F,), median-background-subtracted first-read IQ
        mean projected onto its raw-IQ principal axis, in point-mean standard
        errors. The axis's larger-magnitude component is positive.
    cycle_count: Shape (F,), cycle counts used for pointwise noise scaling.
    """

    main_sigma: NDArray[np.float64]
    first_read_sigma: NDArray[np.float64]
    cycle_count: NDArray[np.float64]


def cycle_curves(sums: CycleSums) -> CycleCurves:
    """Compute the all-feature (a) and first-read (c) live-plot curves.

    sums contains the saved moments defined on CycleSums. Estimate within-point
    covariance by subtracting each point mean outer product from pooled products
    and dividing by sum(cycle_count)-F. Subtract the componentwise median over
    points, Cholesky-whiten and scale by sqrt(cycle_count). Project these values
    onto the first right singular vector of their across-point mean-centred
    matrix to obtain (a); do not mean-centre the projected values themselves.
    For (c), select the principal axis of centred first-read IQ means before
    whitening, then divide its median-subtracted projection by its pointwise
    standard error. Align (a) with (c) using their across-point covariance sign.
    Return the two sigma curves and a count copy without changing sums.

    Raise ValueError for malformed/non-finite records, <2 cycles at any point,
    or a within-point covariance that is not symmetric positive definite.
    No covariance regularization or finite-sample PCA correction is applied.
    Directions estimated on the same points inflate pure-noise variance at
    finite F; sigma units refer to the within-point standard errors.
    """
    _validate_sums(sums)
    counts = sums.cycle_count
    means = sums.feature_sums / counts[:, None]
    whitened = whiten(means, counts, sums.product_sum)
    main_axis = np.linalg.svd(whitened - whitened.mean(axis=0), full_matrices=False)[2][
        0
    ]
    main_sigma = whitened @ main_axis

    first_means = means[:, :2]
    first_centered = first_means - np.median(first_means, axis=0)
    first_axis = np.linalg.svd(
        first_centered - first_centered.mean(axis=0), full_matrices=False
    )[2][0]
    if first_axis[np.argmax(np.abs(first_axis))] < 0:
        first_axis = -first_axis
    first_covariance = (
        sums.product_sum[:2, :2] - first_means.T @ (counts[:, None] * first_means)
    ) / (counts.sum() - len(counts))
    first_read_sigma = (first_centered @ first_axis) / np.sqrt(
        (first_axis @ first_covariance @ first_axis) / counts
    )
    # Covariance has the correlation sign without dividing by zero on flat curves.
    if (main_sigma - main_sigma.mean()) @ (
        first_read_sigma - first_read_sigma.mean()
    ) < 0:
        main_sigma = -main_sigma
    return CycleCurves(main_sigma, first_read_sigma, counts.copy())
