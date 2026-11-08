"""Saved sweep records and their background-significance curve."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ._statistics import block_distances, evidence_logp, normal_sigma, whiten


@dataclass(frozen=True)
class SweepSums:
    """Running sums for F sweep points in acquisition order.

    axis: Shape (F,), sweep coordinates in caller-defined units, unchanged.
    dwell_count: Shape (F,), number of dwells (shots) at each point, >= 2.
    feature_sums: Shape (F, 11), sums of per-dwell features. Columns are
        mean I, mean Q; mean I², IQ, Q²; symmetric lag-1 II, IQ+QI, QQ;
        and summed lag-2..15 II, IQ+QI, QQ. Means use n-k pairs for lag k.
        First two columns use ADC units, remaining columns ADC units squared.
    product_sum: Shape (11, 11), sum of feature outer products over all
        dwells and points. Entry (i,j) has the product of feature i,j units.

    All arrays must be finite. Construction raises ValueError for incompatible
    shapes, non-finite arrays or fewer than 2 dwells.
    Covariance is checked by sweep_curve, not by this record constructor.
    """

    axis: NDArray[np.float64]
    dwell_count: NDArray[np.float64]
    feature_sums: NDArray[np.float64]
    product_sum: NDArray[np.float64]

    def __post_init__(self) -> None:
        _validate_sums(self)

    @classmethod
    def from_augmented(
        cls,
        axis: NDArray[np.float64],
        augmented: NDArray[np.float64],
    ) -> "SweepSums":
        """Pool processor sums of shape (F,12,12) into a saved record.

        Each point contains sum z zᵀ with z=(1, feature_0, ..., feature_10).
        axis contains sweep coordinates in caller-defined units, unchanged.
        Raise ValueError for incompatible shapes or non-finite inputs.
        """
        if axis.ndim != 1 or augmented.shape != (len(axis), 12, 12):
            raise ValueError("augmented sums must have shape (points, 12, 12)")
        if not np.isfinite(augmented).all():
            raise ValueError("augmented sums must contain only finite values")
        return cls(
            axis=axis.copy(),
            dwell_count=augmented[:, 0, 0].copy(),
            feature_sums=augmented[:, 0, 1:].copy(),
            product_sum=augmented[:, 1:, 1:].sum(axis=0),
        )


def _validate_sums(sums: SweepSums) -> None:
    if sums.axis.ndim != 1 or sums.axis.size == 0:
        raise ValueError("axis must have nonempty shape (points,)")
    points = len(sums.axis)
    arrays = (
        ("axis", sums.axis, (points,)),
        ("dwell_count", sums.dwell_count, (points,)),
        ("feature_sums", sums.feature_sums, (points, 11)),
        ("product_sum", sums.product_sum, (11, 11)),
    )
    for name, array, shape in arrays:
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
    if np.any(sums.dwell_count < 2):
        raise ValueError("each point must contain at least 2 dwells")


@dataclass(frozen=True)
class SweepCurve:
    """Results in acquisition order for F points.

    sigma: Shape (F,), nonnegative two-sided normal equivalent of evidence p.
    level: Dimensionless sigma threshold for 5% family false-mark probability.
    iq_mean: Shape (F,), complex mean I + iQ in the input ADC units.
    dwell_count: Shape (F,), dwell counts used for the pointwise noise scaling.
    """

    sigma: NDArray[np.float64]
    level: float
    iq_mean: NDArray[np.complex128]
    dwell_count: NDArray[np.float64]


def sweep_curve(sums: SweepSums) -> SweepCurve:
    """Compute significance above a componentwise-median sweep background.

    sums contains the running sums defined on SweepSums. Subtract point means
    from pooled products to estimate per-dwell noise, then Cholesky-whiten the
    means with sqrt(dwell_count) scaling. Combine independent 2-D IQ and 9-D
    conditional cross-block evidence against the exact chi-square null.
    Return sigma, the 5% family level, mean IQ and counts, without changing sums.
    Raise ValueError for malformed/non-finite records or <2 dwells;
    raise ValueError if pooled within-point covariance is not positive definite.
    The null is asymptotic in dwell count; no small-sample correction is applied.
    """
    # The frozen record's arrays remain mutable after construction.
    _validate_sums(sums)
    means = sums.feature_sums / sums.dwell_count[:, None]
    whitened = whiten(means, sums.dwell_count, sums.product_sum)
    mean_distance, cross_distance = block_distances(whitened)
    return SweepCurve(
        sigma=normal_sigma(evidence_logp(mean_distance, cross_distance)),
        level=float(norm.isf(0.05 / (2 * len(sums.axis)))),
        iq_mean=means[:, 0] + 1j * means[:, 1],
        dwell_count=sums.dwell_count.copy(),
    )
