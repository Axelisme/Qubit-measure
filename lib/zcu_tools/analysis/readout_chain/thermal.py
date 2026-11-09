"""Closed-form score of thermally reset 16-read shots."""

from dataclasses import dataclass
from math import atan2, cos, hypot, sin, sqrt

import numpy as np
from numpy.typing import NDArray

THERMAL_MOMENT_COUNT = 37


@dataclass(frozen=True)
class ThermalChainSums:
    """Saved additive moments of thermally reset shots, in acquisition order.

    axis: Shape (F,), finite setting coordinates in caller-defined units.
    shot_count: Shape (F,), positive integer shot counts, stored as float64.
    feature_sums: Shape (F, 37), finite sums: I,Q of R1,R2,R3,R4,R5,R15,R16;
        R1 I²,IQ,Q²; II′,IQ′,QI′,QQ′ for (R1,R2),(R1,R3),(R1,R4),
        (R1,R5),(R15,R16). Units are ADC or ADC² by moment degree.

    F must be positive. Construction raises ValueError for malformed shapes,
    nonreal/nonnumeric dtypes, non-finite values or nonpositive/noninteger
    counts. Arrays are borrowed; from_processor copies them.
    No moment is averaged over reads or shots.
    """

    axis: NDArray[np.float64]
    shot_count: NDArray[np.float64]
    feature_sums: NDArray[np.float64]

    def __post_init__(self) -> None:
        _validate_sums(self)

    @classmethod
    def from_processor(
        cls, axis: NDArray[np.float64], sums: NDArray[np.float64]
    ) -> "ThermalChainSums":
        """Copy axis (F,) and processor sums (F,38): shot count then moments.

        axis uses caller-defined units. Raise ValueError for empty/mismatched
        shapes, non-finite data or nonpositive/noninteger shot counts.
        """
        if axis.ndim != 1 or sums.shape != (axis.size, 1 + THERMAL_MOMENT_COUNT):
            raise ValueError(
                f"expected processor sums (settings, 38); actual {sums.shape}"
            )
        return cls(axis.copy(), sums[:, 0].copy(), sums[:, 1:].copy())


def _validate_sums(sums: ThermalChainSums) -> None:
    if sums.axis.ndim != 1 or sums.axis.size == 0:
        raise ValueError(
            f"expected nonempty axis (settings,); actual {sums.axis.shape}"
        )
    count = sums.axis.size
    if sums.shot_count.shape != (count,) or sums.feature_sums.shape != (
        count,
        THERMAL_MOMENT_COUNT,
    ):
        raise ValueError(
            f"expected shot_count ({count},) and feature_sums ({count}, 37); "
            f"actual {sums.shot_count.shape} and {sums.feature_sums.shape}"
        )
    if any(
        array.dtype.kind not in "fiu"
        for array in (sums.axis, sums.shot_count, sums.feature_sums)
    ):
        raise ValueError(
            "expected real numeric arrays; actual nonreal/nonnumeric dtype"
        )
    if not (
        np.all(np.isfinite(sums.axis))
        and np.all(np.isfinite(sums.shot_count))
        and np.all(np.isfinite(sums.feature_sums))
    ):
        raise ValueError(
            "expected finite axis, shot_count and feature_sums; actual non-finite"
        )
    if np.any(sums.shot_count <= 0) or np.any(
        sums.shot_count != np.floor(sums.shot_count)
    ):
        raise ValueError(
            "expected positive integer shot counts; actual invalid shot_count"
        )


@dataclass(frozen=True)
class SettingScore:
    """Dimensionless score of one setting, without clipping to physical ranges.

    setting: Coordinate from ThermalChainSums.axis, in caller-defined units.
    score: y=4ρ², larger means greater first-pair g/e contrast.
    score_error: One-sigma sampling error 8|ρ| sqrt((1+ρ²)/n).
    correlation: ρ=c1/v, first-pair covariance / R1 variance on the main axis.
    qnd: QND=(c1*c2+c2*c3+c3*c4)/(c1²+c2²+c3²), with cj the R1,R(1+j)
        covariance projected onto the main axis. Not restricted to [0,1].
    qnd_error: One-sigma QND error v/sqrt(n) times the Euclidean norm of
        QND\'s analytic gradient with respect to c1..c4.
    mist: Largest eigenvalue of S(15,16)-S(1,2), divided by R1 variance v.
    excluded: True exactly when mist > |correlation|; omitted from ranking.
    """

    setting: float
    score: float
    score_error: float
    correlation: float
    qnd: float
    qnd_error: float
    mist: float
    excluded: bool


@dataclass(frozen=True)
class ThermalChainScore:
    """Thermal-chain results in acquisition order, with an eligible ranking.

    settings: One SettingScore per saved setting, including excluded settings.
    ranking: Eligible setting indices, best first by score-score_error. Exact
        ties keep acquisition order. Empty when all settings are excluded;
        otherwise ranking[0] is the best setting's index.
    """

    settings: tuple[SettingScore, ...]
    ranking: tuple[int, ...]


def thermal_chain_score(sums: ThermalChainSums) -> ThermalChainScore:
    """Score thermally reset 16-read shots from additive saved moments.

    sums contains independent shots that each relax to thermal equilibrium
    and then read 16 times without a drive. Divide each sum by its shot count.
    Centre each pair using its own read-position means, then symmetrise its
    covariance to obtain S(a,b). The main axis is the unit eigenvector of the
    largest eigenvalue of S(1,2)+S(1,3)+S(1,4)+S(1,5). Project R1 variance
    and those four covariances onto it to obtain v and c1..c4. A constant IQ
    offset cancels through read-position centring.

    Return all settings and the nonexcluded ranking, without modifying sums.
    SettingScore documents the score, errors, QND and normalised MIST amount.
    No fit, iterative solver or finite difference is used. Invalid saved sums,
    projected R1 variance <=0 or zero first-read covariance norm raise
    ValueError before scoring.
    """
    _validate_sums(sums)
    settings = tuple(
        _score_setting(float(setting), float(shots), features / shots)
        for setting, shots, features in zip(
            sums.axis, sums.shot_count, sums.feature_sums, strict=True
        )
    )
    eligible = [index for index, setting in enumerate(settings) if not setting.excluded]
    ranking = tuple(
        sorted(
            eligible,
            key=lambda index: settings[index].score - settings[index].score_error,
            reverse=True,
        )
    )
    return ThermalChainScore(settings, ranking)


def _largest_eigenvalue(matrix: NDArray[np.float64]) -> float:
    return float((matrix[0, 0] + matrix[1, 1]) / 2) + hypot(
        float((matrix[0, 0] - matrix[1, 1]) / 2), float(matrix[0, 1])
    )


def _score_setting(
    setting: float, shots: float, moments: NDArray[np.float64]
) -> SettingScore:
    means = moments[:14].reshape(7, 2)
    variance = moments[[14, 15, 15, 16]].reshape(2, 2) - np.outer(means[0], means[0])
    cross = moments[17:].reshape(5, 2, 2) - np.einsum(
        "ki,kj->kij", means[[0, 0, 0, 0, 5]], means[[1, 2, 3, 4, 6]]
    )
    symmetric = (cross + cross.transpose(0, 2, 1)) / 2
    main = symmetric[:4].sum(axis=0)
    # A real symmetric 2x2 matrix has this analytic principal-axis angle.
    angle = atan2(float(2 * main[0, 1]), float(main[0, 0] - main[1, 1])) / 2
    axis = np.array([cos(angle), sin(angle)])
    covariances = np.einsum("i,kij,j->k", axis, symmetric[:4], axis)
    v = float(axis @ variance @ axis)
    denominator = float(covariances[:3] @ covariances[:3])
    if v <= 0 or denominator == 0:
        raise ValueError(
            f"setting {setting}: expected variance > 0 and covariance norm > 0; "
            f"actual v={v}, D={denominator}"
        )
    rho = float(covariances[0]) / v
    qnd = float(covariances[:3] @ covariances[1:]) / denominator
    c1, c2, c3, c4 = covariances
    gradient = (
        np.array(
            [c2 - 2 * c1 * qnd, c1 + c3 - 2 * c2 * qnd, c2 + c4 - 2 * c3 * qnd, c3]
        )
        / denominator
    )
    score = 4 * rho**2
    score_error = 8 * abs(rho) * sqrt((1 + rho**2) / shots)
    qnd_error = v / sqrt(shots) * float(np.linalg.norm(gradient))
    mist = _largest_eigenvalue(symmetric[4] - symmetric[0]) / v
    return SettingScore(
        setting=setting,
        score=score,
        score_error=score_error,
        correlation=rho,
        qnd=qnd,
        qnd_error=qnd_error,
        mist=mist,
        excluded=mist > abs(rho),
    )
