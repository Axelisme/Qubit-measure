"""Closed-form score of thermally reset 16-read shots."""

from dataclasses import dataclass
from math import atan2, cos, hypot, sin, sqrt

import numpy as np
from numpy.typing import NDArray

THERMAL_MOMENT_COUNT = 107


@dataclass(frozen=True)
class ThermalChainSums:
    """Saved additive moments of thermally reset shots, in acquisition order.

    axis: Shape (F,), finite setting coordinates in caller-defined units.
    shot_count: Shape (F,), positive integer shot counts, stored as float64.
    feature_sums: Shape (F, 107), finite sums: I,Q of R1..R16;
        R1 I²,IQ,Q²; II′,IQ′,QI′,QQ′ for all 15 adjacent pairs, followed by
        (R1,R3),(R1,R4),(R1,R5). Units are ADC or ADC² by moment degree.
        All four cross components are saved without analysis weighting.

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
        """Copy axis (F,) and processor sums (F,108): shot count then moments.

        axis uses caller-defined units. Raise ValueError for empty/mismatched
        shapes, non-finite data or nonpositive/noninteger shot counts.
        """
        if axis.ndim != 1 or sums.shape != (axis.size, 1 + THERMAL_MOMENT_COUNT):
            raise ValueError(
                f"expected processor sums (settings, 108); actual {sums.shape}"
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
            f"expected shot_count ({count},) and feature_sums ({count}, 107); "
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
    score: ρ² G (|ρ|/(|ρ|+max(M,0)))²; zero at ρ=0. A heuristic utility,
        not a leakage probability or a physical information optimum.
    score_error: Model-conditional approximate scale propagating only ρ and
        QND at fixed M and main axis, ignoring their shared covariance.
        Not a general one-sigma guarantee or a selection correction.
    amplification: G, symmetric two-state 15-lag information ratio averaged
        over determinant 0..1, with effective rather than calibrated SNR.
    correlation: ρ=c1/v, first-pair covariance / R1 variance on the main axis.
    qnd: QND=(c1*c2+c2*c3+c3*c4)/(c1²+c2²+c3²), with cj the R1,R(1+j)
        covariance projected onto the main axis. Not restricted to [0,1].
    qnd_error: Model-conditional approximate scale v/sqrt(n) times the norm
        of QND's analytic gradient with respect to c1..c4.
    mist: M=λmax(Σ(r-8)/20 S(r,r+1))/v for r=1..15. This diagnoses
        within-chain correlation growth, including two-state repopulation;
        it is not an identifiable physical leakage probability or rate.
    """

    setting: float
    score: float
    score_error: float
    correlation: float
    amplification: float
    qnd: float
    qnd_error: float
    mist: float


@dataclass(frozen=True)
class ThermalChainScore:
    """Thermal-chain results in acquisition order, ranked by heuristic score.

    settings: One SettingScore per saved setting in acquisition order.
    ranking: All setting indices by descending score. Exact ties keep
        acquisition order. ranking[0] is the best setting's index; errors
        do not enter ranking and no second threshold excludes settings.
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

    Return all settings and their score ranking, without modifying sums.
    SettingScore documents the score, errors, QND and correlation-growth M.
    G uses eta=clip(QND,0.001,0.995), s=min(|rho|/eta,0.999), q=s/(1-s).
    Integrate J.T solve(V,J) over determinant 0..1 with fixed 24-point
    Gauss-Legendre quadrature, divided by its determinant=0 value. The
    15-lag two-state covariance defines J,V; the scaled solve extends G
    continuously to s=0. This is a model information proxy, not an equality
    between thermal-reset shots and a driven chain.

    Error propagates SE(rho)=sqrt((1+rho²)/n) and the documented QND scale.
    Analytic partials G_eta,G_s are evaluated at clipped eta,s. Model slopes
    are dG/drho=sign(rho)*G_s/eta and dG/dQND=G_eta-s*G_s/eta, using clipped
    s even above its cap. The rho² and M-penalty prefactor uses raw rho.
    These slopes treat clip Jacobians as 1: at/outside a clip boundary they
    are not true derivatives of the clipped output. Score and error have
    continuous limit zero at rho=0. M and main-axis uncertainty are ignored.

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
    ranking = tuple(
        sorted(
            range(len(settings)), key=lambda index: settings[index].score, reverse=True
        )
    )
    return ThermalChainScore(settings, ranking)


def _amplification(rho: float, qnd: float) -> tuple[float, float, float]:
    """Return G and model slopes in rho,QND at clipped eta,s (clip Jacobian=1)."""
    eta = float(np.clip(qnd, 0.001, 0.995))
    s = min(abs(rho) / eta, 0.999)
    q = s / (1 - s)
    nodes, weights = np.polynomial.legendre.leggauss(24)
    u = eta * (nodes + 1)[:, None, None] / 2
    j = np.arange(1, 16)
    distance = np.abs(j[:, None] - j[None, :])
    total = j[:, None] + j[None, :]
    b = (1 + u**2) / (1 - u**2)
    near, far = u**distance, u**total
    transition = (distance + b) * near - (total + b) * far
    covariance = np.eye(15) + 2 * q * (near + far) + q**2 * transition
    # Dividing J by q*eta cancels from the ratio and extends it to q=0.
    signal = j * u[:, 0, :] ** (j - 1)
    solved = np.linalg.solve(covariance, signal[..., None])[..., 0]
    information = np.einsum("ki,ki->k", signal, solved)
    # All quadrature nodes are positive, so these powers have no 0/0 boundary.
    near_u, far_u = distance * u ** (distance - 1), total * u ** (total - 1)
    b_u = 4 * u / (1 - u**2) ** 2
    covariance_u = 2 * q * (near_u + far_u) + q**2 * (
        (distance + b) * near_u - (total + b) * far_u + b_u * (near - far)
    )
    covariance_q = 2 * (near + far) + 2 * q * transition
    signal_u = j * (j - 1) * u[:, 0, :] ** (j - 2)
    information_u = 2 * np.einsum("ki,ki->k", signal_u, solved) - np.einsum(
        "ki,kij,kj->k", solved, covariance_u, solved
    )
    information_q = -np.einsum("ki,kij,kj->k", solved, covariance_q, solved)
    # At determinant=0 the scaled information is exactly 1/(1+q)^2.
    integral = float(weights @ information / 2)
    gain = integral * (1 + q) ** 2
    gain_eta = float(weights @ (information_u * (nodes + 1) / 2) / 2) * (1 + q) ** 2
    gain_s = (
        2 * (1 + q) * integral + (1 + q) ** 2 * float(weights @ information_q / 2)
    ) / (1 - s) ** 2
    sign = float(np.sign(rho))
    return gain, sign * gain_s / eta, gain_eta - s * gain_s / eta


def _largest_eigenvalue(matrix: NDArray[np.float64]) -> float:
    return float((matrix[0, 0] + matrix[1, 1]) / 2) + hypot(
        float((matrix[0, 0] - matrix[1, 1]) / 2), float(matrix[0, 1])
    )


def _score_setting(
    setting: float, shots: float, moments: NDArray[np.float64]
) -> SettingScore:
    means = moments[:32].reshape(16, 2)
    variance = moments[[32, 33, 33, 34]].reshape(2, 2) - np.outer(means[0], means[0])
    adjacent = moments[35:95].reshape(15, 2, 2) - np.einsum(
        "ki,kj->kij", means[:-1], means[1:]
    )
    early = moments[95:].reshape(3, 2, 2) - np.einsum("i,kj->kij", means[0], means[2:5])
    symmetric = np.concatenate([adjacent[:1], early])
    symmetric = (symmetric + symmetric.transpose(0, 2, 1)) / 2
    main = symmetric.sum(axis=0)
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
    difference = np.einsum("k,kij->ij", (np.arange(1, 16) - 8) / 20, adjacent)
    mist = _largest_eigenvalue((difference + difference.T) / 2) / v
    amplification, gain_rho, gain_qnd = _amplification(rho, qnd)
    qnd_error = v / sqrt(shots) * float(np.linalg.norm(gradient))
    score, score_error = 0.0, 0.0
    if rho:
        positive_mist = max(mist, 0)
        magnitude = abs(rho)
        prefactor = rho**2 * (magnitude / (magnitude + positive_mist)) ** 2
        prefactor_rho = float(np.sign(rho)) * (
            2
            * magnitude**3
            * (magnitude + 2 * positive_mist)
            / (magnitude + positive_mist) ** 3
        )
        score = prefactor * amplification
        score_rho = prefactor_rho * amplification + prefactor * gain_rho
        score_qnd = prefactor * gain_qnd
        score_error = hypot(
            score_rho * sqrt((1 + rho**2) / shots), score_qnd * qnd_error
        )
    return SettingScore(
        setting=setting,
        score=score,
        score_error=score_error,
        correlation=rho,
        amplification=amplification,
        qnd=qnd,
        qnd_error=qnd_error,
        mist=mist,
    )
