"""Readout information from saved 32-read block moments."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._estimator import (
    UnfittableMoments,
    bounded_fit,
    physical_coordinates,
)
from ._information import pair_correlation, pair_correlation_gradient
from ._information import pair_information as _pair_information
from ._information import pair_information_gradient as _pair_information_gradient
from ._model import (
    LAG_PAIRS,
    MEANS,
    MOMENT_COUNT,
    SAME_READ,
    FloatArray,
    ModelParameters,
    contiguous_covariance,
    stationary_population,
)


@dataclass(frozen=True)
class ReadoutSums:
    """Saved readout moments for F settings in acquisition order.

    axis: Shape (F,), setting coordinates in caller-defined units.
    block_count: Shape (F,), integer number of contiguous 32-read blocks, >= 2.
    feature_sums: Shape (F, MOMENT_COUNT), MOMENT_COUNT = 113: sums of the
        per-block averages. Columns:
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
        """Copy (F, 1 + MOMENT_COUNT) processor output: count, then moment sums.

        axis has shape (F,) and caller-defined units. Raise ValueError for
        malformed/non-finite output or counts below two or non-integer.
        """
        columns = 1 + MOMENT_COUNT
        if axis.ndim != 1 or sums.shape != (len(axis), columns):
            raise ValueError(f"processor sums must have shape (settings, {columns})")
        return cls(axis.copy(), sums[:, 0].copy(), sums[:, 1:].copy())


@dataclass(frozen=True)
class SettingFit:
    """Bounded three-state readout fit of one setting. Every value is finite.

    setting: the setting's coordinate from ReadoutSums.axis, in its units.
    metric: information per read about a weak drive in an infinite symmetric
        ...DRDRDR... chain, combining all lag j>=1 read products with GLS and
        their exact long-run covariance. Use the two most populated fitted
        states at equal populations, q=d²/(4σ²), and QND η=R̃_aa+R̃_bb-1 of their
        row-renormalised transition block. Dimensionless; larger is better.
    standard_error: one-sigma delta-method sampling error of metric, >= 0,
        using the start-model covariance, final analytic Jacobian and fixed
        weak priors. Asymptotic in block count; not a fit-bias bound.
    qnd: QND η=R̃_aa+R̃_bb-1 for the dominant pair, with each row of its
        transition block renormalised within the pair. Dimensionless, 0<η<1;
        equals λ for an equal-retention pair with no third-state landing.
    qnd_error: one-sigma delta-method sampling error of qnd, >=0, using the
        same sampling covariance as standard_error and QND's analytic gradient.
        It is not a bias bound and excludes pair-selection uncertainty.
    separation: dominant-pair centre distance divided by the fitted per-I/Q
        Gaussian noise standard deviation (d/sigma).
    retention: read retention λ of the two dominant states, in descending
        stationary-population order; 0.01 <= λ <= 0.995. λ is the
        retain-versus-reset probability, not a diagonal transition.
        Individual λ may be unidentifiable when the third state is unseen.
    third_population: stationary population of the remaining fitted state
        (the least populated), between zero and one third.
    """

    setting: float
    metric: float
    standard_error: float
    qnd: float
    qnd_error: float
    separation: float
    retention: tuple[float, float]
    third_population: float


@dataclass(frozen=True)
class SettingFailure:
    """A setting whose saved moments the estimator cannot fit.

    setting: the setting's coordinate from ReadoutSums.axis, in its units.
    reason: the estimator's message, for example "fit hit the evaluation
        budget (1200 evaluations, status 0)", "Singular matrix", or
        "non-finite bounded readout fit". No-signal settings are not failures.
    """

    setting: float
    reason: str


@dataclass(frozen=True)
class ReadoutMetric:
    """Per-setting results for F settings, keyed by setting index 0..F-1.

    fits: index -> SettingFit for each setting the estimator fitted, in
        acquisition order.
    no_signal: indices stopped by the pre-fit no-signal test, in acquisition
        order. Every index is in exactly one of fits, no_signal and failures.
    failures: index -> SettingFailure for each setting it could not fit, in
        acquisition order.
    ranking: the indices in fits, ordered by metric minus standard_error,
        largest first. Exact score ties keep acquisition order. No-signal and
        failed settings are absent. The best setting is ranking[0] when the
        ranking is nonempty; it need not have the largest metric.
    """

    fits: dict[int, SettingFit]
    no_signal: tuple[int, ...]
    failures: dict[int, SettingFailure]
    ranking: tuple[int, ...]


def rank_settings(fits: dict[int, SettingFit]) -> tuple[int, ...]:
    """Rank fitted indices by metric minus standard_error, largest first.

    fits must be in acquisition order. Exact score ties keep that order.
    Return an empty tuple for no fits. Do not modify fits or its values.
    The best setting is the first returned index; its metric need not be largest.
    """
    return tuple(
        sorted(
            fits,
            key=lambda index: fits[index].metric - fits[index].standard_error,
            reverse=True,
        )
    )


def readout_metric(sums: ReadoutSums) -> ReadoutMetric:
    """Fit three states per setting with a closed-form start and bounded trf.

    sums holds the block moments defined on ReadoutSums. Assume a stationary
    contiguous stream, isotropic independent Gaussian read noise, and transition
    R_ij = λ_i δ_ij + (1-λ_i) ρ_j. Use exact contiguous 32-read block covariance
    at the start, removing its least populated state's landing and renormalising
    landing for weights only. Keep the start and weak priors unchanged.
    Ignore stream-end edges.
    Bound λ in [.01, .995] and log landing ratios in [-10.8, 10.8]; retain
    the coordinate map's 1e-5 landing floor. No hardware, file access or input
    mutation occurs.

    Before fitting, sum symmetrised lag 1..4 covariance matrices into K and
    form v=(K_II,K_IQ,K_QQ). With B blocks, report no_signal when the largest
    eigenvalue of K is <=0 or v.T @ inv(Σ) @ v < 2*ln(B). Σ is v's sampling
    covariance at three coincident centres at the measured mean, noise variance
    from the same-read covariance trace/2, λ=.5, and uniform landing. Such
    settings are neither fitted nor ranked.

    The metric is all-lag GLS information per read about a weak drive in an
    infinite symmetric ...DRDRDR... chain, for the two most populated fitted
    states at equal populations. Set q=d²/(4σ²) and QND η=R̃_aa+R̃_bb-1 from
    their row-renormalised transition block, κ=(1+η²+q*(1-η²))/(2η),
    z=κ-sqrt(κ²-1). Then information=4*(η-z)²*(1-η²+η*z+3*η³*z-4*η²*z²) /
    ((1-η²)*(1-η*z)*(1-η²+4*η*z-5*z²-3*η²*z²+4*η*z³)). Larger is better.
    qnd reports QND; qnd_error propagates the same sampling covariance through
    QND's analytic gradient. The third population is a leakage diagnostic;
    all three states fit. Rank fitted settings by metric minus standard_error,
    largest first, keeping acquisition order on exact ties. The best setting
    need not have the largest metric. Ranking does not change fitted values.

    Singular matrices, non-finite fits and exhaustion of the 1200-evaluation
    budget become per-setting SettingFailure values without losing other
    settings. Every index is in exactly one of fits, no_signal and failures.
    Raise ValueError for malformed/non-finite sums or fewer than two blocks.
    """
    # The frozen record's arrays remain mutable after construction.
    _validate_sums(sums)
    fits: dict[int, SettingFit] = {}
    failures: dict[int, SettingFailure] = {}
    no_signal: list[int] = []
    for index, blocks in enumerate(sums.block_count):
        setting = float(sums.axis[index])
        features = sums.feature_sums[index] / blocks
        try:
            if _no_signal(features, float(blocks)):
                no_signal.append(index)
            else:
                fits[index] = _fit_setting(setting, features, float(blocks))
        except (UnfittableMoments, np.linalg.LinAlgError) as error:
            # Isolate: one setting the estimator cannot fit keeps the others.
            failures[index] = SettingFailure(setting, str(error))
    return ReadoutMetric(
        fits=fits,
        no_signal=tuple(no_signal),
        failures=failures,
        ranking=rank_settings(fits),
    )


def _no_signal(features: FloatArray, blocks: float) -> bool:
    mean = features[MEANS]
    lags = features[LAG_PAIRS].reshape(15, 2, 2)[:4] - np.outer(mean, mean)
    pooled = ((lags + lags.transpose(0, 2, 1)) / 2).sum(axis=0)
    if np.linalg.eigvalsh(pooled)[-1] <= 0:
        return True
    ii, _, qq = features[SAME_READ]
    variance = float(ii + qq - mean @ mean) / 2
    null = ModelParameters(
        np.tile(mean, (3, 1)), variance, np.full(3, 0.5), np.full(3, 1 / 3)
    )
    derivative = np.zeros((3, MOMENT_COUNT))
    for lag in range(4):
        column = LAG_PAIRS.start + 4 * lag
        derivative[0, column] = derivative[2, column + 3] = 1
        derivative[1, column + 1 : column + 3] = 0.5
    derivative[0, 0] = -8 * mean[0]
    derivative[1, :2] = -4 * mean[::-1]
    derivative[2, 1] = -8 * mean[1]
    covariance = derivative @ contiguous_covariance(null) @ derivative.T / blocks
    vector = pooled[np.array([0, 0, 1]), np.array([0, 1, 1])]
    statistic = float(vector @ np.linalg.solve(covariance, vector))
    return statistic < 2 * np.log(blocks)


def _fit_setting(setting: float, features: FloatArray, blocks: float) -> SettingFit:
    """Fit one setting's averaged moments and evaluate its dominant-pair metric.

    The error propagates the final-fit sampling covariance through the analytic
    gradient of the pair metric. Raise UnfittableMoments
    or LinAlgError when the moments cannot be fitted or the fit is non-finite.
    """
    coordinates, covariance = bounded_fit(features, blocks)
    model = physical_coordinates(coordinates)
    if not model.is_finite() or model.variance <= 0:
        raise UnfittableMoments("non-finite bounded readout fit")
    population = stationary_population(model)
    order = np.argsort(-population, kind="stable")
    pair = order[:2]

    gradient = _pair_information_gradient(coordinates, pair)
    qnd_gradient = pair_correlation_gradient(coordinates, pair)
    distance = np.linalg.norm(model.centres[pair[1]] - model.centres[pair[0]])
    fit = SettingFit(
        setting=setting,
        metric=_pair_information(model, pair),
        standard_error=float(np.sqrt(max(float(gradient @ covariance @ gradient), 0))),
        qnd=pair_correlation(model, pair),
        qnd_error=float(
            np.sqrt(max(float(qnd_gradient @ covariance @ qnd_gradient), 0))
        ),
        separation=float(distance / np.sqrt(model.variance)),
        retention=(float(model.retention[pair[0]]), float(model.retention[pair[1]])),
        third_population=float(population[order[2]]),
    )
    numbers = [
        fit.metric,
        fit.standard_error,
        fit.qnd,
        fit.qnd_error,
        fit.separation,
        fit.third_population,
    ]
    if not np.isfinite([*numbers, *fit.retention]).all():
        raise UnfittableMoments("non-finite bounded readout result")
    return fit


def _validate_sums(sums: ReadoutSums) -> None:
    if sums.axis.ndim != 1 or sums.axis.size == 0:
        raise ValueError("axis must have nonempty shape (settings,)")
    points = len(sums.axis)
    for name, values, shape in (
        ("axis", sums.axis, (points,)),
        ("block_count", sums.block_count, (points,)),
        ("feature_sums", sums.feature_sums, (points, MOMENT_COUNT)),
    ):
        if values.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
    if np.any(sums.block_count < 2):
        raise ValueError("each setting must contain at least two blocks")
    if np.any(sums.block_count != np.floor(sums.block_count)):
        raise ValueError("block counts must be integer-valued")
