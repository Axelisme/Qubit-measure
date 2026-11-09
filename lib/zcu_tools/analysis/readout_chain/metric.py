"""Readout information from saved 32-read block moments."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._estimator import (
    UnfittableMoments,
    one_step_fit,
    physical_coordinates,
)
from ._information import pair_information as _pair_information
from ._information import pair_information_gradient as _pair_information_gradient
from ._model import MOMENT_COUNT, FloatArray, stationary_population


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
    """One-step readout fit of one setting. Every value is finite.

    setting: the setting's coordinate from ReadoutSums.axis, in its units.
    metric: read-drive-read information for exchange of the two most
        populated fitted states, evaluated at equal pair populations and
        uniform landing within that pair. Dimensionless; larger is better.
    standard_error: one-sigma delta-method error of metric, >= 0, using the
        start-model covariance and the one-step estimator, including its
        fixed weak priors. Asymptotic in block count; not a fit-bias bound.
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
    separation: float
    retention: tuple[float, float]
    third_population: float


@dataclass(frozen=True)
class SettingFailure:
    """A setting whose saved moments the estimator cannot fit.

    setting: the setting's coordinate from ReadoutSums.axis, in its units.
    reason: the estimator's message, for example "readout moments have no
        identifiable main-axis signal", "Singular matrix" or "non-finite
        one-step readout fit".
    """

    setting: float
    reason: str


@dataclass(frozen=True)
class ReadoutMetric:
    """Per-setting results for F settings, keyed by setting index 0..F-1.

    fits: index -> SettingFit for each setting the estimator fitted, in
        acquisition order.
    failures: index -> SettingFailure for each setting it could not fit, in
        acquisition order. Every index is in exactly one of fits and failures.
    ranking: the indices in fits, metric descending; failed settings are
        absent. Exact ties keep acquisition order. Empty when nothing fits.
    """

    fits: dict[int, SettingFit]
    failures: dict[int, SettingFailure]
    ranking: tuple[int, ...]


def readout_metric(sums: ReadoutSums) -> ReadoutMetric:
    """Fit three states per setting with a closed-form start and one GN step.

    sums holds the block moments defined on ReadoutSums. Assume a stationary
    contiguous stream, isotropic independent Gaussian read noise, and transition
    R_ij = λ_i δ_ij + (1-λ_i) ρ_j. Use exact contiguous 32-read block covariance
    at the start and the fixed weak third-state priors. Ignore stream-end edges.
    No hardware, file access or input mutation occurs. The metric discards the
    least populated state only when evaluating the fitted dominant pair.
    Each setting is fitted alone. A setting whose moments cannot be fitted (no
    identifiable main-axis signal, a singular matrix or a non-finite one-step
    result) becomes a SettingFailure; the other settings are unaffected.
    Raise ValueError for malformed/non-finite sums or fewer than two blocks.
    """
    # The frozen record's arrays remain mutable after construction.
    _validate_sums(sums)
    fits: dict[int, SettingFit] = {}
    failures: dict[int, SettingFailure] = {}
    for index, blocks in enumerate(sums.block_count):
        setting = float(sums.axis[index])
        features = sums.feature_sums[index] / blocks
        try:
            fits[index] = _fit_setting(setting, features, float(blocks))
        except (UnfittableMoments, np.linalg.LinAlgError) as error:
            # Isolate: one setting the estimator cannot fit keeps the others.
            failures[index] = SettingFailure(setting, str(error))
    ranking = sorted(fits, key=lambda index: -fits[index].metric)
    return ReadoutMetric(fits=fits, failures=failures, ranking=tuple(ranking))


def _fit_setting(setting: float, features: FloatArray, blocks: float) -> SettingFit:
    """Fit one setting's averaged moments and evaluate its dominant-pair metric.

    The error propagates the one-step sampling covariance through the analytic
    gradient of the pair metric. Raise UnfittableMoments
    or LinAlgError when the moments cannot be fitted or the fit is non-finite.
    """
    coordinates, covariance = one_step_fit(features, blocks)
    model = physical_coordinates(coordinates)
    if not model.is_finite() or model.variance <= 0:
        raise UnfittableMoments("non-finite one-step readout fit")
    population = stationary_population(model)
    order = np.argsort(-population, kind="stable")
    pair = order[:2]

    gradient = _pair_information_gradient(coordinates, pair)
    distance = np.linalg.norm(model.centres[pair[1]] - model.centres[pair[0]])
    fit = SettingFit(
        setting=setting,
        metric=_pair_information(model, pair),
        standard_error=float(np.sqrt(max(float(gradient @ covariance @ gradient), 0))),
        separation=float(distance / np.sqrt(model.variance)),
        retention=(float(model.retention[pair[0]]), float(model.retention[pair[1]])),
        third_population=float(population[order[2]]),
    )
    numbers = [fit.metric, fit.standard_error, fit.separation, fit.third_population]
    if not np.isfinite([*numbers, *fit.retention]).all():
        raise UnfittableMoments("non-finite one-step readout result")
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
