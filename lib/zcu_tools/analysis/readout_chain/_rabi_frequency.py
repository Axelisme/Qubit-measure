"""Frequency initialization for the built-in K-cycle Rabi model."""

import numpy as np

from .cycle import CycleSums
from .model import Array, CycleModel


class RabiCycleModel(CycleModel):
    """Built-in Rabi curve with a frequency start scored on the full chain."""

    def frequency_start(self, sums: CycleSums, means: Array, initial: float) -> float:
        """Choose one joint start from the caller's candidate and a frequency grid.

        The grid has 16 samples per inverse span and reaches the nominal
        Nyquist limit of the distinct sweep points. Evaluate cosines at the
        actual coordinates, including any repeated points. Match the drive's
        zero-referenced phase and allow a free offset and amplitude per feature.
        Rank trial frequencies by the count-weighted explained power in the
        supported directions of pooled within-point covariance. Ties keep the
        earliest candidate. If the sweep has only one coordinate, there is no
        frequency grid. These scores do not supply the joint optimizer's weights.
        """
        points = np.unique(sums.axis)
        if points.size == 1:
            return initial
        span = points[-1] - points[0]
        nyquist = (points.size - 1) / (2 * span)
        frequencies = np.linspace(0, nyquist, 8 * (points.size - 1) + 1)
        frequencies[0] = initial

        weights = np.sqrt(sums.cycle_count)
        whitened = _whitened_variation(sums, means) * weights[:, None]
        best_frequency, best_power = initial, -1.0
        # Score blocks of trials to keep memory linear in the sweep length.
        for offset in range(0, len(frequencies), 128):
            trials = frequencies[offset : offset + 128]
            cosines = np.cos(2 * np.pi * trials[:, None] * sums.axis)
            cosines -= np.average(cosines, axis=1, weights=sums.cycle_count)[:, None]
            cosines *= weights
            power = _explained_power(cosines, whitened)
            index = np.argmax(power)
            if power[index] > best_power:
                best_frequency, best_power = float(trials[index]), float(power[index])
        return best_frequency


def _explained_power(cosines: Array, whitened: Array) -> Array:
    norm_sq = np.sum(cosines**2, axis=1)
    explained = np.sum((cosines @ whitened) ** 2, axis=1)
    return np.divide(
        explained, norm_sq, out=np.zeros_like(explained), where=norm_sq > 0
    )


def _whitened_variation(sums: CycleSums, means: Array) -> Array:
    # Balance the raw products before centering: the feature columns have
    # different units and can carry large offsets. Discard only directions
    # whose pooled variance is unresolved at the precision of those products.
    scale = np.sqrt(np.abs(sums.product_sum.diagonal()))
    scale = np.where(scale > 0, scale, 1)
    scaled = means / scale
    covariance = (sums.product_sum / scale) / scale[:, None]
    covariance -= scaled.T @ (sums.cycle_count[:, None] * scaled)
    tolerance = 8 * max(covariance.shape[0], len(means)) * np.finfo(float).eps
    if not np.allclose(covariance, covariance.T, rtol=0, atol=tolerance):
        raise ValueError("pooled Rabi covariance must be symmetric")
    values, directions = np.linalg.eigh((covariance + covariance.T) / 2)
    if values[0] < -tolerance:
        raise ValueError("pooled Rabi covariance must be positive semidefinite")
    supported = values > tolerance
    return (scaled - scaled[0]) @ (
        directions[:, supported] / np.sqrt(values[supported])
    )
