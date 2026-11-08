"""Stationary three-state Gaussian moments and contiguous-block covariance.

The tail is summed geometrically in steps of 32, rather than truncated at a
retention-dependent horizon. Work per setting therefore stays bounded near λ=1.
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Layout = tuple[IntArray, IntArray, IntArray]
Model = tuple[FloatArray, FloatArray, FloatArray, FloatArray]


def moment_layout() -> Layout:
    """Times, IQ components and degrees in the saved 113-column order."""
    columns = [
        ((0, 0),),
        ((0, 1),),
        ((0, 0), (0, 0)),
        ((0, 0), (0, 1)),
        ((0, 1), (0, 1)),
    ]
    columns += [((0, a), (k, b)) for k in range(1, 16) for a in (0, 1) for b in (0, 1)]
    columns += [
        ((0, a), (0, b), (k, c))
        for k in range(1, 5)
        for a, b in ((0, 0), (0, 1), (1, 1))
        for c in (0, 1)
    ]
    columns += [
        ((0, a), (k, b), (k, c))
        for k in range(1, 5)
        for a in (0, 1)
        for b, c in ((0, 0), (0, 1), (1, 1))
    ]
    times = np.zeros((113, 3), dtype=np.int64)
    components = np.zeros_like(times)
    degrees = np.array([len(column) for column in columns], dtype=np.int64)
    for i, column in enumerate(columns):
        for j, (time, component) in enumerate(column):
            times[i, j] = time
            components[i, j] = component
    return times, components, degrees


LAYOUT = moment_layout()


def stationary_population(theta: FloatArray) -> FloatArray:
    """Stationary population of R_ij = λ_i δ_ij + (1-λ_i)ρ_j."""
    population = theta[10:13] / (1 - theta[7:10])
    return population / population.sum()


def model_arrays(theta: FloatArray) -> Model:
    """Transition powers, summed tail powers, Gaussian emissions and population."""
    centres = theta[:6].reshape(3, 2)
    variance = theta[6]
    retention = theta[7:10]
    landing = theta[10:13]
    transition = np.diag(retention) + (1 - retention[:, None]) * landing
    population = stationary_population(theta)
    powers = np.empty((64, 3, 3))
    powers[0] = np.eye(3)
    for gap in range(1, 64):
        powers[gap] = powers[gap - 1] @ transition
    projector = np.tile(population, (3, 1))
    # Σ_{j>=0}(R^(g+32j)-Π) = (R^g-Π)(I-R^32+Π)^(-1).
    resolvent = np.linalg.solve(
        np.eye(3) - powers[32] + projector, np.eye(3) - projector
    )
    tail = (powers - projector) @ resolvent
    gaussian = np.ones((7, 2, 3))
    gaussian[1] = centres.T
    for degree in range(2, 7):
        gaussian[degree] = (
            centres.T * gaussian[degree - 1]
            + (degree - 1) * variance * gaussian[degree - 2]
        )
    emissions = gaussian[:, 0, None, :] * gaussian[None, :, 1, :]
    return powers, tail, emissions, population


@njit(cache=True)
def product_expectation(
    times: IntArray, components: IntArray, model: Model, boundary: int
) -> float:
    """Sorted-time Gaussian product; boundary>=0 sums the separating tail gap."""
    powers, tail, emissions, population = model
    weights = population.copy()
    i = 0
    previous = times[0]
    while i < len(times):
        time = times[i]
        if i > 0:
            gap = time - previous
            propagator = tail[gap] if previous < boundary <= time else powers[gap]
            weights = weights @ propagator
        n_i, n_q = 0, 0
        while i < len(times) and times[i] == time:
            if components[i] == 0:
                n_i += 1
            else:
                n_q += 1
            i += 1
        weights *= emissions[n_i, n_q]
        previous = time
    return float(weights.sum())


@njit(cache=True)
def moment_means_kernel(layout: Layout, model: Model) -> FloatArray:
    times, components, degrees = layout
    result = np.empty(113)
    for i in range(113):
        degree = degrees[i]
        result[i] = product_expectation(
            times[i, :degree], components[i, :degree], model, -1
        )
    return result


def moment_means(theta: FloatArray) -> FloatArray:
    """Exact 113 stationary feature expectations for the 13-value model vector."""
    return moment_means_kernel(LAYOUT, model_arrays(theta))


@njit(cache=True)
def joint_moment(
    indices: tuple[int, int, int, int], layout: Layout, model: Model
) -> float:
    """E[f_a(t) f_b(t+h)], or its geometric positive-shift tail."""
    a, b, shift, boundary = indices
    times, components, degrees = layout
    count_a, count_b = degrees[a], degrees[b]
    count = count_a + count_b
    ts = np.empty(count, dtype=np.int64)
    cs = np.empty(count, dtype=np.int64)
    ts[:count_a], cs[:count_a] = times[a, :count_a], components[a, :count_a]
    ts[count_a:], cs[count_a:] = times[b, :count_b] + shift, components[b, :count_b]
    # At most six factors; insertion sort avoids allocating a general argsort.
    for i in range(1, count):
        time, component = ts[i], cs[i]
        j = i - 1
        while j >= 0 and ts[j] > time:
            ts[j + 1], cs[j + 1] = ts[j], cs[j]
            j -= 1
        ts[j + 1], cs[j + 1] = time, component
    return product_expectation(ts, cs, model, boundary)


@njit(cache=True)
def anchor_count(span_a: int, span_b: int, shift: int) -> int:
    """Number of valid a anchors whose shifted b anchor is valid modulo 32."""
    count = 0
    for time in range(32 - span_a):
        if (time + shift) % 32 < 32 - span_b:
            count += 1
    return count


@njit(cache=True)
def contiguous_covariance_kernel(layout: Layout, model: Model) -> FloatArray:
    """B times covariance of feature averages over contiguous 32-read blocks."""
    times, _, _ = layout
    spans = np.empty(113, dtype=np.int64)
    for i in range(113):
        spans[i] = times[i].max()
    means = moment_means_kernel(layout, model)
    result = np.empty((113, 113))
    for a in range(113):
        for b in range(a, 113):
            centre = means[a] * means[b]
            total = anchor_count(spans[a], spans[b], 0) * (
                joint_moment((a, b, 0, -1), layout, model) - centre
            )
            for shift in range(1, 48):
                boundary = shift if shift >= 16 else -1
                subtract = 0.0 if shift >= 16 else centre
                total += anchor_count(spans[a], spans[b], shift) * (
                    joint_moment((a, b, shift, boundary), layout, model) - subtract
                )
                total += anchor_count(spans[b], spans[a], shift) * (
                    joint_moment((b, a, shift, boundary), layout, model) - subtract
                )
            result[a, b] = total / ((32 - spans[a]) * (32 - spans[b]))
            result[b, a] = result[a, b]
    return result


def contiguous_covariance(theta: FloatArray) -> FloatArray:
    """Exact asymptotic B*Cov(feature means), including inter-block dependence.

    Lags 0..15 contain overlapping read noise. For h>=16 the factors do not
    overlap. Sum those lags in 32 residue classes using the Markov resolvent.
    The sum is exact up to floating-point error and ignores stream-end edges.
    """
    return contiguous_covariance_kernel(LAYOUT, model_arrays(theta))
