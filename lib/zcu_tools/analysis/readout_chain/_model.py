"""Stationary three-state Gaussian moments and contiguous-block covariance.

The tail is summed geometrically in steps of 32, rather than truncated at a
retention-dependent horizon. Work per setting therefore stays bounded near λ=1.
"""

from functools import cache

import numpy as np
from numba import njit
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Layout = tuple[IntArray, IntArray, IntArray]
Model = tuple[FloatArray, FloatArray, FloatArray, FloatArray]
OverlapLayout = tuple[IntArray, IntArray, IntArray]


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
def product_expectation(times: IntArray, components: IntArray, model: Model) -> float:
    """Stationary expectation of a sorted-time Gaussian product."""
    powers, _, emissions, population = model
    weights = population.copy()
    i = 0
    previous = times[0]
    while i < len(times):
        time = times[i]
        if i > 0:
            gap = time - previous
            weights = weights @ powers[gap]
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
            times[i, :degree], components[i, :degree], model
        )
    return result


def moment_means(theta: FloatArray) -> FloatArray:
    """Exact 113 stationary feature expectations for the 13-value model vector."""
    return moment_means_kernel(LAYOUT, model_arrays(theta))


@njit(cache=True)
def joint_layout(a: int, b: int, shift: int, layout: Layout) -> IntArray:
    """Group a joint product into (gap, I degree, Q degree) time nodes."""
    times, components, degrees = layout
    factors = []
    for column, offset in ((a, 0), (b, shift)):
        for j in range(degrees[column]):
            factors.append((times[column, j] + offset, components[column, j]))
    factors.sort()
    nodes = np.zeros((4, 3), dtype=np.int64)
    node, previous = -1, -1
    for time, component in factors:
        if time != previous:
            node += 1
            # The first node has no propagator.
            nodes[node, 0] = 0 if node == 0 else time - previous
            previous = time
        nodes[node, 1 + component] += 1
    return nodes


@njit(cache=True)
def overlap_layout(layout: Layout) -> OverlapLayout:
    """Precompute finite overlapping products and periodic anchor-pair counts."""
    times, _, _ = layout
    spans = np.array([times[i].max() for i in range(113)])
    size = 0
    for a in range(113):
        for b in range(a, 113):
            size += 1 + spans[a] + spans[b]
    pairs = np.empty((size, 3), dtype=np.int64)
    nodes = np.empty((size, 4, 3), dtype=np.int64)
    counts = np.zeros((16, 16, 32), dtype=np.int64)
    for span_a in range(16):
        for span_b in range(16):
            for shift in range(32):
                for time in range(32 - span_a):
                    if (time + shift) % 32 < 32 - span_b:
                        counts[span_a, span_b, shift] += 1
    row = 0
    for a in range(113):
        for b in range(a, 113):
            pairs[row] = a, b, counts[spans[a], spans[b], 0]
            nodes[row] = joint_layout(a, b, 0, layout)
            row += 1
            for left, right in ((a, b), (b, a)):
                for shift in range(1, spans[left] + 1):
                    pairs[row] = a, b, counts[spans[left], spans[right], shift]
                    nodes[row] = joint_layout(left, right, shift, layout)
                    row += 1
    return pairs, nodes, counts


@cache
def covariance_layout() -> OverlapLayout:
    """Build the parameter-independent overlap plan once, on first use."""
    return overlap_layout(LAYOUT)


@njit(cache=True)
def overlap_expectation(nodes: IntArray, model: Model) -> float:
    """Contract up to four time nodes with three scalar state weights."""
    powers, _, emissions, population = model
    w0, w1, w2 = population[0], population[1], population[2]
    for node in range(4):
        gap, n_i, n_q = nodes[node]
        if n_i + n_q == 0:
            break
        if node > 0:
            p = powers[gap]
            w0, w1, w2 = (
                w0 * p[0, 0] + w1 * p[1, 0] + w2 * p[2, 0],
                w0 * p[0, 1] + w1 * p[1, 1] + w2 * p[2, 1],
                w0 * p[0, 2] + w1 * p[1, 2] + w2 * p[2, 2],
            )
        w0 *= emissions[n_i, n_q, 0]
        w1 *= emissions[n_i, n_q, 1]
        w2 *= emissions[n_i, n_q, 2]
    return w0 + w1 + w2


@njit(cache=True)
def feature_endpoints(layout: Layout, model: Model) -> tuple[FloatArray, FloatArray]:
    """Left state weights and right conditional means for separated products."""
    times, components, degrees = layout
    powers, _, emissions, population = model
    left, right = np.empty((113, 3)), np.empty((113, 3))
    for column in range(113):
        span = times[column].max()
        degree = np.zeros((2, 2), dtype=np.int64)
        for j in range(degrees[column]):
            end = 0 if times[column, j] == 0 else 1
            degree[end, components[column, j]] += 1
        first = emissions[degree[0, 0], degree[0, 1]]
        last = emissions[degree[1, 0], degree[1, 1]]
        left[column] = ((population * first) @ powers[span]) * last
        right[column] = first * (powers[span] @ last)
    return left, right


@njit(cache=True)
def separated_propagators(counts: IntArray, model: Model) -> FloatArray:
    """Sum n_ab(h)(R^(h-span_a)-Π) for every h > span_a, by residue."""
    _, tail, _, _ = model
    result = np.zeros((16, 16, 3, 3))
    for span_a in range(16):
        for span_b in range(16):
            for gap in range(1, 33):
                count = counts[span_a, span_b, (span_a + gap) % 32]
                result[span_a, span_b] += count * tail[gap]
    return result


@njit(cache=True)
def contiguous_covariance_kernel(
    layout: Layout, overlaps: OverlapLayout, model: Model
) -> FloatArray:
    """Finite overlapping products plus contracted, centred geometric tails."""
    times, _, _ = layout
    pairs, nodes, counts = overlaps
    spans = np.array([times[i].max() for i in range(113)])
    left, right = feature_endpoints(layout, model)
    means = left.sum(axis=1)
    result = np.zeros((113, 113))
    for row in range(len(pairs)):
        a, b, count = pairs[row]
        result[a, b] += count * (
            overlap_expectation(nodes[row], model) - means[a] * means[b]
        )
    propagators = separated_propagators(counts, model)
    for a in range(113):
        for b in range(a, 113):
            total = result[a, b]
            for i in range(3):
                for j in range(3):
                    total += (
                        left[a, i] * propagators[spans[a], spans[b], i, j] * right[b, j]
                        + left[b, i]
                        * propagators[spans[b], spans[a], i, j]
                        * right[a, j]
                    )
            result[a, b] = total / ((32 - spans[a]) * (32 - spans[b]))
            result[b, a] = result[a, b]
    return result


def contiguous_covariance(theta: FloatArray) -> FloatArray:
    """Exact asymptotic B*Cov(feature means), including inter-block dependence.

    Evaluate overlapping time supports directly using a cached layout. For
    disjoint supports, contract their endpoint weights with the anchor-weighted
    Markov resolvent. All positive lags beyond the left span enter that tail.
    The sum ignores stream-end edges and is exact up to floating-point error.
    """
    return contiguous_covariance_kernel(
        LAYOUT, covariance_layout(), model_arrays(theta)
    )
