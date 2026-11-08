"""Batched matrix-chain moments and adjoint sensitivities for a K-reset cycle.

Each feature mean is a product of thermal populations, drive matrices and
emission-moment matrices along the 16 reads; the covariance sums the same
products over pairs of features (event products up to fourth order). The
Jacobian uses one forward and one backward pass over each product.
Precomputed propagators skip unobserved links in fourth-order products. Scalar
three-state matvecs avoid temporary arrays in the covariance's innermost loop.
Kernels are serial; callers control parallelism.
"""

import numpy as np
from numba import njit

from ._cycle_emission import cached_emissions, simplex, simplex_gradient
from ._cycle_features import tables


@njit(cache=True, inline="always")
def _multiply(
    a: float, b: float, c: float, matrix: np.ndarray
) -> tuple[float, float, float]:
    return (
        a * matrix[0, 0] + b * matrix[1, 0] + c * matrix[2, 0],
        a * matrix[0, 1] + b * matrix[1, 1] + c * matrix[2, 1],
        a * matrix[0, 2] + b * matrix[1, 2] + c * matrix[2, 2],
    )


@njit(cache=True)
def _adjoint_kernel(
    chain: tuple[np.ndarray, np.ndarray, np.ndarray], table: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drive, pi, matrix = chain
    nx = len(drive)
    means = np.zeros((nx, 65))
    ga = np.zeros((nx, 65, 3, 3))
    gp = np.zeros((nx, 65, 3))
    gm = np.zeros((nx, 65, 15, 3, 3))
    forward = np.empty((25, 3))
    kinds = np.empty(24, dtype=np.int64)
    for x in range(nx):
        for feature in table:
            group, events = feature[0], feature[1]
            last = feature[2 + events - 1]
            forward[0] = pi[x]
            length = 0
            for read in range(last + 1):
                if read % 2 == 0:
                    kinds[length] = -1
                    a, b, c = forward[length]
                    forward[length + 1] = _multiply(a, b, c, drive[x])
                    length += 1
                order = 0
                for event in range(events):
                    if feature[2 + event] == read:
                        order = feature[6 + event]
                kinds[length] = order
                a, b, c = forward[length]
                forward[length + 1] = _multiply(a, b, c, matrix[x, order])
                length += 1
            means[x, group] += forward[length].sum()
            right = np.ones(3)
            for link in range(length - 1, -1, -1):
                kind = kinds[link]
                for i in range(3):
                    for j in range(3):
                        contribution = forward[link, i] * right[j]
                        if kind == -1:
                            ga[x, group, i, j] += contribution
                        else:
                            gm[x, group, kind, i, j] += contribution
                transition = drive[x] if kind == -1 else matrix[x, kind]
                # Reverse multiplication is matrix @ right, not right @ matrix.
                r0 = (
                    transition[0, 0] * right[0]
                    + transition[0, 1] * right[1]
                    + transition[0, 2] * right[2]
                )
                r1 = (
                    transition[1, 0] * right[0]
                    + transition[1, 1] * right[1]
                    + transition[1, 2] * right[2]
                )
                r2 = (
                    transition[2, 0] * right[0]
                    + transition[2, 1] * right[1]
                    + transition[2, 2] * right[2]
                )
                right[0], right[1], right[2] = r0, r1, r2
            gp[x, group] += right
    return means, ga, gp, gm


def mean_jacobian(drive: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Means and derivatives w.r.t. D's first two entries per row, then 14 nuisances."""
    matrix, dm = emission_batch(rows)
    pi = np.array([simplex(*r[7:9]) for r in rows])
    means, ga, gp, gm = _adjoint_kernel((drive, pi, matrix), tables()[0])
    jac = np.empty((len(rows), 65, 20))
    jac[:, :, :6] = (ga[:, :, :, :2] - ga[:, :, :, 2:3]).reshape(len(rows), 65, 6)
    jac[:, :, 6:] = np.einsum("nfoij,noijk->nfk", gm, dm)
    dpi = np.array([simplex_gradient(*r[7:9]) for r in rows])
    jac[:, :, 13:15] += gp @ dpi
    return means, jac


def emission_batch(
    rows: np.ndarray, max_order: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    # Thermal populations do not change emissions; shared rows reuse one evaluation.
    key = rows.copy()
    key[:, 7:9] = 0
    _, indices, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    values = [cached_emissions(tuple(key[i]), max_order) for i in indices]
    return np.array([v[0] for v in values])[inverse], np.array([v[1] for v in values])[
        inverse
    ]


@njit(cache=True)
def _propagators(
    drive: np.ndarray, pi: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    nx = len(drive)
    pre = np.zeros((nx, 16, 3))
    between = np.zeros((nx, 16, 16, 3, 3))
    for x in range(nx):
        state = pi[x].copy()
        for read in range(16):
            if read % 2 == 0:
                state = state @ drive[x]
            pre[x, read] = state
            state = state @ matrix[x, 0]
            transition = np.eye(3)
            for other in range(read + 1, 16):
                if other % 2 == 0:
                    transition = transition @ drive[x]
                between[x, read, other] = transition
                transition = transition @ matrix[x, 0]
    return pre, between


@njit(cache=True)
def _second_kernel(
    chain: tuple[np.ndarray, np.ndarray, np.ndarray], table: np.ndarray
) -> np.ndarray:
    pre, between, matrix = chain
    result = np.zeros((len(pre), 65, 65))
    for x in range(len(pre)):
        for feature in table:
            g, h, factor, events = feature[0], feature[1], feature[2], feature[3]
            read = feature[4]
            a, b, c = pre[x, read]
            a, b, c = _multiply(a, b, c, matrix[x, feature[8]])
            for event in range(1, events):
                other = feature[4 + event]
                a, b, c = _multiply(a, b, c, between[x, read, other])
                a, b, c = _multiply(a, b, c, matrix[x, feature[8 + event]])
                read = other
            value = factor * (a + b + c)
            result[x, g, h] += value
            if g != h:
                result[x, h, g] += value
    return result


def covariance(drive: np.ndarray, rows: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Per-cycle covariance of the stored group sums from fourth-order event products."""
    matrix = emission_batch(rows, max_order=4)[0]
    pi = np.array([simplex(*r[7:9]) for r in rows])
    pre, between = _propagators(drive, pi, matrix)
    second = _second_kernel((pre, between, matrix), tables()[1])
    return second - means[:, :, None] * means[:, None, :]
