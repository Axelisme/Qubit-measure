"""Optional numba kernels for SimEngine population-chain recurrence."""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(cache=True)
def _population_chain_serial_kernel(
    pre_props: NDArray[np.float64],
    relax_props: NDArray[np.float64],
    weights: NDArray[np.float64],
    equilibrium_pop: float,
    readout_q_post: float,
    reps: int,
    nreads: int,
) -> NDArray[np.float64]:
    node_count = pre_props.shape[0]
    states = np.empty((node_count, 4), dtype=np.float64)
    z0 = 2.0 * equilibrium_pop - 1.0
    for node_idx in range(node_count):
        states[node_idx, 0] = 0.0
        states[node_idx, 1] = 0.0
        states[node_idx, 2] = z0
        states[node_idx, 3] = 1.0

    sqrt_q_post = np.sqrt(readout_q_post)
    p_e = np.empty((reps, nreads), dtype=np.float64)
    for rep_idx in range(reps):
        p_mean = 0.0
        for node_idx in range(node_count):
            s0 = states[node_idx, 0]
            s1 = states[node_idx, 1]
            s2 = states[node_idx, 2]
            s3 = states[node_idx, 3]

            pre = pre_props[node_idx]
            r0 = pre[0, 0] * s0 + pre[0, 1] * s1 + pre[0, 2] * s2 + pre[0, 3] * s3
            r1 = pre[1, 0] * s0 + pre[1, 1] * s1 + pre[1, 2] * s2 + pre[1, 3] * s3
            r2 = pre[2, 0] * s0 + pre[2, 1] * s1 + pre[2, 2] * s2 + pre[2, 3] * s3
            r3 = pre[3, 0] * s0 + pre[3, 1] * s1 + pre[3, 2] * s2 + pre[3, 3] * s3

            node_p = 0.5 * (1.0 + r2)
            if node_p < 0.0:
                node_p = 0.0
            elif node_p > 1.0:
                node_p = 1.0
            p_mean += weights[node_idx] * node_p

            d0 = sqrt_q_post * r0
            d1 = sqrt_q_post * r1
            d2 = readout_q_post * r2 + (readout_q_post - 1.0) * r3
            d3 = r3

            relax = relax_props[node_idx]
            states[node_idx, 0] = (
                relax[0, 0] * d0
                + relax[0, 1] * d1
                + relax[0, 2] * d2
                + relax[0, 3] * d3
            )
            states[node_idx, 1] = (
                relax[1, 0] * d0
                + relax[1, 1] * d1
                + relax[1, 2] * d2
                + relax[1, 3] * d3
            )
            states[node_idx, 2] = (
                relax[2, 0] * d0
                + relax[2, 1] * d1
                + relax[2, 2] * d2
                + relax[2, 3] * d3
            )
            states[node_idx, 3] = (
                relax[3, 0] * d0
                + relax[3, 1] * d1
                + relax[3, 2] * d2
                + relax[3, 3] * d3
            )

        for read_idx in range(nreads):
            p_e[rep_idx, read_idx] = p_mean

    return p_e


def population_chain_numba(
    pre_props: NDArray[np.float64],
    relax_props: NDArray[np.float64],
    weights: NDArray[np.float64],
    equilibrium_pop: float,
    readout_q_post: float,
    reps: int,
    nreads: int,
) -> NDArray[np.float64]:
    """Run the JIT population-chain recurrence for a multi-node detune ensemble."""

    pre = np.ascontiguousarray(pre_props, dtype=np.float64)
    relax = np.ascontiguousarray(relax_props, dtype=np.float64)
    node_weights = np.ascontiguousarray(weights, dtype=np.float64)
    return _population_chain_serial_kernel(
        pre, relax, node_weights, equilibrium_pop, readout_q_post, reps, nreads
    )
