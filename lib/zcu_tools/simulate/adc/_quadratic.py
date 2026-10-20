"""Singular-safe continuous integrals of coherent constant-line products."""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from .model import RealArray

ComplexArray = NDArray[np.complex128]


def field_product_integral(
    generators: ComplexArray,
    coupling: RealArray,
    fields: ComplexArray,
    incidents: ComplexArray,
    omegas: RealArray,
    duration: float,
) -> ComplexArray:
    """Integrate alpha_s,r conj(alpha_u,q) with constant carrier forcing.

    generators are the fixed-branch (state,2,2) homogeneous line blocks, 1/ns.
    fields are initial (state,2) sqrt(photons); incidents are carrier amplitudes
    at the interval start, sqrt(photons/ns), with angular detunings omegas rad/ns.
    An augmented exponential integrates the Kronecker-sum product generator.
    No inverse, eigenvector basis, spectral separation or nonzero loss is needed.
    """
    n = len(fields)
    d = 2 + len(incidents)
    blocks = np.zeros((n, d, d), dtype=np.complex128)
    blocks[:, :2, :2] = generators
    blocks[:, :2, 2:] = coupling[None, :, None]
    blocks[:, 2:, 2:] = np.diag(-1j * omegas)
    initial = np.concatenate((fields, np.tile(incidents, (n, 1))), axis=1)
    products = np.empty((n, 2, n, 2), dtype=np.complex128)
    identity = np.eye(d)
    for s in range(n):
        for u in range(s, n):
            generator = np.kron(blocks[s], identity) + np.kron(
                identity, blocks[u].conj()
            )
            # phi1(K*T)*T*z0 via exp([[K,z0],[0,0]]*T). The extra zero
            # eigenvalue also covers resonant, dark and defective line blocks.
            augmented = np.zeros((d * d + 1, d * d + 1), dtype=np.complex128)
            augmented[:-1, :-1] = generator
            augmented[:-1, -1] = np.outer(initial[s], initial[u].conj()).ravel()
            integral = expm(augmented * duration)[:-1, -1].reshape(d, d)
            products[s, :, u, :] = integral[:2, :2]
            if s != u:
                products[u, :, s, :] = integral[:2, :2].conj().T
    if not np.isfinite(products).all():
        raise RuntimeError("constant cavity product integral is nonfinite")
    return products
