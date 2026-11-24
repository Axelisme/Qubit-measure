"""Singular-safe continuous integrals of coherent constant-line products."""

import math

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from .model import RealArray

ComplexArray = NDArray[np.complex128]

_TAYLOR_ORDER = 14
_POWERS = np.arange(_TAYLOR_ORDER + 1, dtype=np.float64)
# Integral of x**(j + k) over the unit interval, for the Taylor product.
_PRODUCT_WEIGHTS = 1 / (1 + np.add.outer(_POWERS, _POWERS))


def field_product_integral(
    generators: ComplexArray,
    coupling: RealArray,
    fields: ComplexArray,
    incidents: ComplexArray,
    omegas: RealArray,
    duration: float,
) -> ComplexArray:
    """Integrate alpha_s,r conj(alpha_u,q) with constant carrier forcing.

    generators are the fixed-branch (state,cavity,cavity) homogeneous line blocks, 1/ns.
    fields are initial (state,cavity) sqrt(photons); incidents are carrier amplitudes
    at the interval start, sqrt(photons/ns), with angular detunings omegas rad/ns.
    Return (state,cavity,state,cavity) products in photon*ns over duration ns.
    Small line blocks share a scaled Taylor integral across all state pairs;
    other cases use augmented exponentials of the Kronecker-sum generators.
    Neither path needs inversion, spectral separation or nonzero loss.
    """
    n = len(fields)
    c = fields.shape[1]
    d = c + len(incidents)
    blocks = np.zeros((n, d, d), dtype=np.complex128)
    blocks[:, :c, :c] = generators
    blocks[:, :c, c:] = coupling[None, :, None]
    blocks[:, c:, c:] = np.diag(-1j * omegas)
    initial = np.concatenate((fields, np.tile(incidents, (n, 1))), axis=1)
    # Share work over at least six state pairs and bound the carrier workspace.
    if n >= 3 and d <= 4:
        gram = _scaled_product_integral(blocks, initial, duration)
        if gram is not None and np.isfinite(gram).all():
            return gram[:, :c, :, :c]
    products = _augmented_product_integral(blocks, initial, c, duration)
    if not np.isfinite(products).all():
        raise RuntimeError("constant cavity product integral is nonfinite")
    return products


def _scaled_product_integral(
    blocks: ComplexArray, initial: ComplexArray, duration: float
) -> ComplexArray | None:
    """Integrate all augmented fields over a short step, then double to duration."""
    bound = np.max(np.sum(abs(blocks), axis=-1)) * duration
    # Cap the amplification of short-step roundoff on undamped modes.
    if not np.isfinite(bound) or bound > 2**19:
        return None
    squarings = math.ceil(math.log2(2 * bound)) if bound > 0.5 else 0
    dt = duration / 2**squarings
    scaled = blocks * dt
    n, d = initial.shape
    term = np.broadcast_to(np.eye(d, dtype=np.complex128), blocks.shape)
    step = term.copy()
    coefficients = np.empty((n, d, _TAYLOR_ORDER + 1), dtype=np.complex128)
    coefficients[:, :, 0] = initial
    # ||scaled||_inf <= 1/2, so the order-14 exponential tail is < 4e-17.
    for order in range(1, _TAYLOR_ORDER + 1):
        term = (term @ scaled) / order
        step += term
        coefficients[:, :, order] = (term @ initial[:, :, None])[:, :, 0]
    series = coefficients.reshape(n * d, -1)
    gram = ((series @ _PRODUCT_WEIGHTS) @ series.conj().T) * dt
    pairs = gram.reshape(n, d, n, d).transpose(0, 2, 1, 3)
    # G(2t)_{s,u} = G(t)_{s,u} + E_s(t) G(t)_{s,u} E_u(t)^H.
    for _ in range(squarings):
        pairs += (step[:, None] @ pairs) @ step.conj().transpose(0, 2, 1)[None, :]
        step = step @ step
    return pairs.transpose(0, 2, 1, 3)


def _augmented_product_integral(
    blocks: ComplexArray, initial: ComplexArray, c: int, duration: float
) -> ComplexArray:
    n, d = initial.shape
    products = np.empty((n, c, n, c), dtype=np.complex128)
    identity = np.eye(d)
    left_blocks = [np.kron(block, identity) for block in blocks]
    right_blocks = [np.kron(identity, block.conj()) for block in blocks]
    for s in range(n):
        for u in range(s, n):
            generator = left_blocks[s] + right_blocks[u]
            # phi1(K*T)*T*z0 via exp([[K,z0],[0,0]]*T). The extra zero
            # eigenvalue also covers resonant, dark and defective line blocks.
            augmented = np.zeros((d * d + 1, d * d + 1), dtype=np.complex128)
            augmented[:-1, :-1] = generator
            augmented[:-1, -1] = np.outer(initial[s], initial[u].conj()).ravel()
            integral = expm(augmented * duration)[:-1, -1].reshape(d, d)
            products[s, :, u, :] = integral[:c, :c]
            if s != u:
                products[u, :, s, :] = integral[:c, :c].conj().T
    return products
