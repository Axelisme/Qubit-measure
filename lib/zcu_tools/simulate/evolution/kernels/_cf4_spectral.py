"""Carrier-independent CF4 nodes and eigenbasis recurrence for coarse guards."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._cf4_grid import coarse_count, spectral_width


@dataclass(frozen=True)
class CF4Nodes:
    """Immutable spectral product for one signed, uniform CF4 phase grid.

    values contains the two node spectra per interval, in GHz. outgoing holds
    the first exponential's eigenvectors. overlaps maps the second eigenbasis
    to the first; transitions maps the preceding outgoing basis (lab basis at
    the first interval) to the second. No array depends on carrier magnitude.
    WorkPlan resources own these arrays, never resident period payloads.
    """

    values: NDArray[np.float64]
    outgoing: NDArray[np.complex128]
    overlaps: NDArray[np.complex128]
    transitions: NDArray[np.complex128]


@dataclass(frozen=True)
class CF4Grids:
    """Complete non-anchor inventory partitioned by coarse interval count.

    by_count pairs each positive interval count with its exact carrier consumers.
    This small WorkPlan fact avoids re-counting the full inventory per key.
    """

    by_count: tuple[tuple[int, frozenset[float]], ...]


def carrier_grids(
    energies: NDArray[np.float64],
    x: NDArray[np.complex128],
    y: NDArray[np.complex128],
    carriers: tuple[float, ...],
) -> CF4Grids:
    """Group signed carriers by the builder's exact coarse-grid arithmetic.

    energies is the model energy vector; x/y are gain-scaled quadratures.
    The caller must supply a same-sign, nonzero carrier inventory.
    """
    h0 = np.diag(energies - energies[0]).astype(complex)
    width = spectral_width(h0, x, y)
    grids: dict[int, set[float]] = {}
    for frequency in carriers:
        count = coarse_count(1.5 * (1 / abs(frequency)) * width)
        grids.setdefault(count, set()).add(frequency)
    return CF4Grids(
        tuple((count, frozenset(values)) for count, values in grids.items())
    )


def build_nodes(
    h0: NDArray[np.complex128],
    x: NDArray[np.complex128],
    y: NDArray[np.complex128],
    sign: float,
    count: int,
) -> CF4Nodes:
    """Build both Hermitian node spectra and adjacent basis overlaps once.

    h0/x/y are square Hermitian matrices in GHz, with gain already applied.
    sign is +1 or -1; count is the positive coarse interval count. Degenerate
    eigenbases are allowed: the paired overlaps cancel their arbitrary gauge.
    """
    a = (3 - 2 * np.sqrt(3)) / 12
    b = (3 + 2 * np.sqrt(3)) / 12
    first = 2 * np.pi * sign * (np.arange(count) + 0.5 - np.sqrt(3) / 6) / count
    last = 2 * np.pi * sign * (np.arange(count) + 0.5 + np.sqrt(3) / 6) / count
    h1 = h0 + np.cos(first)[:, None, None] * x + np.sin(first)[:, None, None] * y
    h2 = h0 + np.cos(last)[:, None, None] * x + np.sin(last)[:, None, None] * y
    values, vectors = np.linalg.eigh(
        np.stack((a * h1 + b * h2, b * h1 + a * h2), axis=1)
    )
    outgoing = np.ascontiguousarray(vectors[:, 0])
    overlaps = np.ascontiguousarray(
        vectors[:, 0].conj().swapaxes(-1, -2) @ vectors[:, 1]
    )
    previous = np.concatenate((np.eye(len(h0), dtype=complex)[None], outgoing[:-1]))
    transitions = np.ascontiguousarray(vectors[:, 1].conj().swapaxes(-1, -2) @ previous)
    return CF4Nodes(values, outgoing, overlaps, transitions)


@njit(nogil=True, cache=True)
def spectral_prefixes(
    values: NDArray[np.float64],
    outgoing: NDArray[np.complex128],
    overlaps: NDArray[np.complex128],
    transitions: NDArray[np.complex128],
    frequency: float,
) -> NDArray[np.complex128]:
    """Evaluate all coarse prefixes at a carrier using a shared node product.

    Arrays are the fields of CF4Nodes for this carrier's sign and grid. Returns
    prefixes in the lab basis, including the identity and full-period endpoint.
    """
    count = len(values)
    size = outgoing.shape[-1]
    prefixes = np.empty((count + 1, size, size), dtype=np.complex128)
    prefixes[0] = np.eye(size, dtype=np.complex128)
    step = 1 / abs(frequency) / count
    current = np.eye(size, dtype=np.complex128)
    for j in range(count):
        current = transitions[j] @ current
        current *= np.exp(-2j * np.pi * step * values[j, 1])[:, None]
        current = overlaps[j] @ current
        current *= np.exp(-2j * np.pi * step * values[j, 0])[:, None]
        prefixes[j + 1] = outgoing[j] @ current
    return prefixes
