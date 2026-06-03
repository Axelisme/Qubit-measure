from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from scqubits.core.storage import SpectrumData


def _fold_unique_fluxs(
    fluxs: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.intp]]:
    """Fold fluxs into [0, 0.5] and deduplicate (the spectrum is periodic+even).

    Returns the sorted unique folded fluxs plus the index arrays needed to map a
    per-unique-flux result back to the original flux order:
    ``(unique_sorted_fluxs, sort_idxs, uni_idxs)`` where the caller writes
    ``out[sort_idxs] = computed`` then ``out = out[uni_idxs]``.
    """
    folded = fluxs % 1.0
    folded = np.where(folded < 0.5, folded, 1.0 - folded)
    folded, uni_idxs = np.unique(folded, return_inverse=True)
    sort_idxs = np.argsort(folded)
    return folded[sort_idxs], sort_idxs, uni_idxs


def calculate_energy(
    params: tuple[float, float, float],
    flux: float,
    cutoff: int = 40,
    evals_count: int = 20,
) -> NDArray[np.float64]:
    """
    Calculate the energy of a fluxonium qubit.
    """

    from scqubits.core.fluxonium import Fluxonium  # lazy import

    fluxonium = Fluxonium(*params, flux=flux, cutoff=cutoff, truncated_dim=evals_count)
    return fluxonium.eigenvals(evals_count=evals_count)


def calculate_energy_vs_flux(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    cutoff: int = 40,
    evals_count: int = 20,
    spectrum_data: Optional[SpectrumData] = None,
) -> tuple[SpectrumData, NDArray[np.float64]]:
    """
    Calculate the energy of a fluxonium qubit.
    """

    from scqubits.core.fluxonium import Fluxonium

    if spectrum_data is not None:
        energies = np.asarray(spectrum_data.energy_table, dtype=np.float64)
        return spectrum_data, energies  # early return

    # because energy is periodic, remove repeated values and record index
    fluxs = fluxs % 1.0
    fluxs = np.where(fluxs < 0.5, fluxs, 1.0 - fluxs)

    fluxs, uni_idxs = np.unique(fluxs, return_inverse=True)
    sort_idxs = np.argsort(fluxs)
    fluxs = fluxs[sort_idxs]

    # calculate energy vs flux
    fluxonium = Fluxonium(*params, flux=0.0, cutoff=cutoff, truncated_dim=evals_count)
    spectrum_data = fluxonium.get_spectrum_vs_paramvals(
        "flux", fluxs, evals_count=evals_count
    )
    energies = np.asarray(spectrum_data.energy_table, dtype=np.float64)

    # rearrange energies to match the original order of fluxs
    energies[sort_idxs, :] = energies
    energies = energies[uni_idxs, :]

    return spectrum_data, energies


def calculate_energy_vs_flux_fast(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    cutoff: int = 40,
    evals_count: int = 20,
) -> tuple[None, NDArray[np.float64]]:
    """Fast drop-in for :func:`calculate_energy_vs_flux` (same energies, ~100x).

    The profiled cost of the scqubits path is rebuilding the Hamiltonian at every
    flux: ``cos_phi_operator(beta)`` calls ``scipy.linalg.cosm`` (a matrix cosine)
    once per flux point — ~90% of the time — while the 40x40 diagonalisation is
    ~1%. Only the cosine's phase ``beta = 2*pi*flux`` changes with flux, so this
    precomputes the flux-independent pieces once and combines them cheaply per
    point using ``cos(phi + beta) = cos(phi) cos(beta) - sin(phi) sin(beta)``
    (exact, since ``beta * I`` commutes with ``phi``).

    Returns ``(None, energies)`` — same shape/values as
    ``calculate_energy_vs_flux``'s second element (verified to ~1e-13 in tests),
    but no ``SpectrumData`` (no caller uses it). Energies are absolute (in the LC
    oscillator basis), matching the scqubits convention.
    """
    from scqubits.core.fluxonium import Fluxonium

    folded_fluxs, sort_idxs, uni_idxs = _fold_unique_fluxs(fluxs)

    fluxonium = Fluxonium(*params, flux=0.0, cutoff=cutoff, truncated_dim=evals_count)
    dim = fluxonium.hilbertdim()

    # Flux-independent pieces, computed ONCE (the expensive cosm/sinm live here).
    lc_diag = np.array(
        [(i + 0.5) * fluxonium.plasma_energy() for i in range(dim)], dtype=np.float64
    )
    cos_phi = np.asarray(fluxonium.cos_phi_operator(beta=0.0), dtype=np.float64)
    sin_phi = np.asarray(fluxonium.sin_phi_operator(beta=0.0), dtype=np.float64)
    EJ = float(fluxonium.EJ)

    energies = np.empty((len(folded_fluxs), evals_count), dtype=np.float64)
    for k, flux in enumerate(folded_fluxs):
        beta = 2.0 * np.pi * flux
        cos_mat = cos_phi * np.cos(beta) - sin_phi * np.sin(beta)
        hamiltonian = np.diag(lc_diag) - EJ * cos_mat
        energies[k] = np.linalg.eigvalsh(hamiltonian)[:evals_count]

    # map per-unique-flux energies back to the original flux order
    energies[sort_idxs, :] = energies
    energies = energies[uni_idxs, :]

    return None, energies
