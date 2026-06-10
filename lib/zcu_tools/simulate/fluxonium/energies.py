from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

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
    spectrum_data: SpectrumData | None = None,
) -> tuple[SpectrumData, NDArray[np.float64]]:
    """Fluxonium energy levels vs external flux — ~100x faster than scqubits.

    scqubits' ``get_spectrum_vs_paramvals`` rebuilds the Hamiltonian at every
    flux point, and ``cos_phi_operator(beta)`` there calls ``scipy.linalg.cosm``
    (a matrix cosine) once per point — profiled at ~90% of the cost, while the
    40x40 diagonalisation is ~1%. Only the cosine's phase ``beta = 2*pi*flux``
    changes with flux, so this precomputes the flux-independent ``cos(phi)`` /
    ``sin(phi)`` operators (and the LC diagonal) ONCE, then combines them cheaply
    per point using ``cos(phi + beta) = cos(phi) cos(beta) - sin(phi) sin(beta)``
    (exact, since ``beta * I`` commutes with ``phi``).

    Pass a previously returned ``spectrum_data`` to skip the computation entirely
    and reuse its ``energy_table`` (handy when iterating interactively on a
    spectrum you've already computed).

    Returns ``(SpectrumData, energies)``: the ``SpectrumData`` mirrors the
    scqubits one (``energy_table`` / ``param_vals`` over the folded-unique flux
    grid; ``state_table`` is None, as when states are not stored), and
    ``energies`` is per-INPUT-flux (reordered back from the folded grid). Energies
    are absolute (in the LC oscillator basis), per the scqubits convention; they
    match a direct scqubits computation to ~1e-13 (see tests).
    """
    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.storage import SpectrumData

    if spectrum_data is not None:
        energies = np.asarray(spectrum_data.energy_table, dtype=np.float64)
        return spectrum_data, energies  # reuse a precomputed spectrum, skip work

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

    # Per folded-unique flux (the SpectrumData grid).
    unique_energies = np.empty((len(folded_fluxs), evals_count), dtype=np.float64)
    for k, flux in enumerate(folded_fluxs):
        beta = 2.0 * np.pi * flux
        cos_mat = cos_phi * np.cos(beta) - sin_phi * np.sin(beta)
        hamiltonian = np.diag(lc_diag) - EJ * cos_mat
        unique_energies[k] = np.linalg.eigvalsh(hamiltonian)[:evals_count]

    spectrum_data = SpectrumData(
        energy_table=unique_energies.copy(),
        system_params=fluxonium.get_initdata(),
        param_name="flux",
        param_vals=folded_fluxs,
    )

    # Map the per-unique energies back to the original input flux order.
    energies = np.empty_like(unique_energies)
    energies[sort_idxs, :] = unique_energies
    energies = energies[uni_idxs, :]

    return spectrum_data, energies
