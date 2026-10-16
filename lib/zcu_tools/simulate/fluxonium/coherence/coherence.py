from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from zcu_tools.simulate.fluxonium.scq_settings import scq_t1_default_warning

if TYPE_CHECKING:
    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.storage import SpectrumData


def calculate_eff_t1_with(
    flux: float,
    noise_channels: list[tuple[str, dict[str, Any]]],
    Temp: float,
    fluxonium: Fluxonium,
    esys: tuple[NDArray[np.float64], NDArray[np.complex128]] | None = None,
    **other_noise_options,
) -> float:
    """Return effective 1→0 T1 in ns, setting ``fluxonium.flux`` to ``flux``.

    ``Temp`` is K and ``flux`` is Φ/Φ₀. Channel tuples contain scqubits names and
    per-channel options. Optional ``esys`` reuses eigenpairs at this flux.
    Other options pass to scqubits; ``total`` defaults to True (both directions).
    """
    fluxonium.flux = flux
    with scq_t1_default_warning(enabled=False):
        # With energy/h in GHz, scqubits already returns physical ns (ADR-0071).
        return fluxonium.t1_effective(
            noise_channels=noise_channels,
            common_noise_options=dict(i=1, j=0, T=Temp, **other_noise_options),
            esys=esys,
        )


def calculate_eff_t1(
    flux: float,
    noise_channels: list[tuple[str, dict[str, Any]]],
    Temp: float,
    params: tuple[float, float, float],
    cutoff: int = 40,
    evals_count: int = 20,
    **other_noise_options,
) -> float:
    """Return effective 1→0 T1 in ns for ``params=(EJ, EC, EL)`` in GHz.

    ``flux`` is Φ/Φ₀, ``Temp`` is K, and ``cutoff``/``evals_count`` select the
    oscillator basis/retained levels. Channel tuples and additional options
    use scqubits conventions; ``total`` defaults to True (both directions).
    """
    from scqubits.core.fluxonium import Fluxonium

    fluxonium = Fluxonium(*params, flux=flux, cutoff=cutoff, truncated_dim=evals_count)

    return calculate_eff_t1_with(
        flux, noise_channels, Temp, fluxonium, esys=None, **other_noise_options
    )


def calculate_eff_t1_vs_flux_with(
    fluxs: NDArray[np.float64],
    noise_channels: list[tuple[str, dict[str, Any]]],
    Temp: float,
    fluxonium: Fluxonium,
    spectrum_data: SpectrumData | None = None,
    **other_noise_options,
) -> NDArray[np.float64]:
    """Return effective 1→0 T1 in ns for a one-dimensional flux grid.

    ``fluxs`` must be a 1D array of shape ``(N,)`` in Φ/Φ₀. The output has shape
    ``(N,)`` and preserves the input flux order. ``Temp`` is K.
    Channel tuples and additional options use scqubits conventions;
    ``total`` defaults to True (both directions).
    Optional ``spectrum_data`` must match this flux grid. This call sets
    ``fluxonium.flux`` to each successive flux, ending at the last point.
    """
    start_t = time.time()
    pbar = None

    eff_t1s = np.zeros_like(fluxs, dtype=np.float64)
    with scq_t1_default_warning(enabled=False):
        for i, flux in enumerate(fluxs):
            fluxonium.flux = flux

            esys = None
            if spectrum_data is not None:
                esys = (spectrum_data.energy_table[i, :], spectrum_data.state_table[i])

            eff_t1s[i] = fluxonium.t1_effective(
                noise_channels=noise_channels,
                common_noise_options=dict(i=1, j=0, T=Temp, **other_noise_options),
                esys=esys,
            )

            if time.time() - start_t > 1:
                if pbar is None:
                    pbar = tqdm(total=len(fluxs), desc="Calculating t1", leave=False)
                    pbar.update(i)
                pbar.update()
    if pbar is not None:
        pbar.close()

    return eff_t1s


def calculate_eff_t1_vs_flux(
    fluxs: NDArray[np.float64],
    noise_channels: list[tuple[str, dict[str, Any]]],
    Temp: float,
    params: tuple[float, float, float],
    cutoff: int = 40,
    evals_count: int = 20,
    **other_options,
) -> NDArray[np.float64]:
    """Return effective 1→0 T1 in ns for a one-dimensional flux grid.

    ``fluxs`` must be a 1D array of shape ``(N,)`` in Φ/Φ₀. The output has shape
    ``(N,)`` and preserves the input flux order.
    ``params=(EJ, EC, EL)`` are GHz and ``Temp`` is K.
    ``cutoff``/``evals_count`` select the oscillator basis/retained levels.
    Channel tuples and additional options use scqubits conventions;
    ``total`` defaults to True (both directions).
    """
    from scqubits.core.fluxonium import Fluxonium

    fluxonium = Fluxonium(*params, flux=0.0, cutoff=cutoff, truncated_dim=evals_count)
    spectrum_data = fluxonium.get_spectrum_vs_paramvals(
        "flux",
        fluxs,
        evals_count=evals_count,
        subtract_ground=True,
        get_eigenstates=True,
    )
    return calculate_eff_t1_vs_flux_with(
        fluxs, noise_channels, Temp, fluxonium, spectrum_data, **other_options
    )
