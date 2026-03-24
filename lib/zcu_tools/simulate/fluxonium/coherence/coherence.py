from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
from typing_extensions import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.storage import SpectrumData


def calculate_eff_t1_with(
    flux: float,
    noise_channels: list[tuple[str, dict[str, Any]]],
    Temp: float,
    fluxonium: Fluxonium,
    esys: Optional[tuple[NDArray[np.float64], NDArray[np.complex128]]] = None,
    **other_noise_options,
) -> float:
    import scqubits.settings as scq_settings

    old, scq_settings.T1_DEFAULT_WARNING = scq_settings.T1_DEFAULT_WARNING, False

    fluxonium.flux = flux
    t1s = fluxonium.t1_effective(
        noise_channels=noise_channels,
        common_noise_options=dict(i=1, j=0, T=Temp, **other_noise_options),
        esys=esys,  # type: ignore
    )

    scq_settings.T1_DEFAULT_WARNING = old

    # scqubits returns t1 in units of ns/rad, so we need to convert to ns
    return 2 * np.pi * t1s


def calculate_eff_t1(
    flux: float,
    noise_channels: list[tuple[str, dict[str, Any]]],
    Temp: float,
    params: tuple[float, float, float],
    cutoff: int = 40,
    evals_count: int = 20,
    **other_noise_options,
) -> float:
    from scqubits.core.fluxonium import Fluxonium

    fluxonium = Fluxonium(*params, flux=flux, cutoff=cutoff, truncated_dim=evals_count)

    return calculate_eff_t1_with(
        flux, noise_channels, Temp, fluxonium, esys=None, **other_noise_options
    )


def calculate_eff_t1_vs_flx_with(
    fluxs: NDArray[np.float64],
    noise_channels: list[tuple[str, dict[str, Any]]],
    Temp: float,
    fluxonium: Fluxonium,
    spectrum_data: Optional[SpectrumData] = None,
    **other_noise_options,
) -> NDArray[np.float64]:
    import scqubits.settings as scq_settings

    old, scq_settings.T1_DEFAULT_WARNING = scq_settings.T1_DEFAULT_WARNING, False

    start_t = time.time()
    pbar = None

    eff_t1s = np.zeros_like(fluxs, dtype=np.float64)
    for i, flx in enumerate(fluxs):
        fluxonium.flux = flx

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

    scq_settings.T1_DEFAULT_WARNING = old

    return 2 * np.pi * eff_t1s


def calculate_eff_t1_vs_flx(
    fluxs: NDArray[np.float64],
    noise_channels: list[tuple[str, dict[str, Any]]],
    Temp: float,
    params: tuple[float, float, float],
    cutoff: int = 40,
    evals_count: int = 20,
    **other_options,
) -> NDArray[np.float64]:
    from scqubits.core.fluxonium import Fluxonium

    fluxonium = Fluxonium(*params, flux=0.0, cutoff=cutoff, truncated_dim=evals_count)
    spectrum_data = fluxonium.get_spectrum_vs_paramvals(
        "flux",
        fluxs,
        evals_count=evals_count,
        subtract_ground=True,
        get_eigenstates=True,
    )
    return calculate_eff_t1_vs_flx_with(
        fluxs, noise_channels, Temp, fluxonium, spectrum_data, **other_options
    )
