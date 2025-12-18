from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.storage import SpectrumData


def calculate_eff_t1_with(
    flx: float,
    noise_channels: list,
    Temp: float,
    fluxonium: Fluxonium,
    esys: Optional[Tuple[NDArray[np.float64], NDArray[np.complex128]]] = None,
    **other_noise_options,
) -> float:
    import scqubits.settings as scq_settings

    old, scq_settings.T1_DEFAULT_WARNING = scq_settings.T1_DEFAULT_WARNING, False

    fluxonium.flux = flx
    t1s = fluxonium.t1_effective(
        noise_channels=noise_channels,
        common_noise_options=dict(i=1, j=0, T=Temp, **other_noise_options),
        esys=esys,
    )

    scq_settings.T1_DEFAULT_WARNING = old

    return 2 * np.pi * t1s  # convert units


def calculate_eff_t1(
    flx: float,
    noise_channels: list,
    Temp: float,
    params: Tuple[float, float, float],
    cutoff: int = 40,
    evals_count: int = 20,
    **other_noise_options,
) -> float:
    from scqubits.core.fluxonium import Fluxonium

    fluxonium = Fluxonium(*params, flux=flx, cutoff=cutoff, truncated_dim=evals_count)

    return calculate_eff_t1_with(
        flx, noise_channels, Temp, fluxonium, esys=None, **other_noise_options
    )


def calculate_eff_t1_vs_flx_with(
    flxs: NDArray[np.float64],
    noise_channels: list,
    Temp: float,
    fluxonium: Fluxonium,
    spectrum_data: Optional[SpectrumData] = None,
    **other_noise_options,
) -> NDArray[np.float64]:
    import scqubits.settings as scq_settings

    old, scq_settings.T1_DEFAULT_WARNING = scq_settings.T1_DEFAULT_WARNING, False

    eff_t1s = np.zeros_like(flxs, dtype=np.float64)
    for i, flx in enumerate(flxs):
        fluxonium.flux = flx

        esys = None
        if spectrum_data is not None:
            esys = (spectrum_data.energy_table[i, :], spectrum_data.state_table[i])

        eff_t1s[i] = fluxonium.t1_effective(
            noise_channels=noise_channels,
            common_noise_options=dict(i=1, j=0, T=Temp, **other_noise_options),
            esys=esys,
        )

    scq_settings.T1_DEFAULT_WARNING = old

    return 2 * np.pi * eff_t1s


def calculate_eff_t1_vs_flx(
    flxs: NDArray[np.float64],
    noise_channels: list,
    Temp: float,
    params: Tuple[float, float, float],
    cutoff: int = 40,
    evals_count: int = 20,
    **other_noise_options,
) -> NDArray[np.float64]:
    from scqubits.core.fluxonium import Fluxonium

    fluxonium = Fluxonium(*params, flux=0.0, cutoff=cutoff, truncated_dim=evals_count)
    spectrum_data = fluxonium.get_spectrum_vs_paramvals(
        "flux",
        flxs,
        evals_count=evals_count,
        subtract_ground=True,
        get_eigenstates=True,
    )
    return calculate_eff_t1_vs_flx_with(
        flxs, noise_channels, Temp, fluxonium, spectrum_data, **other_noise_options
    )
