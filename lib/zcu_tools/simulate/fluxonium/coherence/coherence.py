from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    # otherwise, lazy import
    import scqubits as scq


def calculate_eff_t1_with(
    flx: float,
    noise_channels: list,
    Temp: float,
    fluxonium: "scq.Fluxonium",
    esys: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> float:
    import scqubits as scq  # lazy import

    scq.settings.T1_DEFAULT_WARNING = False
    fluxonium.flux = flx
    return fluxonium.t1_effective(
        noise_channels=noise_channels,
        common_noise_options=dict(i=1, j=0, T=Temp),
        esys=esys,
    )


def calculate_eff_t1(
    flx: float,
    noise_channels: list,
    Temp: float,
    params: Tuple[float, float, float],
    cutoff: int = 40,
    evals_count: int = 20,
) -> float:
    import scqubits as scq  # lazy import

    scq.settings.T1_DEFAULT_WARNING = False
    fluxonium = scq.Fluxonium(
        *params, flux=flx, cutoff=cutoff, truncated_dim=evals_count
    )
    return fluxonium.t1_effective(
        noise_channels=noise_channels, common_noise_options=dict(i=1, j=0, T=Temp)
    )


def calculate_eff_t1_vs_flx_with(
    flxs: np.ndarray,
    noise_channels: list,
    Temp: float,
    fluxonium: "scq.Fluxonium",
    spectrum_data: Optional["scq.SpectrumData"] = None,
) -> np.ndarray:
    import scqubits as scq  # lazy import

    scq.settings.T1_DEFAULT_WARNING = False
    return np.asarray(
        [
            fluxonium.set_and_return("flux", flx).t1_effective(
                noise_channels=noise_channels,
                common_noise_options=dict(i=1, j=0, T=Temp),
                esys=(spectrum_data.energy_table[i, :], spectrum_data.state_table[i]),
            )
            for i, flx in enumerate(flxs)
        ]
    )


def calculate_eff_t1_vs_flx(
    flxs: np.ndarray,
    noise_channels: list,
    Temp: float,
    params: Tuple[float, float, float],
    cutoff: int = 40,
    evals_count: int = 20,
) -> np.ndarray:
    from scqubits import Fluxonium  # lazy import

    fluxonium = Fluxonium(*params, flux=0.0, cutoff=cutoff, truncated_dim=evals_count)
    spectrum_data = fluxonium.get_spectrum_vs_paramvals(
        "flux",
        flxs,
        evals_count=evals_count,
        subtract_ground=True,
        get_eigenstates=True,
    )
    return calculate_eff_t1_vs_flx_with(
        flxs, noise_channels, Temp, fluxonium, spectrum_data
    )
