from typing import Tuple

import numpy as np

from zcu_tools.simulate.fluxonium.branch.floquet import calc_branch_infos


def calc_snr(
    params, r_f, g, flx, qub_dim, qub_cutoff, max_photon, rf_w
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    branchs = [0, 1]

    amps = np.arange(0.0, 2 * g * np.sqrt(max_photon), rf_w)
    photons = (amps / (2 * g)) ** 2

    branch_infos, fbasis_n = calc_branch_infos(
        branchs,
        params=params,
        r_f=r_f,
        g=g,
        flx=flx,
        qub_dim=qub_dim,
        qub_cutoff=qub_cutoff,
        photons=photons,
    )

    branch_energies = {
        b: np.array(
            [fbasis.e_quasi[branch_infos[b][n]] for n, fbasis in enumerate(fbasis_n)]
        )
        for b in branchs
    }

    f01_over_n = branch_energies[1] - branch_energies[0]
    chi_over_n = (f01_over_n[1:] - f01_over_n[:-1]) / (photons[1:] - photons[:-1])

    def signal_diff(x):
        return 1 - np.exp(-(x**2) / (2 * rf_w**2))

    snrs = np.abs(signal_diff(chi_over_n) * np.sqrt(photons[:-1]))

    return photons[:-1], chi_over_n, snrs
