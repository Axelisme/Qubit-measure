import numpy as np
from scipy.optimize import minimize


def calculate_spectrum(flxs, EJ, EC, EL, evals_count=4, cutoff=50):
    from scqubits import Fluxonium

    fluxonium = Fluxonium(EJ, EC, EL, flux=0.0, cutoff=cutoff)
    spectrumData = fluxonium.get_spectrum_vs_paramvals(
        "flux", flxs, evals_count=evals_count
    )

    return spectrumData.energy_table


def fit_spectrum(flxs, fpts, EJb, ECb, ELb, maxlevel=3, maxfun=10):
    """
    Fit the fluxonium spectrum to the experimental spectrum
    flxs: 1D array of flux values, shape (n,)
    fpts: 2D array of transition frequencies, shape (n, 2)
    """

    def loss_func(params):
        EJ, EC, EL = params
        energies = calculate_spectrum(flxs, EJ, EC, EL, maxlevel)  # (n, m)

        fs = []
        for i in [0, 1]:
            for j in range(i + 1, len(energies[0])):
                fs.append(energies[:, j] - energies[:, i])
        fs = np.vstack(fs).T  # (n, m)

        # the loss the the squared difference between the experimental and clostest calculated frequencies
        loss_fs = (fpts[:, None, :] - fs[:, :, None]) ** 2  # (n, m, 2)
        loss_fs = np.min(loss_fs, axis=1)  # (n, 2)

        return np.nansum(loss_fs)

    # initial guess
    EJ = (EJb[0] + EJb[1]) / 2
    EC = (ECb[0] + ECb[1]) / 2
    EL = (ELb[0] + ELb[1]) / 2

    bounds = (EJb, ECb, ELb)
    res = minimize(
        loss_func,
        (EJ, EC, EL),
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxfun": maxfun},
    )

    return res.x

