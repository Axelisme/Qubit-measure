from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit

from zcu_tools.notebook.util.fitting import dual_gauss_func, fit_dual_gauss


class GEAnchorFitter1D:
    def __init__(self, bins: np.ndarray) -> None:
        self.bins = bins

        self.anchor_g = None
        self.anchor_e = None
        self.sigma = None

    def fit(self, signals_g: np.ndarray, signals_e: np.ndarray) -> None:
        """
        Fit the anchor points for ground and excited states based on the provided signals.
        It assumesthat the population of ground state and excited are exactly inverse in signals_g and signals_e.
        """
        max_g = np.max(signals_g)
        min_e = np.min(signals_e)

        if self.anchor_g is not None:
            init_guess = (self.anchor_g, self.anchor_e, self.sigma, max_g, min_e)
        else:
            signals_all = signals_g + signals_e
            params_ge, _ = fit_dual_gauss(self.bins, signals_all)
            init_guess = (
                params_ge[1],  # x_c1
                params_ge[4],  # x_c2
                params_ge[2],  # sigma
                max_g,  # yscale1
                min_e,  # yscale2
            )

        mid_x = 0.5 * (self.bins.max() + self.bins.min())
        bounds = (
            [self.bins.min(), mid_x, 0.0, 0.0, 0.0],
            [
                mid_x,
                self.bins.max(),
                0.5 * (self.bins.max() - self.bins.min()),
                max(max_g, min_e),
                max(max_g, min_e),
            ],
        )

        # Fit the dual Gaussian model to the signals
        def conjugate_dual_gauss(_, c1, c2, sigma, p1, p2):
            sim_g = dual_gauss_func(self.bins, p1, c1, sigma, p2, c2, sigma)
            sim_e = dual_gauss_func(self.bins, p2, c2, sigma, p1, c1, sigma)

            return np.concatenate((sim_g, sim_e))

        popt, _ = curve_fit(
            conjugate_dual_gauss,
            np.concatenate((self.bins, self.bins)),
            np.concatenate((signals_g, signals_e)),
            p0=init_guess,
            bounds=bounds,
        )

        self.anchor_g = popt[0]
        self.anchor_e = popt[1]
        self.sigma = popt[2]

    def predict_population(
        self, bins: np.ndarray, signals: np.ndarray
    ) -> Tuple[float, float]:
        if self.anchor_g is None:
            raise ValueError("Model has not been fitted yet.")

        fixed_params = (
            None,
            self.anchor_g,
            self.sigma,
            None,
            self.anchor_e,
            self.sigma,
        )

        popt, _ = fit_dual_gauss(bins, signals, fixedparams=fixed_params)

        return popt[0], popt[3]


if __name__ == "__main__":
    # Example usage
    signals_G = np.random.normal(-1.0, 1, 100000)
    signals_E = np.random.normal(1.0, 1, 100000)

    g_num = 70000
    signals_g = np.concatenate((signals_G[:g_num], signals_E[g_num:]))
    signals_e = np.concatenate((signals_E[:g_num], signals_G[g_num:]))

    signals_g, bins = np.histogram(signals_g, bins=50)
    signals_e, _ = np.histogram(signals_e, bins=50)

    bins = bins[:-1]

    fitter = GEAnchorFitter1D(bins)
    fitter.fit(signals_g, signals_e)

    print(
        f"Anchor G: {fitter.anchor_g}, Anchor E: {fitter.anchor_e}, Sigma: {fitter.sigma}"
    )

    pop_gg, pop_ge = fitter.predict_population(bins, signals_g)
    pop_eg, pop_ee = fitter.predict_population(bins, signals_e)
    print(f"Predicted Population G: {pop_gg / (pop_gg + pop_ge)}")
    print(f"Predicted Population E: {pop_ee / (pop_eg + pop_ee)}")

    import matplotlib.pyplot as plt
    from zcu_tools.notebook.util.fitting import dual_gauss_func

    plt.plot(bins, signals_g, label="Signal G")
    plt.plot(bins, signals_e, label="Signal E")
    plt.plot(
        bins,
        dual_gauss_func(
            bins,
            pop_gg,
            fitter.anchor_g,
            fitter.sigma,
            pop_ge,
            fitter.anchor_e,
            fitter.sigma,
        ),
    )
    plt.plot(
        bins,
        dual_gauss_func(
            bins,
            pop_eg,
            fitter.anchor_g,
            fitter.sigma,
            pop_ee,
            fitter.anchor_e,
            fitter.sigma,
        ),
    )
    plt.axvline(fitter.anchor_g, color="r", linestyle="--", label="Anchor G")
    plt.axvline(fitter.anchor_e, color="g", linestyle="--", label="Anchor E")
    plt.xlabel("Bins")
    plt.ylabel("Signals")
    plt.legend()
    plt.title("GE Anchor Fitter 1D Example")
    plt.show()
