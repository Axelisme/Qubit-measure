import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from scipy.optimize import bisect


@njit(parallel=True)
def calculate_density(
    Is: np.ndarray, Qs: np.ndarray, Ibins: np.ndarray, Qbins: np.ndarray, sigma: float
) -> np.ndarray:
    result = np.zeros((len(Ibins), len(Qbins)))
    for i in prange(len(Is)):
        x, y = Is[i], Qs[i]
        for j in range(len(Ibins)):
            for k in range(len(Qbins)):
                result[j, k] += (1 / (2 * np.pi * sigma**2)) * np.exp(
                    -((Ibins[j] - x) ** 2 + (Qbins[k] - y) ** 2) / (2 * sigma**2)
                )
    result /= np.sum(result)  # normalize
    return result


def calculate_ge_density(Ige, Qge, numbins=40):
    Ig, Ie = Ige
    Qg, Qe = Qge

    sigma_I = (max(Ig) - min(Ig)) / numbins
    sigma_Q = (max(Qg) - min(Qg)) / numbins
    sigma = (sigma_I + sigma_Q) / 2

    Ibins = np.linspace(np.min(Ige), np.max(Ige), numbins)
    Qbins = np.linspace(np.min(Qge), np.max(Qge), numbins)

    # Calculate the density of the points
    Hg = calculate_density(Ig, Qg, Ibins, Qbins, sigma)
    He = calculate_density(Ie, Qe, Ibins, Qbins, sigma)

    return Hg, He, Ibins, Qbins


def find_root(Ds, As, Bs):
    Ks = As - Bs

    def f(p):
        result = np.sum(Ds * (Ks / (p * Ks + Bs)))
        result = np.clip(result, -10, 10)
        return result

    # fig, ax = plt.subplots()
    # ps = np.linspace(0.001, 0.999, 1000)
    # ax.plot(ps, [f(p) for p in ps])
    # ax.axhline(0, color="red", linestyle="--")
    # ax.set_ylim(-2, 2)
    # plt.show()

    if f(0.001) * f(0.999) > 0:
        return 0.5

    root = bisect(f, 0.001, 0.999, maxiter=100, xtol=1e-6)
    return root


def calculate_p(Is, Qs, Hg, He, Ibins, Qbins):
    sigma_I = (max(Ibins) - min(Ibins)) / len(Ibins)
    sigma_Q = (max(Qbins) - min(Qbins)) / len(Qbins)
    sigma = (sigma_I + sigma_Q) / 2

    Ds = calculate_density(Is, Qs, Ibins, Qbins, sigma)

    return find_root(Ds, Hg, He)


def main():
    # Original code
    N = 1000000

    disper = (0.5, 0.0)
    numbins = 30

    def generate_random_points(num_points, x_c, y_c, std_dev):
        Is = np.random.normal(x_c, std_dev, num_points)
        Qs = np.random.normal(y_c, std_dev, num_points)

        return Is, Qs

    Ig, Qg = generate_random_points(N, 0, 0, 1)
    Ie, Qe = generate_random_points(N, disper[0], disper[1], 1)
    Hg, He, Ibins, Qbins = calculate_ge_density((Ig, Ie), (Qg, Qe), numbins=numbins)

    sample_N = 1000

    t_ps = np.linspace(0.001, 0.999, 100)
    g_ps = []
    xs = []
    for p in t_ps:
        sIg, sQg = generate_random_points(int(p * sample_N), 0, 0, 1)
        sIe, sQe = generate_random_points(
            int((1 - p) * sample_N), disper[0], disper[1], 1
        )

        sIs = np.concatenate((sIg, sIe))
        sQs = np.concatenate((sQg, sQe))
        pg = calculate_p(sIs, sQs, Hg, He, Ibins, Qbins)
        g_ps.append(pg)
        xs.append((np.mean(sIs), np.mean(sQs)))
    g_ps = np.array(g_ps)
    xs = np.array(xs)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(xs[:, 0], color="black")
    axs[1].plot(t_ps, g_ps)
    axs[1].plot(t_ps, t_ps, linestyle="--", color="red")
    plt.show()


if __name__ == "__main__":
    main()
