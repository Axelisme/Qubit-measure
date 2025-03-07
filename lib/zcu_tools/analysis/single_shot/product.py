import numpy as np
import matplotlib.pyplot as plt

from .base import fidelity_func


def calculate_orth_vec(Is, Qs, numbins=40):
    """View I/Q as a vector and find orthogonal vector to g-e separation"""
    Ig, Ie = Is
    Qg, Qe = Qs

    # Calculate the density of the points
    Ibins = np.linspace(min(Ig), max(Ig), numbins)
    Qbins = np.linspace(min(Qg), max(Qg), numbins)
    Hg, _, _ = np.histogram2d(Ig, Qg, bins=(Ibins, Qbins))
    He, _, _ = np.histogram2d(Ie, Qe, bins=(Ibins, Qbins))

    # calculate the orthonormal vector
    # define inner product as sum of products of densities
    # Kg/Ke are on the plane defined by Hg/He
    # the inner product of Kg/Ke with He/Hg is 0
    # the inner product of Kg/Ke with Hg/He is 1
    Kg = Hg - np.sum(Hg * He) / np.sum(He * He) * He
    Ke = He - np.sum(He * Hg) / np.sum(Hg * Hg) * Hg
    Kg /= np.sum(Kg * Hg)  # normalize
    Ke /= np.sum(Ke * He)  # normalize

    return Kg - Ke, Ibins, Qbins


def get_ge_prefer(Iv, Qv, base_vecs):
    K, Ibins, Qbins = base_vecs

    dI = Ibins[1] - Ibins[0]
    dQ = Qbins[1] - Qbins[0]
    I_idx = ((Iv - Ibins[0]) / dI).astype(int)
    Q_idx = ((Qv - Qbins[0]) / dQ).astype(int)

    # truncate the index to the valid range
    I_idx = np.clip(I_idx, 0, len(Ibins) - 2)
    Q_idx = np.clip(Q_idx, 0, len(Qbins) - 2)

    return K[I_idx, Q_idx]


def test_inner_product():
    np.random.seed(0)
    N = 10000

    _, axs = plt.subplots(1, 3, figsize=(12, 4))

    Ig = np.random.normal(0, 1, N)
    Qg = np.random.normal(0, 1, N)
    Ie = np.random.normal(0.3, 1, N)
    Qe = np.random.normal(0.3, 1, N)
    Is = (Ig, Ie)
    Qs = (Qg, Qe)
    base_vecs = calculate_orth_vec(Is, Qs)

    xlims = (Ig.min(), Ig.max())
    ylims = (Qg.min(), Qg.max())
    axs[0].scatter(Ig, Qg, c="b", s=1)
    axs[0].scatter(Ie, Qe, c="r", s=1)
    axs[0].set_xlim(xlims)
    axs[0].set_ylim(ylims)

    Pg = get_ge_prefer(Ig, Qg, base_vecs)
    Pe = get_ge_prefer(Ie, Qe, base_vecs)

    # scatter plot each point with color indicating preference
    axs[1].scatter(Ig, Qg, c=Pg, s=1)
    axs[2].scatter(Ie, Qe, c=Pe, s=1)
    axs[1].set_xlim(xlims)
    axs[1].set_ylim(ylims)
    axs[2].set_xlim(xlims)
    axs[2].set_ylim(ylims)

    # apply 0 as the threshold
    Pg, Pe = Pg > 0, Pe > 0
    tp, tn = np.sum(Pg), np.sum(~Pe)
    fp, fn = np.sum(Pe), np.sum(~Pe)
    fid = fidelity_func(tp, tn, fp, fn)
    print(fid)

    plt.show()


if __name__ == "__main__":
    test_inner_product()
