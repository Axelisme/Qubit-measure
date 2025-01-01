import matplotlib.pyplot as plt
import numpy as np


def rotate(I_orig, Q_orig, theta):
    I_new = I_orig * np.cos(theta) - Q_orig * np.sin(theta)
    Q_new = I_orig * np.sin(theta) + Q_orig * np.cos(theta)
    return I_new, Q_new


def scatter_plot(Is, Qs, ax, title=None):
    Ig, Ie = Is
    Qg, Qe = Qs
    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)

    ax.scatter(Ig, Qg, label="g", color="b", marker=".", edgecolor="None", alpha=0.2)
    ax.scatter(Ie, Qe, label="e", color="r", marker=".", edgecolor="None", alpha=0.2)
    plt_params = dict(color="k", linestyle=":", marker="o", markersize=5)
    ax.plot(xg, yg, markerfacecolor="b", **plt_params)
    ax.plot(xe, ye, markerfacecolor="r", **plt_params)
    ax.set_xlabel("I [ADC levels]")
    ax.set_ylabel("Q [ADC levels]")
    ax.legend(loc="upper right")
    if title is not None:
        ax.set_title(title, fontsize=14)
    ax.axis("equal")


def hist(Ig, Ie, numbins=200, ax=None, title=None):
    I_tot = np.concatenate((Ie, Ig))
    xlims = [np.min(I_tot), np.max(I_tot)]

    bins = np.linspace(xlims[0], xlims[1], numbins)
    if ax is not None:
        plt_params = dict(bins=bins, range=xlims, alpha=0.5)
        ng, *_ = ax.hist(Ig, color="b", label="g", **plt_params)
        ne, *_ = ax.hist(Ie, color="r", label="e", **plt_params)
        ax.set_ylabel("Counts", fontsize=14)
        ax.set_xlabel("I [ADC levels]", fontsize=14)
        ax.legend(loc="upper right")
        if title is not None:
            ax.set_title(title, fontsize=14)
    else:
        ng, *_ = np.histogram(Ig, bins=bins, range=xlims)
        ne, *_ = np.histogram(Ie, bins=bins, range=xlims)

    return ng, ne, bins


def cumulate_plot(ng, ne, bins, ax, title=None):
    ax.plot(bins[:-1], np.cumsum(ng), "b", label="g")
    ax.plot(bins[:-1], np.cumsum(ne), "r", label="e")
    if title is not None:
        ax.set_title(title, fontsize=14)
    ax.legend()
    ax.set_xlabel("I [ADC levels]", fontsize=14)


def fidelity_func(tp, tn, fp, fn):
    # this method calculates fidelity as (Ngg+Nee)/N = Ngg/N + Nee/N=(0.5N-Nge)/N + (0.5N-Neg)/N = 1-(Nge+Neg)/N
    return (tp + fn) / (tp + tn + fp + fn)


def calculate_fidelity(ng, ne, bins, threshold=None):
    cum_ng, cum_ne = np.cumsum(ng), np.cumsum(ne)

    if threshold is not None:
        tind = np.searchsorted(bins, threshold)
    else:
        contrast = np.abs(2 * (cum_ng - cum_ne) / (ng.sum() + ne.sum()))
        tind = contrast.argmax()

    # fid = 0.5 * (1 - ng[tind:].sum() / ng.sum() + 1 - ne[:tind].sum() / ne.sum())
    tp, fp = ng[tind:].sum(), ne[tind:].sum()
    tn, fn = ng[:tind].sum(), ne[:tind].sum()
    fid = fidelity_func(tp, tn, fp, fn)

    return fid, bins[tind]


def fitting_and_plot(Is, Qs, classify_func, plot=True, numbins=200):
    Ig, Ie = Is
    Qg, Qe = Qs

    if plot:
        _, axs = plt.subplots(2, 2, figsize=(8, 8))
        scatter_plot((Ig, Ie), (Qg, Qe), axs[0, 0], "Unrotated")

    # Calculate the angle of rotation
    out_dict = classify_func(Ig, Qg, Ie, Qe)
    theta = out_dict["theta"]

    # Rotate the IQ data
    Ig, Qg = rotate(Ig, Qg, theta)
    Ie, Qe = rotate(Ie, Qe, theta)

    if plot:
        scatter_plot((Ig, Ie), (Qg, Qe), axs[0, 1], "Rotated")
        ng, ne, bins = hist(Ig, Ie, numbins, axs[1, 0])
    else:
        ng, ne, bins = hist(Ig, Ie, numbins)

    fid, threshold = calculate_fidelity(ng, ne, bins, out_dict.get("threshold"))

    if plot:
        axs[0, 1].axvline(threshold, color="0.2", linestyle="--")

        title = "${F}_{ge}$"
        axs[1, 0].set_title(f"Histogram ({title}: {fid:.3%})", fontsize=14)
        axs[1, 0].axvline(threshold, color="0.2", linestyle="--")

        cumulate_plot(ng, ne, bins, axs[1, 1], "Cumulative Counts")
        axs[1, 1].axvline(threshold, color="0.2", linestyle="--")

        plt.subplots_adjust(hspace=0.25, wspace=0.15)
        plt.tight_layout()
        plt.show()

    return fid, threshold, theta * 180 / np.pi  # fids: ge, gf, ef
