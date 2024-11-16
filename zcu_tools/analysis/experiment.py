import matplotlib.pyplot as plt
import numpy as np

from . import fitting as ft

figsize = (8, 6)


def spectrum_analyze(x, y, asym=False):
    mag = np.abs(y)
    pha = np.unwrap(np.angle(y))
    if asym:
        pOpt_mag, _ = ft.fit_asym_lor(x, mag)
        pOpt_pha, _ = ft.fit_asym_lor(x, pha)
        curve_mag = ft.asym_lorfunc(x, *pOpt_mag)
        curve_pha = ft.asym_lorfunc(x, *pOpt_pha)
    else:
        pOpt_mag, _ = ft.fitlor(x, mag)
        pOpt_pha, _ = ft.fitlor(x, pha)
        curve_mag = ft.lorfunc(x, *pOpt_mag)
        curve_pha = ft.lorfunc(x, *pOpt_pha)
    res_mag, kappa_mag = pOpt_mag[3], 2 * pOpt_mag[4]
    res_pha, kappa_pha = pOpt_pha[3], 2 * pOpt_pha[4]

    fig, axs = plt.subplots(2, 1, figsize=figsize)
    axs[0].plot(x, mag, label="mag", marker="o", markersize=3)
    axs[0].plot(x, curve_mag, label=f"fit, $kappa$={kappa_mag:.2f}")
    axs[0].axvline(res_mag, color="r", ls="--", label=f"$f_res$ = {res_mag:.2f}")
    axs[0].set_title("mag.", fontsize=15)
    axs[0].legend()

    axs[1].plot(x, pha, label="pha", marker="o", markersize=3)
    axs[1].plot(x, curve_pha, label=f"fit, $kappa$={kappa_pha:.2f}")
    axs[1].axvline(res_pha, color="r", ls="--", label=f"$f_res$ = {res_pha:.2f}")
    axs[1].set_title("pha.", fontsize=15)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    return round(res_mag, 2), round(res_pha, 2)


def dispersive_analyze(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, asym=False):
    y1 = np.abs(y1)  # type: ignore
    y2 = np.abs(y2)  # type: ignore
    if asym:
        pOpt1, _ = ft.fit_asym_lor(x, y1)
        pOpt2, _ = ft.fit_asym_lor(x, y2)
        curve1 = ft.asym_lorfunc(x, *pOpt1)
        curve2 = ft.asym_lorfunc(x, *pOpt2)
    else:
        pOpt1, _ = ft.fitlor(x, y1)
        pOpt2, _ = ft.fitlor(x, y2)
        curve1 = ft.lorfunc(x, *pOpt1)
        curve2 = ft.lorfunc(x, *pOpt2)
    res1, kappa1 = pOpt1[3], 2 * pOpt1[4]
    res2, kappa2 = pOpt2[3], 2 * pOpt2[4]

    plt.figure(figsize=figsize)
    plt.title(f"$chi=${(res2-res1):.3f}, unit = MHz", fontsize=15)
    plt.plot(x, y1, label="e", marker="o", markersize=3)
    plt.plot(x, y2, label="g", marker="o", markersize=3)
    plt.plot(x, curve1, label=f"fite, $kappa$ = {kappa1:.2f}MHz")
    plt.plot(x, curve2, label=f"fitg, $kappa$ = {kappa2:.2f}MHz")
    plt.axvline(res1, color="r", ls="--", label=f"$f_res$ = {res1:.2f}")
    plt.axvline(res2, color="g", ls="--", label=f"$f_res$ = {res2:.2f}")
    plt.legend()

    plt.figure(figsize=figsize)
    plt.plot(x, y1 - y2)
    diff_curve = curve1 - curve2
    max_id = np.argmax(diff_curve)
    min_id = np.argmin(diff_curve)
    plt.plot(x, diff_curve)
    plt.axvline(
        x[np.argmax(diff_curve)],  # type: ignore
        color="r",
        ls="--",
        label=f"max SNR1 = {x[max_id]:.2f}",  # type: ignore
    )
    plt.axvline(
        x[np.argmin(curve1 - curve2)],  # type: ignore
        color="g",
        ls="--",
        label=f"max SNR2 = {x[max_id]:.2f}",
    )
    plt.legend()
    plt.show()

    if np.abs(diff_curve[max_id]) >= np.abs(diff_curve[min_id]):
        return x[max_id], x[min_id]
    else:
        return x[min_id], x[max_id]


def amprabi_analyze(x: int, y: float):
    y = np.abs(y)
    pOpt, _ = ft.fitsin(x, y)

    freq = pOpt[2]
    phase = pOpt[3] % 360 - 180
    if phase < 0:
        pi_gain = (0.25 - phase / 360) / freq
        pi2_gain = -phase / 360 / freq
    else:
        pi_gain = (0.75 - phase / 360) / freq
        pi2_gain = (0.5 - phase / 360) / freq

    plt.figure(figsize=figsize)
    plt.plot(x, y, label="meas", ls="-", marker="o", markersize=3)
    plt.plot(x, ft.sinfunc(x, *pOpt), label="fit")
    plt.title("Amplitude Rabi", fontsize=15)
    plt.xlabel("$gain$", fontsize=15)
    plt.axvline(pi_gain, ls="--", c="red", label=f"$\pi$ gain={pi_gain:.1f}")
    plt.axvline(pi2_gain, ls="--", c="red", label=f"$\pi/2$ gain={(pi2_gain):.1f}")
    plt.legend(loc=4)
    plt.tight_layout()
    plt.show()

    return pi_gain, pi2_gain, np.max(y) - np.min(y)


def T1_analyze(x: float, y: float):
    y = np.abs(y)
    pOpt, pCov = ft.fitexp(x, y)
    t1 = pOpt[2]
    sim = ft.expfunc(x, *pOpt)

    plt.figure(figsize=figsize)
    plt.plot(x, y, label="meas", ls="-", marker="o", markersize=3)
    plt.plot(x, sim, label="fit")
    plt.title(f"T1 = {t1:.2f}$\mu s$", fontsize=15)
    plt.xlabel("$t\ (\mu s)$", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return t1


def T2fringe_analyze(x: float, y: float):
    y = np.abs(y)
    pOpt, pCov = ft.fitdecaysin(x, y)
    decay, detune = pOpt[4], pOpt[2]
    sim = ft.decaysin(x, *pOpt)
    error = np.sqrt(np.diag(pCov))

    plt.figure(figsize=figsize)
    plt.plot(x, y, label="meas", ls="-", marker="o", markersize=3)
    plt.plot(x, sim, label="fit")
    plt.title(
        f"T2 fringe = {decay:.2f}$\mu s, detune = {detune:.2f}MHz \pm {(error[2])*1e3:.2f}kHz$",
        fontsize=15,
    )
    plt.xlabel("$t\ (\mu s)$", fontsize=15)
    plt.ylabel("Population", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return decay, detune


def T2decay_analyze(x: float, y: float):
    y = np.abs(y)
    pOpt, pCov = ft.fitexp(x, y)
    decay = pOpt[2]

    plt.figure(figsize=figsize)
    plt.plot(x, y, label="meas", ls="-", marker="o", markersize=3)
    plt.plot(x, ft.expfunc(x, *pOpt), label="fit")
    plt.title(f"T2 decay = {decay:.2f}$\mu s$", fontsize=15)
    plt.xlabel("$t\ (\mu s)$", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return decay


def singleshot_analysis(Is, Qs, plot=True, verbose=True, title=None):
    Ig, Ie = Is
    Qg, Qe = Qs

    numbins = 200

    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)

    def verbose_print(name, Is, Qs):
        Ig, Ie = Is
        Qg, Qe = Qs
        xg, yg = np.median(Ig), np.median(Qg)
        xe, ye = np.median(Ie), np.median(Qe)
        print(f"{name}:")
        print(
            f"Ig {xg:.2f} +/- {np.std(Ig):.2f} \t Qg {yg:.2f} +/- {np.std(Qg):.2f} \t Amp g {np.abs(xg+1j*yg):.2f} +/- {np.std(np.abs(Ig + 1j*Qg)):.2f}"
        )
        print(
            f"Ie {xe:.2f} +/- {np.std(Ie):.2f} \t Qe {ye:.2f} +/- {np.std(Qe):.2f} \t Amp e {np.abs(xe+1j*ye):.2f} +/- {np.std(np.abs(Ig + 1j*Qe)):.2f}"
        )

    if verbose:
        verbose_print("Unrotated", (Ig, Ie), (Qg, Qe))

    def scatter_plot(name, ax, Is, Qs):
        Ig, Ie = Is
        Qg, Qe = Qs
        xg, yg = np.median(Ig), np.median(Qg)
        xe, ye = np.median(Ie), np.median(Qe)

        if title is not None:
            plt.suptitle(title)
        ax.scatter(
            Ig, Qg, label="g", color="b", marker=".", edgecolor="None", alpha=0.2
        )
        ax.scatter(
            Ie, Qe, label="e", color="r", marker=".", edgecolor="None", alpha=0.2
        )
        ax.plot(
            [xg],
            [yg],
            color="k",
            linestyle=":",
            marker="o",
            markerfacecolor="b",
            markersize=5,
        )
        ax.plot(
            [xe],
            [ye],
            color="k",
            linestyle=":",
            marker="o",
            markerfacecolor="r",
            markersize=5,
        )
        ax.set_xlabel("I [ADC levels]")
        ax.set_ylabel("Q [ADC levels]")
        ax.legend(loc="upper right")
        ax.set_title(name, fontsize=14)
        ax.axis("equal")

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        scatter_plot("Unrotated", axs[0, 0], (Ig, Ie), (Qg, Qe))

    theta = -np.arctan2((ye - yg), (xe - xg))

    I_tot = np.concatenate((Ie, Ig))
    span = (np.max(I_tot) - np.min(I_tot)) / 2
    midpoint = (np.max(I_tot) + np.min(I_tot)) / 2
    xlims = [midpoint - span, midpoint + span]
    ng, binsg = np.histogram(Ig, bins=numbins, range=xlims)  # type: ignore
    ne, binse = np.histogram(Ie, bins=numbins, range=xlims)  # type: ignore
    contrast = np.abs(
        ((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum()))
    )

    # Rotate the IQ data
    Ig_new = Ig * np.cos(theta) - Qg * np.sin(theta)
    Qg_new = Ig * np.sin(theta) + Qg * np.cos(theta)

    Ie_new = Ie * np.cos(theta) - Qe * np.sin(theta)
    Qe_new = Ie * np.sin(theta) + Qe * np.cos(theta)

    # New means of each blob
    xg, yg = np.median(Ig_new), np.median(Qg_new)
    xe, ye = np.median(Ie_new), np.median(Qe_new)
    if verbose:
        verbose_print("Rotated", (Ig_new, Ie_new), (Qg_new, Qe_new))

    xlims = [(xg + xe) / 2 - span, (xg + xe) / 2 + span]

    if plot:
        scatter_plot("Rotated", axs[0, 1], (Ig_new, Ie_new), (Qg_new, Qe_new))
        # X and Y ranges for histogram

        ng, binsg, _ = axs[1, 0].hist(
            Ig_new, bins=numbins, range=xlims, color="b", label="g", alpha=0.5
        )
        ne, binse, _ = axs[1, 0].hist(
            Ie_new, bins=numbins, range=xlims, color="r", label="e", alpha=0.5
        )
        axs[1, 0].set_ylabel("Counts", fontsize=14)
        axs[1, 0].set_xlabel("I [ADC levels]", fontsize=14)
        axs[1, 0].legend(loc="upper right")

    else:
        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims)  # type: ignore
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims)  # type: ignore

    """Compute the fidelity using overlap of the histograms"""
    # this method calculates fidelity as 1-2(Neg + Nge)/N
    contrast = np.abs(
        ((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum()))
    )
    tind = contrast.argmax()
    threshold = binsg[tind]
    # this method calculates fidelity as (Ngg+Nee)/N = Ngg/N + Nee/N=(0.5N-Nge)/N + (0.5N-Neg)/N = 1-(Nge+Neg)/N
    fid = 0.5 * (1 - ng[tind:].sum() / ng.sum() + 1 - ne[:tind].sum() / ne.sum())
    if verbose:
        print(f"g correctly categorized: {100*(1-ng[tind:].sum()/ng.sum()):.3f}%")
        print(f"e correctly categorized: {100*(1-ne[:tind].sum()/ne.sum()):.3f}%")

    if plot:
        title = "${F}_{ge}$"
        axs[1, 0].set_title(f"Histogram ({title}: {100*fid:.3}%)", fontsize=14)
        axs[1, 0].axvline(threshold, color="0.2", linestyle="--")

        axs[1, 1].set_title("Cumulative Counts", fontsize=14)
        axs[1, 1].plot(binsg[:-1], np.cumsum(ng), "b", label="g")
        axs[1, 1].plot(binse[:-1], np.cumsum(ne), "r", label="e")
        axs[1, 1].axvline(threshold, color="0.2", linestyle="--")
        axs[1, 1].legend()
        axs[1, 1].set_xlabel("I [ADC levels]", fontsize=14)

        plt.subplots_adjust(hspace=0.25, wspace=0.15)
        plt.tight_layout()
        plt.show()

    return fid, threshold, theta * 180 / np.pi  # fids: ge, gf, ef
