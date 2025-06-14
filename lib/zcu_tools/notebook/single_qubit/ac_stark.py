from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import NonUniformImage

from zcu_tools.notebook.single_qubit.process import minus_background, rotate2real
from zcu_tools.notebook.util.fitting import fitlor

from .general import figsize


def get_resonance_freq(
    pdrs: np.ndarray, fpts: np.ndarray, amps: np.ndarray, cutoff=None
) -> np.ndarray:
    s_pdrs = []
    s_fpts = []
    prev_freq = fitlor(fpts, amps[0])[0][3]

    fitparams = [None, None, None, prev_freq, None]
    for pdr, amp in zip(pdrs, amps):
        curr_freq = fitlor(fpts, amp, fitparams=fitparams)[0][3]
        if abs(curr_freq - prev_freq) < 0.1 * (fpts[-1] - fpts[0]):
            s_pdrs.append(pdr)
            s_fpts.append(curr_freq)

            prev_freq = curr_freq
            fitparams[3] = curr_freq

    return np.array(s_pdrs), np.array(s_fpts)


def analyze_ac_stark_shift(
    pdrs: np.ndarray,
    fpts: np.ndarray,
    signals: np.ndarray,
    chi: float,
    kappa: float,
    deg: int = 2,
    cutoff: Optional[float] = None,
) -> float:
    # apply cutoff if provided
    if cutoff is not None:
        valid_indices = np.where(pdrs < cutoff)[0]
        pdrs = pdrs[valid_indices]
        signals = signals[valid_indices, :]

    amps = rotate2real(minus_background(signals, axis=1)).real
    amps /= np.std(amps, axis=1, keepdims=True)
    s_pdrs, s_fpts = get_resonance_freq(pdrs, fpts, amps)

    pdrs2 = pdrs**2
    s_pdrs2 = s_pdrs**2

    # fitting max_freqs with ax2 + bx + c
    x2_fit = np.linspace(min(pdrs2), max(pdrs2), 100)
    if deg == 1:
        b, c = np.polyfit(s_pdrs2, s_fpts, 1)
        y_fit = b * x2_fit + c
    elif deg == 2:
        a, b, c = np.polyfit(s_pdrs2, s_fpts, 2)
        y_fit = a * x2_fit**2 + b * x2_fit + c
    else:
        raise ValueError(f"Degree {deg} is not supported.")

    # Calculate the Stark shift
    eta = kappa**2 / (kappa**2 + chi**2)
    ac_coeff = abs(b) / (2 * eta * chi)

    avg_n = ac_coeff * pdrs2

    # plot the data and the fitted polynomial
    fig, ax1 = plt.subplots(figsize=figsize)

    # Use NonUniformImage for better visualization with pdr^2 as x-axis
    im = NonUniformImage(ax1, cmap="viridis", interpolation="nearest")
    im.set_data(avg_n, fpts, amps.T)
    im.set_extent([avg_n[0], avg_n[-1], fpts[0], fpts[-1]])
    ax1.add_image(im)

    # Set proper limits for the plot
    ax1.set_xlim(avg_n[0], avg_n[-1])
    ax1.set_ylim(fpts[0], fpts[-1])

    # Plot the resonance frequencies and fitted curve
    ax1.plot(ac_coeff * s_pdrs2, s_fpts, ".", c="k")

    # Fit curve in terms of pdr^2
    label = r"$\bar n$" + f" = {ac_coeff:.2g} " + r"$gain^2$"
    n_fit = ac_coeff * x2_fit
    ax1.plot(n_fit, y_fit, "-", label=label)

    # Create secondary x-axis for pdr^2 (Readout Gain²)
    ax2 = ax1.twiny()

    # main x-axis: avg_n, secondary x-axis: pdr^2
    # avg_n = ac_coeff * pdrs^2
    ax1.set_xticks(ax1.get_xticks())
    # ax1.set_xticklabels([f"{avg_n:.1f}" for avg_n in ax1.get_xticks()])
    ax1.set_xlabel(r"Average Photon Number ($\bar n$)", fontsize=14)

    # 上方次 x 軸顯示 pdr
    avgn_ticks = ax1.get_xticks()
    pdr_ticks = np.sqrt(avgn_ticks / ac_coeff)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(avgn_ticks)
    ax2.set_xticklabels([f"{pdr:.2f}" for pdr in pdr_ticks])
    ax2.set_xlabel("Readout Gain (a.u.)", fontsize=14)

    ax1.set_ylabel("Qubit Frequency (MHz)", fontsize=14)
    ax1.legend(fontsize="x-large")
    ax1.tick_params(axis="both", which="major", labelsize=12)

    fig.tight_layout()
    plt.show()

    return ac_coeff
