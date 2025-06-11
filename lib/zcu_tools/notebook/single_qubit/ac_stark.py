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

    # plot the data and the fitted polynomial
    fig, ax1 = plt.subplots(figsize=figsize)

    # Use NonUniformImage for better visualization with pdr^2 as x-axis
    im = NonUniformImage(ax1, cmap="viridis", interpolation="nearest")
    im.set_data(pdrs2, fpts, amps.T)
    im.set_extent([pdrs2[0], pdrs2[-1], fpts[0], fpts[-1]])
    ax1.add_image(im)

    # Set proper limits for the plot
    ax1.set_xlim(pdrs2[0], pdrs2[-1])
    ax1.set_ylim(fpts[0], fpts[-1])

    # Plot the resonance frequencies and fitted curve
    ax1.plot(s_pdrs2, s_fpts, ".", c="k")

    # Fit curve in terms of pdr^2
    label = r"$\bar n$" + f" = {ac_coeff:.2g} x"
    ax1.plot(x2_fit, y_fit, "-", label=label)

    # Create secondary x-axis for pdr^2 (Readout Gain²)
    ax2 = ax1.twiny()

    # 主 x 軸顯示 avg_n，次 x 軸顯示 pdr^2
    # avg_n = ac_coeff * pdrs^2
    avg_n_ticks = ac_coeff * ax1.get_xticks()
    ax1.set_xticks(ax1.get_xticks())
    ax1.set_xticklabels([f"{avg_n:.1f}" for avg_n in avg_n_ticks])
    ax1.set_xlabel(r"Average Photon Number ($\bar n$)")

    # 上方次 x 軸顯示 pdr^2，使用科學記號顯示小數點後一位
    pdr2_ticks = ax1.get_xticks()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(pdr2_ticks)
    ax2.set_xticklabels([f"{pdr2:.1e}" for pdr2 in pdr2_ticks])
    ax2.set_xlabel("Readout Gain² (a.u.²)")

    ax1.set_ylabel("Qubit Frequency (MHz)")
    ax1.set_title("AC Stark Shift Analysis")
    ax1.legend()

    fig.tight_layout()
    plt.show()

    return ac_coeff
