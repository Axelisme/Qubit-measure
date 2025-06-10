from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

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
    s_pdrs, s_fpts = get_resonance_freq(pdrs, fpts, amps)

    # fitting max_freqs with ax2 + bx + c
    x_fit = np.linspace(min(pdrs), max(pdrs), 100)
    if deg == 1:
        b, c = np.polyfit(s_pdrs**2, s_fpts, 1, w=1 / (0.1 + s_pdrs))
        y_fit = b * x_fit**2 + c
    elif deg == 2:
        a, b, c = np.polyfit(s_pdrs**2, s_fpts, 2, w=1 / (0.1 + s_pdrs))
        y_fit = a * x_fit**4 + b * x_fit**2 + c
    else:
        raise ValueError(f"Degree {deg} is not supported.")

    # Calculate the Stark shift
    eta = kappa**2 / (kappa**2 + chi**2)
    stark_shift_coeff = abs(b) / (2 * eta * chi)

    # plot the data and the fitted polynomial
    plt.figure(figsize=figsize)
    plt.imshow(
        amps.T,
        aspect="auto",
        extent=[pdrs[0], pdrs[-1], fpts[0], fpts[-1]],
        origin="lower",
        cmap="viridis",
    )
    plt.plot(s_pdrs, s_fpts, ".", c="k")
    plt.plot(
        x_fit, y_fit, "-", label=f"Stark Shift = {stark_shift_coeff:.2f} x\u00b2 MHz"
    )
    plt.xlabel("Readout Gain (a.u.)")
    plt.ylabel("Qubit Frequency (MHz)")
    plt.title("AC Stark Shift Analysis")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return stark_shift_coeff
