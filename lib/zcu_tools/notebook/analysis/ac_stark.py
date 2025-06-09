import numpy as np
import matplotlib.pyplot as plt
from zcu_tools.notebook.single_qubit.process import minus_background, rotate2real


def analyze_ac_stark_shift(pdrs, fpts, signals, chi, kappa, cutoff=None):
    amps = rotate2real(minus_background(signals, axis=1)).real
    max_idxs = np.argmax(amps, axis=1)
    max_freqs = fpts[max_idxs]

    # apply cutoff if provided
    if cutoff is not None:
        valid_indices = np.where(pdrs < cutoff)
        pdrs = pdrs[valid_indices]
        max_freqs = max_freqs[valid_indices]
        amps = amps[valid_indices, :]

    x_fit = np.linspace(min(pdrs), max(pdrs), 100)

    # fitting max_freqs with ax^2 + bx + c
    # a, b, c = np.polyfit(pdrs, max_freqs, 2)
    # y_fit = a * x_fit**2 + b * x_fit + c
    b, c = np.polyfit(pdrs, max_freqs, 1)
    y_fit = b * x_fit + c

    # Calculate the Stark shift
    eta = kappa**2 / (kappa**2 + chi**2)
    stark_shift_coeff = abs(b) / (2 * eta * chi)

    # plot the data and the fitted polynomial
    plt.figure(figsize=(10, 6))
    plt.imshow(
        amps.T,
        aspect="auto",
        extent=[min(pdrs), max(pdrs), min(fpts), max(fpts)],
        origin="lower",
        cmap="viridis",
    )
    plt.plot(pdrs, max_freqs, "o", label="Max Frequencies")

    plt.plot(x_fit, y_fit, "-", label="Fitted Polynomial")
    plt.xlabel("Probe Power (a.u.)")
    plt.ylabel("Resonator Frequency (MHz)")
    plt.title("AC Stark Shift Analysis")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return stark_shift_coeff
