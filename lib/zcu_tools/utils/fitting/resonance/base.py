from typing import List, Optional, Tuple, Union, cast

import numpy as np
import scipy as sp

from ..base import fit_func


def get_rough_edelay(fpts, signals) -> float:
    signal_ratios = signals[1:] / (signals[:-1] + 1e-12)

    slope = np.median(np.angle(signal_ratios)) / (fpts[1] - fpts[0])

    return -slope / (2 * np.pi)


def remove_edelay(fpts, signals, edelay: float) -> np.ndarray:
    return np.exp(1j * 2 * np.pi * fpts * edelay) * signals


def calc_M(xs, ys) -> np.ndarray:
    zs = xs**2 + ys**2
    N = len(zs)
    Mx = np.sum(xs)
    My = np.sum(ys)
    Mz = np.sum(zs)
    Mxx = np.sum(xs**2)
    Myy = np.sum(ys**2)
    Mzz = np.sum(zs**2)
    Mxz = np.sum(xs * zs)
    Myz = np.sum(ys * zs)
    Mxy = np.sum(xs * ys)

    M = np.array(
        [
            [Mzz, Mxz, Myz, Mz],
            [Mxz, Mxx, Mxy, Mx],
            [Myz, Mxy, Myy, My],
            [Mz, Mx, My, N],
        ]
    )
    return M


def fit_circle_params(xs, ys) -> Tuple[float, float, float]:
    """[center_x, center_y, radius]"""
    mean_x, mean_y = np.mean(xs), np.mean(ys)
    xs = xs - mean_x
    ys = ys - mean_y

    # calculate M matrix
    M = calc_M(xs, ys)
    B = np.array(
        [
            [0, 0, 0, -2],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-2, 0, 0, 0],
        ]
    )

    eigvals, eigvecs = sp.linalg.eig(M, B)
    eigvals = eigvals.real

    # find the smallest non-negative eigenvalue
    min_eigval = np.min(eigvals[eigvals >= 0])
    eigvec = eigvecs[:, eigvals == min_eigval][:, 0]

    A, B, C, D = eigvec
    center_x = mean_x - B / (2 * A)
    center_y = mean_y - C / (2 * A)
    radius = np.sqrt(B**2 + C**2 - 4 * A * D) / abs(2 * A)

    return center_x, center_y, radius


def fit_edelay(fpts, signals) -> float:
    rough_edelay = get_rough_edelay(fpts, signals)
    signals = remove_edelay(fpts, signals, rough_edelay)

    def loss_func(edelay):
        rot_signals = remove_edelay(fpts, signals, edelay)
        xc, yc, r0 = fit_circle_params(rot_signals.real, rot_signals.imag)
        return np.sum(
            (r0 - np.sqrt((rot_signals.real - xc) ** 2 + (rot_signals.imag - yc) ** 2))
            ** 2
        )

    fit_range = 5.0 / np.ptp(fpts)
    edelays = np.linspace(-fit_range, fit_range, 1000)
    loss_values = [loss_func(edelay) for edelay in edelays]
    edelay = edelays[np.argmin(loss_values)] + rough_edelay

    return edelay


def calc_phase(signals, xc, yc) -> np.ndarray:
    return np.unwrap(np.angle(signals - (xc + 1j * yc)))


def phase_func(fpts, resonant_f, Ql, theta0: float) -> np.ndarray:
    return theta0 + 2 * np.arctan(2 * Ql * (1 - fpts / resonant_f))


def fit_resonant_params(
    fpts, signals, circle_params: Tuple[float, float, float], fit_theta0=True
) -> Tuple[float, float, float]:
    """[resonant_freq, Ql, theta0]"""
    phases = calc_phase(signals, circle_params[0], circle_params[1])

    magnitudes = np.abs(signals - 0.5 * (signals[0] + signals[-1]))
    fwhm = np.ptp(fpts) * np.sum(magnitudes > 0.5 * np.max(magnitudes)) / len(fpts)

    init_freq = fpts[np.argmax(np.abs(np.diff(signals)))]
    init_Ql = 2 * init_freq / fwhm
    init_theta0 = 0.5 * (np.max(phases) + np.min(phases))

    fixedparams: List[Union[float, None]] = [None] * 3
    if not fit_theta0:
        init_theta0 = np.angle(circle_params[0] + 1j * circle_params[1]).item()
        while init_theta0 < np.min(phases):
            init_theta0 += 2 * np.pi
        while init_theta0 > np.max(phases):
            init_theta0 -= 2 * np.pi
        fixedparams[2] = init_theta0

    pOpt, _ = fit_func(
        fpts,
        phases,
        phase_func,
        init_p=[init_freq, init_Ql, init_theta0],
        bounds=(
            [np.min(fpts), 0, init_theta0 - np.pi],
            [np.max(fpts), 5 * init_Ql, init_theta0 + np.pi],
        ),
        fixedparams=fixedparams,
    )

    return cast(Tuple[float, float, float], tuple(pOpt))


def normalize_signal(
    signals: np.ndarray, circle_params: Tuple[float, float, float], a0: complex
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    xc, yc, r0 = circle_params
    center = xc + 1j * yc

    norm_signals = signals / a0
    norm_center = center / a0
    norm_xc, norm_yc = norm_center.real, norm_center.imag
    norm_r0 = r0 / np.abs(a0)

    return norm_signals, (norm_xc, norm_yc, norm_r0)
