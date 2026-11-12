"""Pure numerical ALLXY coherent-error model and fitting."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class AllXYFitResult:
    """Coherent-error fit and signal-domain curve in caller sequence order.

    center: midpoint of the g/e levels, in the input signal unit.
    contrast: signed g-minus-e level difference, in the input signal unit.
    amplitude_error: relative drive amplitude error; +0.02 means 2% overrotation.
    detune_param: dimensionless Reed detuning expansion parameter, not Hz.
    predict_signals: float64 array (N,) of fitted signals in the input unit,
        ordered exactly as the supplied gate-pair sequence.
    power_err: mean absolute Bloch-z deviation over the supplied pairs caused
        by the fitted amplitude error alone, with detuning zero.
    detune_err: mean absolute Bloch-z deviation over the supplied pairs caused
        by the fitted detuning alone, with amplitude error zero.
    residual_rms: signal-domain RMS residual divided by half absolute contrast,
        in Bloch-z units. The error summaries are not gate infidelities.
    """

    center: float
    contrast: float
    amplitude_error: float
    detune_param: float
    predict_signals: NDArray[np.float64]
    power_err: float
    detune_err: float
    residual_rms: float


def predict_state_with_error(
    gates: tuple[str, str], power_err: float, detune_err: float
) -> float:
    """Predict scalar Bloch-z after an ordered ALLXY gate pair.

    ``gates`` contains two names from I, X90, Y90, X180, Y180, in playback
    order. Only pairs in the Reed thesis table (including reversed I pairs)
    are supported; other pairs raise ValueError. Ground is +1, excited -1.
    ``power_err`` is the pi/2 rotation-angle error in radians, equal to
    (pi/2) times the relative amplitude error. ``detune_err`` is the
    dimensionless detuning expansion parameter, not a frequency in Hz.
    The low-order polynomial is not clipped to [-1, 1]. Inputs are not mutated;
    there is no plotting or I/O. Non-finite error parameters propagate through
    the polynomial; callers supply small finite expansion parameters.
    """
    ep = power_err
    ed = detune_err

    # reference: https://rsl.yale.edu/sites/default/files/2024-08/2013-RSL-Thesis-Matthew-Reed.pdf
    # page 154

    if gates == ("I", "I"):
        state = 1
    elif gates in [("X180", "X180"), ("Y180", "Y180")]:
        state = 1 - 8 * ep**2 - (np.pi**2 / 32) * ed**4
    elif gates in [("X180", "Y180"), ("Y180", "X180")]:
        state = 1 - 4 * ep**2 - ed**2
    elif gates in [("X90", "I"), ("Y90", "I"), ("I", "X90"), ("I", "Y90")]:
        state = -ep + (1 - np.pi / 2) * ed**2
    elif gates == ("X90", "Y90"):
        state = ep**2 - 2 * ed
    elif gates == ("Y90", "X90"):
        state = ep**2 + 2 * ed
    elif gates in [("X90", "Y180"), ("X180", "Y90")]:
        state = ep - ed
    elif gates in [("Y90", "X180"), ("Y180", "X90")]:
        state = ep + ed
    elif gates in [("X90", "X180"), ("X180", "X90"), ("Y90", "Y180"), ("Y180", "Y90")]:
        state = 3 * ep + (3 * np.pi / 8) * ed**2
    elif gates in [("X180", "I"), ("Y180", "I"), ("I", "X180"), ("I", "Y180")]:
        state = -1 + 2 * ep**2 + 0.5 * ed**2
    elif gates in [("X90", "X90"), ("Y90", "Y90")]:
        state = -1 + 2 * ep**2 + 2 * ed**2
    else:
        raise ValueError(f"Invalid gate pair: {gates}")
    return state


def fit_allxy_errors(
    sequence: Sequence[tuple[str, str]],
    real_signals: NDArray[np.float64],
) -> AllXYFitResult:
    """Fit the Reed ALLXY model to real signals, without IQ rotation or I/O.

    ``sequence`` is the ordered gate-pair list, with an (I, I) reference pair;
    pair names and supported pairs follow ``predict_state_with_error``.
    ``real_signals`` is a finite float64 array (N,), one signal per pair in
    the same order, in any consistent signal unit with nonzero contrast.
    Inputs are not mutated. Center starts at the measured extrema midpoint;
    signed contrast starts at their difference, positive if (I, I) is high.
    Center, contrast, angle error ep (radians) and dimensionless detuning ed
    are always fitted. With ptp = max(real_signals) - min(real_signals), center
    is bounded to [min - ptp, max + ptp]. Contrast keeps its initial sign and
    its absolute value is bounded to [0.5 * ptp, 3 * ptp], allowing g/e levels
    beyond the measured extrema. Both errors start at zero and are bounded to
    [-0.2, 0.2]. The model includes coherent errors only, not T1/T2 decay.
    The result owns the curve and separated Bloch-z error summaries;
    see AllXYFitResult for every field/unit.
    Missing reference/unsupported pairs, non-finite signals or infeasible
    bounds raise ValueError. Malformed arrays propagate NumPy/SciPy errors.
    Optimizer nonconvergence raises RuntimeError; SciPy warnings propagate.
    Zero measured contrast raises ValueError because the bounds coincide.
    No fallback or clipping is applied.
    """
    g_signal = real_signals[sequence.index(("I", "I"))]
    init_center = 0.5 * (np.max(real_signals) + np.min(real_signals))
    signal_span = np.ptp(real_signals)
    init_contrast = signal_span
    if g_signal < init_center:
        init_contrast = -init_contrast

    def calc_sim_signal(
        seq: tuple[str, str], center: float, contrast: float, ep: float, ed: float
    ) -> float:
        return center + 0.5 * contrast * predict_state_with_error(seq, ep, ed)

    contrast_lower, contrast_upper = sorted((0.5 * init_contrast, 3 * init_contrast))
    params, *_ = curve_fit(
        lambda _, *args: [calc_sim_signal(seq, *args) for seq in sequence],
        np.arange(len(sequence)),
        real_signals,
        p0=(init_center, init_contrast, 0.0, 0.0),
        bounds=(
            (np.min(real_signals) - signal_span, contrast_lower, -0.2, -0.2),
            (np.max(real_signals) + signal_span, contrast_upper, 0.2, 0.2),
        ),
    )
    center, contrast, ep, ed = params

    predict_signals = [
        calc_sim_signal(seq, center, contrast, ep, ed) for seq in sequence
    ]
    # ep is the pi/2 rotation-angle error, (pi/2) * amplitude error.
    amplitude_error = ep / (np.pi / 2)
    residual_rms = np.sqrt(
        np.mean((real_signals - np.asarray(predict_signals)) ** 2)
    ) / (0.5 * np.abs(contrast))

    perfect_states = [predict_state_with_error(seq, 0.0, 0.0) for seq in sequence]
    power_err = np.mean(
        [
            np.abs(predict_state_with_error(seq, ep, 0.0) - perf_state)
            for seq, perf_state in zip(sequence, perfect_states, strict=False)
        ]
    )
    detune_err = np.mean(
        [
            np.abs(predict_state_with_error(seq, 0.0, ed) - perf_state)
            for seq, perf_state in zip(sequence, perfect_states, strict=False)
        ]
    )
    return AllXYFitResult(
        center=float(center),
        contrast=float(contrast),
        amplitude_error=float(amplitude_error),
        detune_param=float(ed),
        predict_signals=np.asarray(predict_signals, dtype=np.float64),
        power_err=float(power_err),
        detune_err=float(detune_err),
        residual_rms=float(residual_rms),
    )
