"""Projection and optional Native persistence of mux per-tone IQ moments."""

from typing import overload

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.shot_uncertainty import iq_shot_sem
from zcu_tools.experiment import VariableAxisSpec, VariableSpec, VariableZSpec
from zcu_tools.experiment.axes_spec import LoadedVariableData
from zcu_tools.utils.process import find_rotate_angle

_COLUMNS = (0, 1, 2, 3, 4, 5)


@overload
def project_iq(
    signals: NDArray[np.complex128],
    moments: NDArray[np.float64],
    *,
    skip_for_axis: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


@overload
def project_iq(
    signals: NDArray[np.complex128],
    moments: None,
    *,
    skip_for_axis: int = 0,
) -> tuple[NDArray[np.float64], None]: ...


def project_iq(
    signals: NDArray[np.complex128],
    moments: NDArray[np.float64] | None,
    *,
    skip_for_axis: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Project a tone's complex mean and optional shot SEM onto the same axis.

    Use the existing principal IQ direction, taking I when there are too few
    values to estimate it. skip_for_axis omits leading positions only when
    estimating the direction, not when returning points. NaN marks missing
    IQ; moments are the corresponding six sums, or None for mean-only data.
    Return projected mean and optional SEM of the same shape. Invalid moments
    raise ValueError from the SEM estimator; no input array is modified.
    """
    try:
        angle = find_rotate_angle(signals[skip_for_axis:])
    except ValueError:
        real = signals.real
        direction = (1.0, 0.0)
    else:
        rotation = np.exp(-1j * angle)
        real = (signals * rotation).real
        direction = (float(rotation.real), -float(rotation.imag))
    return real, None if moments is None else iq_shot_sem(moments, direction=direction)


def iq_moments_variable(
    iq_axes: tuple[VariableAxisSpec, ...],
) -> VariableSpec:
    """Declare optional float64 IQ moments on the mean's tone/point grid.

    iq_axes describes the mean's inner-first axes; the innermost moments axis
    contains columns [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²]. The Result shape is (K,P,6).
    """
    return VariableSpec(
        "iq_moments",
        (VariableAxisSpec(label="Moment", unit="", values=_COLUMNS), *iq_axes),
        VariableZSpec("iq_moments", label="IQ moments", unit=""),
        required=False,
    )


def restore_iq_moments(
    moments: LoadedVariableData | None, mean: LoadedVariableData
) -> NDArray[np.float64] | None:
    """Restore optional moments; reject axes different from the mean's grid.

    Only an absent Native variable yields None. Changed column coordinates or
    tone/point coordinates raise ValueError. The returned array is not copied.
    """
    if moments is None:
        return None
    columns, *grid = moments.axes
    if not np.array_equal(columns, _COLUMNS) or any(
        not np.array_equal(a, b) for a, b in zip(grid, mean.axes, strict=True)
    ):
        raise ValueError("/vars/iq_moments: expected six columns on the mean IQ grid")
    return moments.z


def validate_iq_moments(
    moments: NDArray[np.float64] | None, signals: NDArray[np.complex128]
) -> None:
    """Raise for wrong float64 shape or invalid counts/covariance on the IQ grid.

    Accept None for mean-only records and NaN for unavailable moments. The SEM
    estimator owns covariance and count validation, including rounding tolerance.
    """
    if moments is None:
        return
    if moments.dtype != np.float64 or moments.shape != (*signals.shape, 6):
        raise ValueError("iq_moments must be float64 (tones, points, 6)")
    iq_shot_sem(moments, direction=(1.0, 0.0))
