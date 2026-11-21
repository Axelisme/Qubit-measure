"""Coordinate conversion for the ADC receiver's line delay."""

from math import isfinite
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

_RealScalar = float | np.floating[Any] | np.integer[Any]
_TimeArray = NDArray[np.float64]


def valid_time_of_flight_ns(delay_ns: float) -> bool:
    """Return whether the receiver delay in ns is finite and nonnegative."""
    return isfinite(delay_ns) and delay_ns >= 0


@overload
def adc_to_device_time_ns(
    adc_time_ns: _RealScalar, *, time_of_flight_ns: float
) -> float: ...


@overload
def adc_to_device_time_ns(
    adc_time_ns: _TimeArray, *, time_of_flight_ns: float
) -> _TimeArray: ...


def adc_to_device_time_ns(
    adc_time_ns: _RealScalar | _TimeArray, *, time_of_flight_ns: float
) -> float | _TimeArray:
    """Return device time in ns by subtracting the receiver's line delay once.

    Pass finite ADC times in ns from SampleGrid or TimeBase, as a real scalar
    or a float64 ndarray. Finiteness of those times is not rechecked. Pass
    Receiver.time_of_flight_ns, directly or from the channel's delay mapping,
    as time_of_flight_ns. A nonfinite or negative delay raises ValueError.

    The result is a Python float or a new float64 array of the same shape.
    Negative device times are valid. Use the result for emitted-time
    coordinates; do not shift it again or replace ADC times used for
    demodulation.
    """
    if not valid_time_of_flight_ns(time_of_flight_ns):
        raise ValueError(
            f"time_of_flight_ns must be finite and nonnegative: {time_of_flight_ns}"
        )
    if isinstance(adc_time_ns, np.ndarray):
        return np.asarray(adc_time_ns - time_of_flight_ns)
    return float(float(adc_time_ns) - time_of_flight_ns)
