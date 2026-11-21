"""Coordinate conversion for the ADC receiver's line delay."""

from math import isfinite
from typing import Any, overload

import numpy as np

from .model import RealArray

_RealScalar = float | np.floating[Any] | np.integer[Any]


@overload
def adc_to_device_time_ns(
    adc_time_ns: _RealScalar, *, time_of_flight_ns: float
) -> float: ...


@overload
def adc_to_device_time_ns(
    adc_time_ns: RealArray, *, time_of_flight_ns: float
) -> RealArray: ...


def adc_to_device_time_ns(
    adc_time_ns: _RealScalar | RealArray, *, time_of_flight_ns: float
) -> float | RealArray:
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
    if not isfinite(time_of_flight_ns) or time_of_flight_ns < 0:
        raise ValueError(
            f"time_of_flight_ns must be finite and nonnegative: {time_of_flight_ns}"
        )
    if isinstance(adc_time_ns, np.ndarray):
        return np.asarray(adc_time_ns - time_of_flight_ns)
    return float(float(adc_time_ns) - time_of_flight_ns)
