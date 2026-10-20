"""State-conditioned vacuum fields before scattering and receiver calibration."""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ._trajectories import PathFields
from .model import ReadoutModel, RealArray, readonly_array
from .request import ReadoutPulse


def cavity_fields(
    model: ReadoutModel,
    pulses: tuple[ReadoutPulse, ...],
    times_ns: RealArray,
) -> NDArray[np.complex128]:
    """Return coherent intracavity alpha, shaped (time,state,2), in sqrt(photons).

    model sets joint-state order, resonant frequencies, reciprocal shared-line
    damping and the rotating reference. pulses are incident fields in that
    reference's pulse-local convention. Each starts from vacuum; their fields
    add coherently, including ringdown and overlapping/different-start supports.
    times_ns is a finite nonempty strictly increasing emitted-ns vector.
    Times before a pulse starts return zero. Empty pulses return vacuum.

    States remain fixed branches: directional population rates do not mix them.
    Receiver delay, gain, offset, noise and mismatch scattering phases do not
    change intracavity fields. Output owns its complex128 storage. Invalid
    coordinates or envelope values raise ValueError; solver failure raises
    RuntimeError. The field solver uses rtol=1e-9 and atol=1e-11.
    """
    times = readonly_array(times_ns, "cavity times")
    if times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("cavity times must be nonempty and strictly increasing")
    fields = np.zeros((len(times), len(model.state_labels), 2), dtype=np.complex128)
    for pulse in pulses:
        # Integrate near this carrier, then restore the caller's pulse-local
        # rotating coordinates. Absolute microwave frequencies need no ODE steps.
        carrier_model = replace(model, reference_frequency_ghz=pulse.frequency_ghz)
        path = PathFields(carrier_model, pulse, times, 1e-9, 1e-11)
        phase = np.exp(
            -2j
            * np.pi
            * (pulse.frequency_ghz - model.reference_frequency_ghz)
            * (times - pulse.start_ns)
        )
        fields += path.baseline(times) * phase[:, None, None]
    return fields
