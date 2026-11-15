"""Small public-Engine inputs for legacy, sideband and refused Floquet rows."""

from dataclasses import replace

import numpy as np
from zcu_tools.simulate.evolution import EnvelopeSpec, SecularDrive, SecularModel

from .accuracy_cases import (
    AccuracyCase,
    accuracy_case,
    coherent_source,
    drive_track,
    timeline,
)


def floquet_case(name: str) -> AccuracyCase:
    """Return a fresh caller-owned source and constant two-tone timeline.

    Names are legacy, sideband, bound-refusal, dynamic-range, invalid-frame,
    mixed-duration and mixed-gain. Mixed inputs share one frame and probe bin
    across an accepted row and a refused row. Unknown names raise ValueError.
    Durations are ns, carriers GHz, and every initial density has coherence.
    """
    if name == "legacy":
        return accuracy_case("two_carriers")
    if name not in (
        "sideband",
        "bound-refusal",
        "dynamic-range",
        "invalid-frame",
        "mixed-duration",
        "mixed-gain",
    ):
        raise ValueError(f"unknown Floquet case: {name}")
    size = 3 if name == "invalid-frame" else 2
    zero = np.zeros((size, size), complex)
    drive = zero.copy()
    drive[0, 1] = drive[1, 0] = 0.02
    rates = np.zeros((size, size))
    rates[0, 1] = 0.001
    if name == "invalid-frame":
        rates[0, 1] = rates[1, 2] = 0.001
    if name == "dynamic-range":
        rates[0, 1] = 0.1
    model = SecularModel(
        (0,),
        name,
        np.arange(size, dtype=np.float64) * 0.13,
        np.eye(size, dtype=complex),
        (
            SecularDrive("stark", zero if name == "invalid-frame" else drive, zero),
            SecularDrive("probe", drive, zero),
        ),
        rates,
        np.zeros((size, size)),
    )
    batch = timeline(2 if name in ("mixed-duration", "mixed-gain") else 1, 16.0)
    if name == "dynamic-range":
        batch = replace(batch, end_ns=np.array([160.0]))
    elif name == "mixed-duration":
        batch = replace(batch, end_ns=np.array([16.0, 16000.0]))
    envelope = EnvelopeSpec("constant")
    batch = replace(
        batch,
        tracks=(
            drive_track(batch, envelope, frequency=0.13, gain=0.2, port="stark"),
            drive_track(
                batch,
                envelope,
                frequency=0.11,
                gain=0.2 if name == "bound-refusal" else 0.0001,
                port="probe",
            ),
        ),
    )
    if name == "mixed-gain":
        batch = replace(
            batch,
            tracks=(
                batch.tracks[0],
                replace(batch.tracks[1], amplitude=np.array([0.0001, 0.2])),
            ),
        )
    return AccuracyCase(name, coherent_source(model), batch, ("floquet", "dop853"))
