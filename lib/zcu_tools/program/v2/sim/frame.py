"""Phase-coherent drives at several carriers within one rotating frame.

Lowering expresses idle evolution in one reference frame ``f_ref`` and builds
each drive segment in its own carrier frame (``delta = 2*pi*(f_qubit -
f_drive)``, phase = cfg phase).  QICK DDS generators are phase-coherent: a drive
at ``f_drive`` has phase ``phi + 2*pi*(f_drive - f_ref)*t`` in the reference
frame.  With ``Delta = 2*pi*(f_drive - f_ref)``, a drive segment starting at
``t0`` with duration ``T`` maps the reference-frame Bloch vector by::

    U_ref = R_z(Delta*T) @ U_drive(phase + Delta*t0)

``R_z`` commutes with idle precession and T1/T2 decay and shifts a later
drive's phase by its angle, so each trailing ``R_z`` moves to the end of the
timeline, where it leaves ``P_e`` unchanged.  Drive ``k`` therefore needs the
phase ``phi_k + integral_0^t0_k (Delta_k - Delta(t)) dt``, where ``Delta(t)`` is
the carrier offset of the segment active at ``t`` (0 while idle).

Time starts at 0 at the start of each shot.  Hardware measures DDS phase from
absolute tProc time, so the model matches hardware only when every carrier
offset times the shot period is a whole number of cycles.
"""

from __future__ import annotations

from collections.abc import Sequence

from .bloch import Segment


def align_drive_phases(
    segments: Sequence[Segment], frame_delta: float
) -> list[Segment]:
    """Give each drive segment its phase-coherent phase in the reference frame.

    ``frame_delta`` is the idle detuning (rad/µs) of the reference frame.  A
    segment's carrier offset is ``frame_delta - segment.delta``, so idle
    segments have offset 0 and a timeline with one carrier comes back
    unchanged.
    """

    aligned: list[Segment] = []
    clock = 0.0
    played = 0.0  # integral of the active carrier offset over the clock
    for segment in segments:
        offset = frame_delta - segment.delta
        shift = offset * clock - played
        if shift != 0.0:
            segment = segment._replace(phase=segment.phase + shift)
        aligned.append(segment)
        clock += segment.t
        played += offset * segment.t
    return aligned
