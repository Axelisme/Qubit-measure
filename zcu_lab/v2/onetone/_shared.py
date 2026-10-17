"""Starting-value policy shared by the three single-tone spectroscopy Forms."""

from zcu_lab.templates import READOUT_PULSE_LENGTH_US


def initial_ro_length() -> float:
    """Return an editable RO window in us, 0.01 us shorter than the readout pulse.

    The pulse source is the readout Template, not resonator calibration. This
    Open/Reset seed is a direct value; later pulse edits do not recompute it.
    No calibration is read or written, and no clamping is applied.
    """
    return READOUT_PULSE_LENGTH_US - 0.01
