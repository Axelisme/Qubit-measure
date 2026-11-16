"""Constant-tone prepare-key construction shared with the t6 Kernel protocol."""

import numpy as np
from numpy.typing import NDArray

from .kernel import SegmentTable


def prepare_keys(table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.void]:
    """Return structured t6 prepare keys aligned with constant-tone table rows.

    rows indexes valid rows with one constant-envelope tone. Keys contain flux
    level, shift content, port, signed carrier GHz, signed gain and duration ns;
    phase and start time are apply-only. The Engine supplies source namespacing.
    The returned array owns its storage and preserves signed-zero bits.
    Its layout is opaque to consumers.
    """
    keys = np.empty(
        len(rows),
        dtype=[
            ("level", "<i8", (table.level.shape[1],)),
            ("shift", "S64"),
            ("port", "<i8"),
            ("frequency", "<f8"),
            ("amplitude", "<f8"),
            ("duration", "<f8"),
        ],
    )
    if not len(rows):
        return keys
    keys["level"] = table.level[rows]
    keys["shift"] = table.shift_content[rows]
    keys["port"] = table.tone_port[rows, 0]
    keys["frequency"] = table.tone_frequency_ghz[rows, 0]
    keys["amplitude"] = table.tone_amplitude[rows, 0]
    keys["duration"] = table.duration_ns[rows]
    return keys
