"""Index layouts of the cycle fit's coordinate vectors.

A shared row holds the 14 coordinates every point shares. The moment Jacobian
starts with the first two drive probabilities per state, then the shared row.
The curve-model nuisance vector holds twelve endpoint coordinates, then the
shared row. A simplex is an (s, f) pair with probabilities (s*f, s*(1-f), 1-s);
see _cycle_emission.
"""


def shifted(row_slice: slice, offset: int) -> slice:
    """Return a shared-row slice placed in a vector whose shared row starts at offset."""
    return slice(row_slice.start + offset, row_slice.stop + offset)


# A shared row, in this order.
ROW_CENTERS = slice(0, 6)  # (I, Q) per state
ROW_VARIANCE = 6
ROW_THERMAL = slice(7, 9)  # (s, f)
ROW_RETENTION = slice(9, 12)  # one per state
ROW_LANDING = slice(12, 14)  # (s, f)
ROW_SIZE = 14

# Local moment Jacobian: six drive probabilities, then the shared row.
TABLE_DRIVE = slice(0, 6)
TABLE_SHARED = slice(TABLE_DRIVE.stop, TABLE_DRIVE.stop + ROW_SIZE)
TABLE_THERMAL = shifted(ROW_THERMAL, TABLE_SHARED.start)
TABLE_SIZE = TABLE_SHARED.stop

# The 26 nuisance coordinates that follow the curve parameters.
ENDPOINTS = slice(0, 12)  # by state, then endpoint 0/1, then (s, f)
SHARED = slice(ENDPOINTS.stop, ENDPOINTS.stop + ROW_SIZE)
CENTERS = shifted(ROW_CENTERS, SHARED.start)
VARIANCE = SHARED.start + ROW_VARIANCE
THERMAL = shifted(ROW_THERMAL, SHARED.start)
RETENTION = shifted(ROW_RETENTION, SHARED.start)
LANDING = shifted(ROW_LANDING, SHARED.start)
NUISANCE_SIZE = SHARED.stop
