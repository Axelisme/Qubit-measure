"""Numeric interval tools shared by the global mux flux Forms."""


def global_flux_interval(
    integers: list[float | None] | None,
    halves: list[float | None] | None,
) -> tuple[float | None, float | None]:
    """Return the union of actual selected-device intervals, or two blanks.

    Both lists must align to the same selected qubits. Any absent calibration
    leaves both endpoints unfilled. Inputs are validated component fields.
    """
    if (
        integers is None
        or halves is None
        or any(value is None for value in (*integers, *halves))
    ):
        return None, None
    endpoints = [
        endpoint
        for integer, half in zip(integers, halves, strict=True)
        if integer is not None and half is not None
        for endpoint in (1.1 * integer - 0.1 * half, 1.1 * half - 0.1 * integer)
    ]
    return min(endpoints), max(endpoints)
