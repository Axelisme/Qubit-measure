"""ADC-owned probability tolerance and explicit preparation for sampling."""

import numpy as np

from .model import RealArray

# End-to-end population accuracy is 1e-2, with a per-segment budget of 1e-3.
# This leaves ~38x margin over measured shaped-kernel trace drift of 2.62e-8,
# while probability preparation remains three orders below a segment budget.
POPULATION_TOLERANCE = 1e-6


class PopulationValidationError(ValueError):
    """Rejected population rows with a machine-readable count.

    rejected_rows is the positive number of rows that fail the reported
    validation rule. message names that rule; the exception text adds the count.
    No partially normalized result is available.
    """

    def __init__(self, message: str, rejected_rows: int) -> None:
        self.rejected_rows = rejected_rows
        super().__init__(f"{message}: {rejected_rows} rows rejected")


def validate_populations(values: RealArray, *, minimum: float = 0.0) -> RealArray:
    """Validate finite (*points, states) rows without changing them.

    The state axis must be nonempty. Row sums must lie in the closed float64
    interval [1-POPULATION_TOLERANCE, 1+POPULATION_TOLERANCE]. minimum is the
    lowest allowed entry, zero for sampling and -POPULATION_TOLERANCE only for
    explicit preparation. PopulationValidationError reports rejected_rows for
    the first failed rule. Invalid array shapes raise plain ValueError.
    """
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim < 1 or probabilities.shape[-1] == 0:
        raise ValueError("populations must have a nonempty state axis")
    nonfinite = np.count_nonzero(~np.isfinite(probabilities).all(axis=-1))
    if nonfinite:
        raise PopulationValidationError("populations must be finite", int(nonfinite))
    sums = probabilities.sum(axis=-1)
    bad_trace = (sums < 1 - POPULATION_TOLERANCE) | (sums > 1 + POPULATION_TOLERANCE)
    count = np.count_nonzero(bad_trace)
    if count:
        raise PopulationValidationError(
            "population trace outside tolerance", int(count)
        )
    count = np.count_nonzero(np.any(probabilities < minimum, axis=-1))
    if count:
        raise PopulationValidationError(
            "negative populations outside tolerance", int(count)
        )
    return probabilities


def normalize_populations(values: RealArray) -> RealArray:
    """Explicitly prepare evolution populations for ADC sampling, without mutation.

    values: finite float64 (*points, states), in readout state order, with a
        nonempty state axis. Each row sum must be in [1-1e-6, 1+1e-6], including
        the float64 endpoints, and each entry must be at least -1e-6.
    Return an owned array: set only tolerated negative entries to zero, then
    divide each row by its sum. No upper clipping occurs. Invalid
    array shapes raise ValueError. Nonfinite values or residuals outside
    tolerance raise PopulationValidationError with rejected_rows for the first
    failed rule; no partially repaired result is returned.

    The 1e-6 tolerance is 1000x below the 1e-3 segment budget and 10000x below
    the 1e-2 end-to-end population requirement. It gives ~38x margin above the
    measured 2.62e-8 shaped-kernel trace drift. Sampling validates but never
    calls this function: the environment must request preparation explicitly.
    """
    probabilities = validate_populations(values, minimum=-POPULATION_TOLERANCE)
    prepared = np.maximum(probabilities, 0)
    prepared /= prepared.sum(axis=-1, keepdims=True)
    return prepared
