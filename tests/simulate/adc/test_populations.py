"""Explicit probability preparation and validation at the public ADC seam."""

import numpy as np
import pytest
from zcu_tools.simulate import adc


@pytest.mark.parametrize("trace", [1 - 9e-7, 1 + 9e-7])
def test_population_input_accepts_engine_trace_residual_without_repair(trace):
    values = np.array([0.25, 0.75]) * trace
    try:
        mixture = adc.PopulationInput(("g", "e"), values)
    except ValueError as error:
        pytest.fail(f"in-budget engine trace residual must be accepted: {error}")
    np.testing.assert_array_equal(mixture.values, values)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([[0.25 * (1 + 2.62e-8), 0.75 * (1 + 2.62e-8)]], [[0.25, 0.75]]),
        ([[-1e-6, 1.000001]], [[0, 1]]),
        ([[np.nextafter(-1e-6, -1), 1.000001]], [[0, 1]]),
        (
            [[0.0, 0.999999], [0.0, 1.000001]],
            [[0, 1], [0, 1]],
        ),
    ],
)
def test_explicit_normalization_repairs_trace_residuals_and_negatives(values, expected):
    assert callable(getattr(adc, "normalize_populations", None)), (
        "ADC must expose explicit probability normalization"
    )
    original = np.array(values)
    before = original.copy()
    actual = adc.normalize_populations(original)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(actual.sum(axis=-1), 1, rtol=0, atol=2e-16)
    np.testing.assert_array_equal(original, before)
    assert not np.shares_memory(actual, original)


def test_explicit_normalization_clips_negatives_without_a_lower_bound():
    values = np.array([[-6.2e-5, 0.2500155, 0.7500465], [-2.0, 0.75, 2.25]])
    before = values.copy()

    actual = adc.normalize_populations(values)

    np.testing.assert_allclose(actual, [[0, 0.25, 0.75], [0, 0.25, 0.75]], atol=1e-15)
    np.testing.assert_array_equal(values, before)
    assert actual.flags.owndata
    assert not np.shares_memory(actual, values)


@pytest.mark.parametrize(
    "values, message",
    [
        ([[0.0, np.nextafter(1 - 1e-6, 0)]], "trace.*1 row"),
        ([[0.0, np.nextafter(1 + 1e-6, 2)]], "trace.*1 row"),
        ([[0.2, 0.9], [0.1, 0.8], [0.5, 0.5]], "trace.*2 rows"),
        ([[np.nan, 1]], "finite"),
        ([[np.inf, 1]], "finite"),
        ([[], []], "state"),
        (1.0, "state"),
    ],
)
def test_explicit_normalization_rejects_out_of_contract_rows(values, message):
    assert callable(getattr(adc, "normalize_populations", None)), (
        "ADC must expose explicit probability normalization"
    )
    with pytest.raises(ValueError, match=message):
        adc.normalize_populations(np.array(values))
