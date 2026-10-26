"""Author-selected device coordinates from record tables and their dotted CSVs."""

import numpy as np
import pandas as pd
import pytest
from zcu_tools.resources.record_table import RecordFluxFrame, flux_from_frame


@pytest.mark.parametrize("flatten", [False, True])
def test_flux_uses_explicit_then_row_frame_then_named_fallback(flatten: bool) -> None:
    rows = [
        {"flux": 0.8, "flux_int": "unused", "devices": {"bias": {"value": "unused"}}},
        {
            "devices": {
                "bias": {"value": 3, "value_unit": "arbitrary"},
                "other": {"value": 99},
            },
            "flux_int": 1,
            "flux_period": 4,
        },
        {"devices": {"bias": {"value": 4, "value_unit": "V"}}},
    ]
    table = pd.json_normalize(rows) if flatten else pd.DataFrame(rows)
    table.index = [8, 2, 5]
    original = table.copy(deep=True)

    flux = flux_from_frame(
        table, device_name="bias", fallback_frame=RecordFluxFrame("V", 2, 8)
    )

    np.testing.assert_allclose(flux, [0.8, 0.5, 0.25])
    pd.testing.assert_frame_equal(table, original)


def test_flux_row_frame_accepts_numeric_dotted_columns_without_units() -> None:
    table = pd.DataFrame(
        {"devices.bias.value": [3, 5], "flux_int": [1, 1], "flux_period": [4, 4]}
    )

    np.testing.assert_allclose(flux_from_frame(table, device_name="bias"), [0.5, 1.0])


@pytest.mark.parametrize(
    ("present", "missing"),
    [("flux_int", "flux_period"), ("flux_period", "flux_int")],
)
@pytest.mark.parametrize("null_column", [False, True])
def test_incomplete_used_row_frame_fails_without_fallback(
    present: str, missing: str, null_column: bool
) -> None:
    table = pd.DataFrame(
        {
            "flux": [0.8, np.nan],
            "devices.bias.value": [np.nan, 3.0],
            "devices.bias.value_unit": [None, "V"],
            present: [np.nan, 1.0],
        },
        index=["explicit", "partial"],
    )
    if null_column:
        table[missing] = np.nan

    with pytest.raises(ValueError, match=f"partial.*{missing}|{missing}.*partial"):
        flux_from_frame(
            table, device_name="bias", fallback_frame=RecordFluxFrame("V", 1, 4)
        )
