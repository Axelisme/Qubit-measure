"""Receiver background validation at the shared simulation-settings seam."""

import pytest
from pydantic import ValidationError
from zcu_tools.resources.simulation_schema import Receiver


def test_receiver_omitted_background_matches_explicit_flat_background():
    values = {"ro_ch": 0, "gain": 1.0, "noise_psd_adc2_ns": ((0.0, 0.0), (0.0, 0.0))}
    omitted = Receiver.model_validate(values)
    assert hasattr(omitted, "background"), "schema must accept optional background"
    explicit = Receiver.model_validate({**values, "background": {}})
    assert omitted == explicit
    assert omitted.background.ripple_depth == 0.0
    assert omitted.background.slope_db_per_ghz == 0.0
    configured = Receiver.model_validate(
        {
            **values,
            "background": {
                "ripple_depth": 0.1,
                "ripple_period_mhz": 30.0,
                "ripple_phase_deg": 45.0,
                "slope_db_per_ghz": -3.0,
                "pivot_ghz": 7.0,
            },
        }
    )
    assert configured.background.ripple_period_mhz == 30.0
    assert Receiver.model_validate(configured.model_dump()) == configured


@pytest.mark.parametrize("depth", [-0.1, 1.0, float("nan")])
def test_receiver_schema_rejects_out_of_range_background_depth(depth):
    with pytest.raises(ValidationError, match=r"background\.ripple_depth"):
        Receiver.model_validate(
            {
                "ro_ch": 0,
                "gain": 1.0,
                "noise_psd_adc2_ns": ((0.0, 0.0), (0.0, 0.0)),
                "background": {"ripple_depth": depth},
            }
        )
