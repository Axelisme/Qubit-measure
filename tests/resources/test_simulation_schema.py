"""Receiver background validation at the shared simulation-settings seam."""

import pytest
from pydantic import ValidationError
from zcu_tools.resources.simulation_schema import Receiver


def test_receiver_background_roundtrips_optional_line_structure():
    values = {
        "ro_ch": 0,
        "gain": 1.0,
        "noise_psd_adc2_ns": ((0.0, 0.0), (0.0, 0.0)),
        "background": {
            "extra_ripples": [{"depth": 0.04, "period_mhz": 44.0, "phase_deg": 10.0}],
            "roughness": {"amplitude": 0.02, "correlation_mhz": 50.0, "seed": 9},
            "band_edges": {
                "low": {"corner_ghz": 3.7, "rolloff_db_per_ghz": 30.0},
                "high": {"corner_ghz": 8.0, "rolloff_db_per_ghz": 20.0},
            },
        },
    }
    receiver = Receiver.model_validate(values)
    assert receiver.model_dump()["background"]["roughness"]["seed"] == 9
    assert Receiver.model_validate(receiver.model_dump()) == receiver


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


@pytest.mark.parametrize(
    ("background", "field"),
    [
        (
            {"extra_ripples": [{"depth": -0.01, "period_mhz": 44.0}]},
            "extra_ripples.0.depth",
        ),
        (
            {"extra_ripples": [{"depth": 1.0, "period_mhz": 44.0}]},
            "extra_ripples.0.depth",
        ),
        (
            {"extra_ripples": [{"depth": float("nan"), "period_mhz": 44.0}]},
            "extra_ripples.0.depth",
        ),
        (
            {"extra_ripples": [{"depth": 0.1, "period_mhz": 0.0}]},
            "extra_ripples.0.period_mhz",
        ),
        (
            {"extra_ripples": [{"depth": 0.1, "period_mhz": -44.0}]},
            "extra_ripples.0.period_mhz",
        ),
        (
            {"extra_ripples": [{"depth": 0.1, "period_mhz": float("inf")}]},
            "extra_ripples.0.period_mhz",
        ),
        (
            {
                "extra_ripples": [
                    {"depth": 0.1, "period_mhz": 44.0, "phase_deg": float("nan")}
                ]
            },
            "extra_ripples.0.phase_deg",
        ),
        (
            {"roughness": {"amplitude": -0.1, "correlation_mhz": 50.0}},
            "roughness.amplitude",
        ),
        (
            {"roughness": {"amplitude": float("inf"), "correlation_mhz": 50.0}},
            "roughness.amplitude",
        ),
        (
            {"roughness": {"amplitude": 0.02, "correlation_mhz": 0.0}},
            "roughness.correlation_mhz",
        ),
        (
            {"roughness": {"amplitude": 0.02, "correlation_mhz": float("nan")}},
            "roughness.correlation_mhz",
        ),
        (
            {"roughness": {"amplitude": 0.02, "correlation_mhz": 50.0, "seed": -1}},
            "roughness.seed",
        ),
        (
            {"roughness": {"amplitude": 0.02, "correlation_mhz": 50.0, "seed": 1.5}},
            "roughness.seed",
        ),
        (
            {
                "band_edges": {
                    "low": {"corner_ghz": float("nan"), "rolloff_db_per_ghz": 20.0}
                }
            },
            "band_edges.low.corner_ghz",
        ),
        (
            {"band_edges": {"high": {"corner_ghz": 8.0, "rolloff_db_per_ghz": -1.0}}},
            "band_edges.high.rolloff_db_per_ghz",
        ),
        (
            {"band_edges": {"high": {"corner_ghz": 8.0, "rolloff_db_per_ghz": 0.0}}},
            "band_edges.high.rolloff_db_per_ghz",
        ),
        (
            {
                "band_edges": {
                    "high": {"corner_ghz": 8.0, "rolloff_db_per_ghz": float("inf")}
                }
            },
            "band_edges.high.rolloff_db_per_ghz",
        ),
    ],
)
def test_receiver_schema_names_invalid_optional_background_fields(background, field):
    with pytest.raises(ValidationError) as error:
        Receiver.model_validate(
            {
                "ro_ch": 0,
                "gain": 1.0,
                "noise_psd_adc2_ns": ((0.0, 0.0), (0.0, 0.0)),
                "background": background,
            }
        )
    assert error.value.errors()[0]["loc"] == (
        "background",
        *(int(part) if part.isdigit() else part for part in field.split(".")),
    )


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
