"""Named SQUID inputs preserve missing fields at the neutral boundary."""

import pytest


def test_transmon_parameters_decode_partial_inputs_without_defaults() -> None:
    from zcu_tools.simulate.transmon import TransmonParams

    values = {"EJmax": 20.0, "d": 0.0}
    params = TransmonParams.from_mapping(values)
    assert params.missing() == ("EC", "flux_half", "flux_period", "flux_bias")
    assert params.to_dict() == {
        "EJmax": 20.0,
        "EC": None,
        "d": 0.0,
        "flux_half": None,
        "flux_period": None,
        "flux_bias": None,
    }
    with pytest.raises(ValueError, match="Missing parameters: EC"):
        params.build_predictor()
    assert values == {"EJmax": 20.0, "d": 0.0}


@pytest.mark.parametrize(
    "values, field",
    [
        ({"EL": 1.0}, "EL"),
        ({"EJmax": True}, "EJmax"),
        ({"d": float("inf")}, "d"),
        ({"EC": "0.3"}, "EC"),
    ],
)
def test_transmon_mapping_rejects_malformed_supplied_values(values, field) -> None:
    from zcu_tools.simulate.transmon import TransmonParams

    with pytest.raises(ValueError, match=field):
        TransmonParams.from_mapping(values)
