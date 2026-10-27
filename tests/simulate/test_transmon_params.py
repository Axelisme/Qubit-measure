"""Named SQUID inputs preserve missing fields at the neutral boundary."""

import numpy as np
import pytest


def test_transmon_parameters_decode_partial_inputs_without_defaults() -> None:
    from zcu_tools.simulate.transmon import TransmonParams

    values = {"EJmax": 20.0, "d": 0.0}
    params = TransmonParams.from_mapping(values)
    assert params.missing() == ("EC",)
    assert params.to_dict() == {
        "EJmax": 20.0,
        "EC": None,
        "d": 0.0,
    }
    with pytest.raises(ValueError, match="Missing parameters: EC"):
        params.build_predictor()
    assert values == {"EJmax": 20.0, "d": 0.0}


def test_energy_only_model_queries_reduced_flux_without_affine() -> None:
    from zcu_tools.simulate.transmon import TransmonParams

    model = TransmonParams(20.0, 0.3, 0.0).build_predictor()
    fluxs, frequencies = model.predict_frequencies_mhz(
        np.array([0.5, 0.0, 1.5]), ((0, 1),)
    )
    assert fluxs == pytest.approx([0.5, 0.0, 1.5])
    assert frequencies[0] == pytest.approx([1200.0, 6613.448853504742, 1200.0])


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
