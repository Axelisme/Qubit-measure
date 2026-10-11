"""Named simulator parameters, independent of a component or session."""

import dataclasses

import pytest
from zcu_tools.simulate.fluxonium import predict


def test_partial_params_report_missing_and_refuse_model() -> None:
    params = predict.FluxoniumParams(EJ=4.0, EC=1.0, flux_bias=0.0)
    assert params.missing() == ("EL", "flux_half", "flux_period")
    with pytest.raises(ValueError, match="EL.*flux_half.*flux_period"):
        params.build_predictor()


def test_complete_params_build_independent_mhz_model() -> None:
    params = predict.FluxoniumParams(5.0, 1.0, 0.5, 0.3, 0.8, 0.02)
    assert params.missing() == ()
    model = params.build_predictor()
    assert model.value_to_flux(0.38) == pytest.approx(0.625)
    assert model.predict_freq(0.36) == pytest.approx(1707.3748654723192, abs=1e-7)
    model.update_bias(0.1)
    assert params.flux_bias == 0.02
    assert params.build_predictor().flux_bias == 0.02


@pytest.mark.parametrize(
    "field,value",
    [
        ("EJ", -1.0),
        ("EC", 0.0),
        ("EL", float("nan")),
        ("flux_period", 0.0),
        ("flux_bias", float("inf")),
    ],
)
def test_complete_params_reject_invalid_physics(field: str, value: float) -> None:
    params = dataclasses.replace(
        predict.FluxoniumParams(4.0, 1.0, 0.5, 0.3, 0.8, 0.0), **{field: value}
    )
    with pytest.raises(ValueError, match=field):
        params.build_predictor()
