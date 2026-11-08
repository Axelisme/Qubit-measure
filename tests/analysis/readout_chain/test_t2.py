"""Public ideal-pulse T2 curves, including the third-state lifetime."""

import numpy as np
import pytest
from zcu_tools.analysis.readout_chain import t2_echo_model, t2_ramsey_model

from ._t2_samples import bloch_curves, t2_nuisance


def test_echo_separates_transverse_and_third_state_decay() -> None:
    model = t2_echo_model()

    values = model.curve(np.log(2) * np.array([0.0, 1.0, 2.0]), {"t2": 1.0, "tf": 0.5})

    np.testing.assert_allclose(
        values, [[0.0, 1.0, 1.0], [0.25, 0.75, 0.25], [0.375, 0.625, 0.0625]]
    )


def test_ramsey_detuning_is_cycles_per_delay_unit() -> None:
    model = t2_ramsey_model()
    parameters = {"t2": 1 / np.log(2), "detuning": 0.5, "tf": 0.5 / np.log(2)}

    values = model.curve(np.array([0.0, 0.5, 1.0, 2.0]), parameters)

    np.testing.assert_allclose(
        values, [[0, 1, 1], [0.5, 0.5, 0.5], [0.75, 0.25, 0.25], [0.375, 0.625, 0.0625]]
    )
    assert model.oscillation_frequency == "detuning"


@pytest.mark.parametrize("factory", [t2_echo_model, t2_ramsey_model])
def test_ideal_analyzer_removes_longitudinal_relaxation(factory) -> None:
    model = factory()
    parameters = {"t2": 1.2, "tf": 0.65}
    if model.oscillation_frequency:
        parameters["detuning"] = 0.8
    axis = np.linspace(0.0, 3.0, 13)
    expected = model.curve(axis, parameters)
    for t1 in [0.6, 2.4, 24.0]:
        physical = bloch_curves(
            axis, parameters, t1=t1, echo=not model.oscillation_frequency
        )
        np.testing.assert_allclose(physical, expected, atol=1e-14)
    moments = model.moments(axis, parameters, t2_nuisance(1.8))
    assert np.isfinite(moments.jacobian).all()
    assert moments.jacobian.shape == (13, 65, len(moments.parameter_names))


@pytest.mark.parametrize("factory", [t2_echo_model, t2_ramsey_model])
def test_delay_domain_rejects_negative_and_nonfinite_values(factory) -> None:
    model = factory()
    parameters = {"t2": 1.2, "tf": 0.65, "detuning": 0.8}
    for axis in [np.array([-0.1]), np.array([np.inf]), np.array([np.nan])]:
        with pytest.raises(ValueError, match="delay"):
            model.curve(axis, parameters)
