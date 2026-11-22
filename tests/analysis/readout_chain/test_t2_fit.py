"""Shot-simulated T2 fits with physical longitudinal relaxation in the delay."""

import numpy as np
import pytest
from zcu_tools.analysis.readout_chain import (
    CycleFit,
    CycleModel,
    CycleNuisances,
    CycleSums,
    fit_cycle,
    t2_echo_model,
    t2_ramsey_model,
)

from ._exact_moments import curve_moments
from ._t2_samples import sampled_t2_sums, t2_nuisance


@pytest.mark.parametrize("factory", [t2_echo_model, t2_ramsey_model])
def test_exact_moments_recover_t2_and_joint_predictions(factory) -> None:
    model = factory()
    truth = {"t2": 1.2, "tf": 0.65}
    init = {"t2": 1.1, "tf": 0.7}
    if model.oscillation_frequency:
        truth["detuning"] = 0.8
        init["detuning"] = 0.78
    axis = np.linspace(0.05, 3.0, 13)
    moments = curve_moments(model, axis, truth, t2_nuisance(1.8))
    sums = CycleSums(
        axis, np.full(len(axis), 2000.0), 2000 * moments.means, np.zeros((65, 65))
    )

    result = fit_cycle(sums, model, init)

    for name, value in truth.items():
        assert result.parameters[name] == pytest.approx(value, rel=1e-5)
        assert result.errors[name] > 0
    assert isinstance(result.nuisances, CycleNuisances)
    predicted = curve_moments(model, axis, result.parameters, result.nuisances)
    np.testing.assert_allclose(predicted.means, moments.means, atol=1e-6)
    assert result.J < 1e-7
    # Ramsey oscillates, so it reports a reliability; the echo decay does not.
    if factory is t2_ramsey_model:
        assert result.reliability is not None
    else:
        assert result.reliability is None


def shot_fit(model: CycleModel, contrast: float, count: int, seed: int) -> CycleFit:
    truth = {"t2": 1.2, "tf": 0.65}
    init = {"t2": 1.1, "tf": 0.7}
    if model.oscillation_frequency:
        truth["detuning"] = 0.8
        init["detuning"] = 0.78
    sums = sampled_t2_sums(
        model, np.linspace(0.05, 3.0, 13), truth, contrast, count, seed
    )

    return fit_cycle(sums, model, init)


@pytest.mark.slow
@pytest.mark.parametrize("factory", [t2_echo_model, t2_ramsey_model])
@pytest.mark.parametrize("contrast,count", [(1.8, 2000), (0.7, 20000)])
@pytest.mark.parametrize("seed", [9101, 9102])
def test_shot_fits_resolve_t2_and_detuning_with_longitudinal_relaxation(
    factory, contrast: float, count: int, seed: int
) -> None:
    model = factory()
    truth = {"t2": 1.2, "tf": 0.65, "detuning": 0.8}
    result = shot_fit(model, contrast, count, seed)

    # A few fixed replicates guard recovery/scale, not 95% calibration. The task
    # evidence uses more replicates and reports every miss rather than seed-picking.
    for name in ("t2", "detuning") if model.oscillation_frequency else ("t2",):
        assert result.errors[name] > 0
        assert abs(result.parameters[name] - truth[name]) < 4 * result.errors[name]
    if model.oscillation_frequency:
        assert result.reliability == pytest.approx(
            1 / (2 * np.pi * (3.0 - 0.05) * result.errors["detuning"])
        )
    else:
        assert result.reliability is None
