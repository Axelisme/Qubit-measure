"""Reconstruction and immutability contracts of public fitter outputs."""

import copy
import pickle
from operator import methodcaller

import numpy as np
import pytest
from zcu_tools.analysis.fitting import (
    fit_decay,
    fit_decay_fringe,
    fit_dual_decay,
    fit_gauss_decay,
    fit_gauss_decay_fringe,
    fit_rabi,
)
from zcu_tools.analysis.fitting.results import (
    DecayFit,
    DecayFringeFit,
    DualDecayFit,
    GaussDecayFit,
    GaussDecayFringeFit,
    RabiFit,
)

from tests.analysis.fitting._fit_contract import assert_frozen_fields

type NamedFit = (
    DecayFit | DecayFringeFit | DualDecayFit | GaussDecayFit | GaussDecayFringeFit | RabiFit
)


@pytest.fixture(params=("decay", "dual", "fringe", "gauss", "gauss_fringe", "rabi", "rabi_decay"))
def result(request: pytest.FixtureRequest) -> NamedFit:
    """Fit synthetic observations through each supported public model."""
    xs = np.linspace(0.0, 20.0, 120)
    envelope = np.exp(-xs / 5.0)
    oscillation = np.cos(2 * np.pi * 0.5 * xs)
    if request.param == "decay":
        return fit_decay(xs, 0.2 + 0.8 * envelope)
    if request.param == "dual":
        return fit_dual_decay(
            xs,
            0.2 + 0.3 * np.exp(-xs / 2.0) + 0.6 * envelope,
            fit_params=(0.2, 0.3, 2.0, 0.6, 5.0),
        )
    if request.param == "fringe":
        return fit_decay_fringe(xs, 0.2 + 0.8 * envelope * oscillation)
    if request.param == "gauss":
        return fit_gauss_decay(xs, 0.2 + 0.8 * np.exp(-0.5 * ((xs - 2.0) / 3.0) ** 2))
    if request.param == "gauss_fringe":
        return fit_gauss_decay_fringe(xs, 0.2 + 0.8 * np.exp(-((xs / 5.0) ** 2)) * oscillation)
    decay = request.param == "rabi_decay"
    return fit_rabi(
        xs, 0.2 + 0.8 * oscillation * (envelope if decay else 1), decay=decay
    )


@pytest.mark.parametrize("operation", ("copy", "deepcopy", "pickle"))
def test_reconstruction_preserves_all_fields_and_quality(
    result: NamedFit, operation: str
) -> None:
    if operation == "copy":
        restored = copy.copy(result)
    elif operation == "deepcopy":
        restored = copy.deepcopy(result)
    else:
        restored = pickle.loads(pickle.dumps(result))

    assert type(restored) is type(result)
    assert len(restored) == len(result)
    np.testing.assert_array_equal(restored[:-2], result[:-2])
    np.testing.assert_array_equal(restored.curve, result.curve)
    np.testing.assert_array_equal(restored.params, result.params)
    np.testing.assert_array_equal(restored.cov, result.cov)
    assert restored.quality == result.quality
    assert_frozen_fields(restored, ("curve", "params", "cov", "quality"))


@pytest.mark.parametrize("operation", ("make", "replace"))
def test_changed_values_require_refitting_for_consistent_quality(
    result: NamedFit, operation: str
) -> None:
    # NamedTuple documents these underscore-prefixed methods as public API.
    invoke = (
        methodcaller("_make", tuple(result))
        if operation == "make"
        else methodcaller("_replace", curve=result.curve + 1.0)
    )
    with pytest.raises(TypeError, match="rerun.*fitter.*quality"):
        invoke(result)
