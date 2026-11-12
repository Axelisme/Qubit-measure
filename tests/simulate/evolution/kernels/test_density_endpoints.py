"""Caller-controlled endpoint accuracy through the shared numerical seam."""

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.kernel import SegmentTable
from zcu_tools.simulate.evolution.kernels._dop853 import Tolerances
from zcu_tools.simulate.evolution.kernels._dop853_data import pack_segments
from zcu_tools.simulate.evolution.kernels.dop853 import (
    DOP853Kernel,
    DOP853Plan,
    EndpointData,
    advance_endpoints,
    prepare_models,
)

from tests.simulate.evolution.contract.kernel_suite import KernelCase, synthetic_model
from tests.simulate.evolution.contract.reference import density_error, reference


@pytest.fixture
def endpoint_case() -> KernelCase:
    model = synthetic_model()
    vector = np.array([1.0, 0.4j, -0.3], dtype=complex)
    vector /= np.linalg.norm(vector)
    density = np.outer(vector, vector.conj())
    table = SegmentTable(
        timeline=np.array([0]),
        start_ns=np.array([0.0]),
        duration_ns=np.array([4.0]),
        level=np.array([[0]]),
        tones=np.array([1]),
        tone_port=np.array([[0]]),
        tone_frequency_ghz=np.array([[-0.37]]),
        tone_amplitude=np.array([[0.51]]),
        tone_theta_rad=np.array([[0.43]]),
        tone_envelope=np.array([[0]]),
        tone_envelope_offset_ns=np.array([[0.0]]),
        envelopes=(EnvelopeSpec("constant"),),
    )
    return KernelCase(
        DOP853Kernel(), ModelSource(density, lambda _: model), table, density[None]
    )


def test_tighter_controls_improve_full_density(endpoint_case: KernelCase) -> None:
    case = endpoint_case
    models = prepare_models([case.source.model((0,))]).models
    segments = pack_segments(case.table)
    plan = DOP853Plan(np.array(["interaction"]), np.array([-1]))
    expected = reference(case, 0, case.states[0])[None]
    errors = []
    for rtol, atol in ((1e-3, 1e-5), (2e-11, 2e-13)):
        actual = case.states.copy()
        advance_endpoints(
            EndpointData(models, segments),
            (np.array([0]), np.array([0])),
            actual,
            case.budget,
            plan,
            controls=Tolerances(rtol, atol, np.inf, np.inf, 0),
        )
        errors.append(density_error(actual, expected))
    print(f"Full-density errors: loose={errors[0]:.17g}, tight={errors[1]:.17g}")
    assert errors[1] < errors[0]
    assert errors[1] < 1e-9


@pytest.mark.parametrize("budget", [0.0, -1.0, np.inf, np.nan])
@pytest.mark.parametrize("empty", [False, True])
def test_explicit_controls_still_require_valid_budget(
    endpoint_case: KernelCase, budget: float, empty: bool
) -> None:
    case = endpoint_case
    rows = np.arange(0 if empty else 1, dtype=np.int64)
    with pytest.raises(ValueError, match="budget must be positive and finite"):
        advance_endpoints(
            EndpointData(
                prepare_models([case.source.model((0,))]).models,
                pack_segments(case.table),
            ),
            (rows, np.zeros(len(rows), dtype=np.int64)),
            case.states[rows].copy(),
            budget,
            DOP853Plan(np.array(["interaction"]), np.array([-1])),
            controls=Tolerances(2e-11, 2e-13, np.inf, np.inf, 0),
        )
