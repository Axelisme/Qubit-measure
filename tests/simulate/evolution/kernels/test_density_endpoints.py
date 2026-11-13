"""Budget validation through the shared density endpoint seam."""

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.kernel import SegmentTable
from zcu_tools.simulate.evolution.kernels._dop853_data import pack_segments
from zcu_tools.simulate.evolution.kernels.dop853 import (
    DOP853Kernel,
    DOP853Plan,
    EndpointData,
    advance_endpoints,
    prepare_models,
)

from tests.simulate.evolution.contract.kernel_suite import KernelCase, synthetic_model


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


@pytest.mark.parametrize("budget", [0.0, -1.0, np.inf, np.nan])
@pytest.mark.parametrize("empty", [False, True])
def test_endpoints_require_valid_budget(
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
        )
