"""Period playback regressions against the original constant-tone builders."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.kernel import (
    ApplyContext,
    KernelContext,
    SegmentTable,
)
from zcu_tools.simulate.evolution.kernels.floquet_period import FloquetPeriodKernel
from zcu_tools.simulate.evolution.workplan import WorkPlan

from ..contract.kernel_suite import synthetic_model


def playback(frequency: float, duration: float, phase: float) -> NDArray[np.complex128]:
    """Play one coherent three-level row through the kernel contract."""
    model = synthetic_model(3)
    model = replace(
        model,
        energies_ghz=np.array([0.0, 0.05, 8.0]),
        rates_per_ns=model.rates_per_ns * 1e-5,
        dephasing_per_ns=model.dephasing_per_ns * 1e-5,
    )
    psi = np.array([1.0, 1j, -1.0], dtype=np.complex128) / np.sqrt(3)
    state = np.outer(psi, psi.conj())[None]
    source = ModelSource(state[0], lambda level: model)
    table = SegmentTable(
        timeline=np.array([0]),
        start_ns=np.array([7.3]),
        duration_ns=np.array([duration]),
        level=np.zeros((1, 1), dtype=np.int64),
        tones=np.ones(1, dtype=np.int64),
        tone_port=np.zeros((1, 1), dtype=np.int64),
        tone_frequency_ghz=np.array([[frequency]]),
        tone_amplitude=np.array([[0.002]]),
        tone_theta_rad=np.array([[phase]]),
        tone_envelope=np.zeros((1, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((1, 1)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    kernel = FloquetPeriodKernel()
    rows = np.array([0], dtype=np.int64)
    plan = WorkPlan()
    try:
        context = KernelContext(source, 1e-3, plan)
        assert kernel.guard(table, rows, context).all()
        keys = kernel.prepare_keys(table, rows, context)
        prepared = kernel.prepare(keys, table, rows, context)
        exceeded = kernel.apply(prepared, rows, table, rows, state, ApplyContext(1e-3))
        assert not exceeded.any()
    finally:
        plan.close()
    return state[0]


# Frozen full densities from the original Sambe builder, not a strict solver.
@pytest.mark.parametrize(
    "frequency,duration,phase,expected",
    [
        (
            0.05,
            150.0,
            0.37,
            [
                [
                    (0.1506626646278542 + 7.806252322459159e-24j),
                    (0.20329347608462323 - 0.19089094817550778j),
                    (-0.19241965740038658 - 0.11473718762938805j),
                ],
                [
                    (0.20329347608462334 + 0.19089094817550778j),
                    (0.5161954791982539 + 1.0408333044865135e-17j),
                    (-0.11426874589221242 - 0.39863149564087996j),
                ],
                [
                    (-0.19241965740038663 + 0.11473718762938806j),
                    (-0.1142687458922124 + 0.39863149564087985j),
                    (0.33314185617389064 - 1.3552522412685132e-20j),
                ],
            ],
        ),
        (
            -0.05,
            300.0,
            5.9,
            [
                [
                    (0.3351341881830036 - 1.5178840389448516e-18j),
                    (0.04970284887367334 - 0.3295869939206092j),
                    (-0.33328799505768497 - 0.025018856055245946j),
                ],
                [
                    (0.049702848873673384 + 0.32958699392060936j),
                    (0.3315276269157145 - 8.673607334081838e-19j),
                    (-0.02482554165058058 - 0.3314984321748862j),
                ],
                [
                    (-0.3332879950576849 + 0.025018856055245946j),
                    (-0.024825541650580568 + 0.3314984321748862j),
                    (0.3333381849012776 + 1.0164388251977337e-20j),
                ],
            ],
        ),
        (
            0.05,
            10000.0,
            0.37,
            [
                [
                    (0.3334480946431159 - 1.4571659451999098e-16j),
                    (-4.651901216135778e-05 - 0.3329223841421223j),
                    (-0.33265384009637317 - 0.015884737041897724j),
                ],
                [
                    (-4.651901216168218e-05 + 0.33292238414212105j),
                    (0.33345940242289646 + 9.887906073502456e-17j),
                    (0.01588624137787255 - 0.33265948415543906j),
                ],
                [
                    (-0.332653840096373 + 0.01588473704189751j),
                    (0.015886241377872576 + 0.332659484155439j),
                    (0.3330925029339039 + 9.486762368512181e-20j),
                ],
            ],
        ),
        (
            -0.05,
            10000.0,
            -2.41,
            [
                [
                    (0.33686565751003683 - 4.9005838623301954e-17j),
                    (-0.31990366615081983 - 0.09228341730091916j),
                    (0.280987274666549 - 0.18188626648169065j),
                ],
                [
                    (-0.3199036661508176 + 0.09228341730091925j),
                    (0.3299203632451881 + 5.984786029597979e-17j),
                    (-0.2173858067507277 + 0.25013780242502787j),
                ],
                [
                    (0.28098727466655016 + 0.18188626648169068j),
                    (-0.21738580675072794 - 0.25013780242502687j),
                    (0.3332139792446865 - 7.453884718116714e-20j),
                ],
            ],
        ),
    ],
)
def test_sambe_playback_preserves_density(frequency, duration, phase, expected):
    np.testing.assert_allclose(
        playback(frequency, duration, phase), expected, atol=1e-10, rtol=0
    )
