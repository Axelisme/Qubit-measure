"""Cost-selected constant-tone preparation through the scoped C5 interface."""

from dataclasses import replace

import numpy as np
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.kernel import SegmentTable
from zcu_tools.simulate.evolution.kernels.floquet_period import FloquetPeriodKernel

from ..contract.kernel_suite import KernelCase, reference, synthetic_model


def sambe_case():
    model = synthetic_model()
    model = replace(
        model,
        energies_ghz=np.array([0.0, 4.0, 8.0]),
        rates_per_ns=np.zeros((3, 3)),
        dephasing_per_ns=np.zeros((3, 3)),
    )
    source = ModelSource(np.ones((3, 3), dtype=complex) / 3, lambda level: model)
    table = SegmentTable(
        timeline=np.arange(2),
        start_ns=np.array([0.0, 13.0]),
        duration_ns=np.array([0.73, 21.7]),
        level=np.zeros((2, 1), dtype=np.int64),
        tones=np.ones(2, dtype=np.int64),
        tone_port=np.zeros((2, 1), dtype=np.int64),
        tone_frequency_ghz=np.array([[0.05], [-0.05]]),
        tone_amplitude=np.array([[0.01], [-0.02]]),
        tone_theta_rad=np.array([[1.73], [-5.9]]),
        tone_envelope=np.zeros((2, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((2, 1)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    states = np.repeat(source.initial_density[None], 2, axis=0)
    return KernelCase(FloquetPeriodKernel(), source, table, states)


def test_low_carrier_uses_sambe_and_preserves_translated_coherent_tail():
    case = sambe_case()
    kernel, workspace = case.kernel.evolve_scope(case.table, case.source)
    rows = np.arange(2)
    prepared = kernel.prepare(
        kernel.prepare_keys(case.table, rows), case.table, rows, case.source, case.budget
    )
    assert getattr(workspace, "sambe_builds", 0) == 2
    assert workspace.attempts == workspace.hits == workspace.nbytes == 0
    actual = case.states.copy()
    kernel.apply(prepared, rows, case.table, rows, actual, case.budget)
    for row in rows:
        np.testing.assert_allclose(
            actual[row], reference(case, row, case.states[row]), atol=1e-4, rtol=0
        )
    assert prepared.nbytes > 0
