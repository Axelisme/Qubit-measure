"""Exact-input fractional tails share work without sharing carrier phase."""

import cProfile
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource, SecularDrive
from zcu_tools.simulate.evolution.kernel import (
    ApplyContext,
    KernelContext,
    SegmentTable,
)
from zcu_tools.simulate.evolution.kernels.shaped_sambe import ShapedSambeKernel
from zcu_tools.simulate.evolution.workplan import WorkPlan

from ..contract.kernel_suite import synthetic_model


@pytest.mark.parametrize(
    "changed", ["carrier", "amplitude", "duration", "port", "model", "level"]
)
def test_fractional_tail_reuses_exact_inputs_but_not_phase(changed: str) -> None:
    model = synthetic_model()
    drive = model.drives[0]
    model = replace(
        model,
        drives=(
            drive,
            SecularDrive("other", drive.in_phase_ghz / 2, drive.quadrature_ghz / 2),
        ),
    )
    initial = np.ones((3, 3), dtype=np.complex128) / 3
    source = ModelSource(initial, lambda _: model)
    table = SegmentTable(
        timeline=np.arange(2),
        start_ns=np.zeros(2),
        duration_ns=np.full(2, 2.7),
        level=np.zeros((2, 1), dtype=np.int64),
        tones=np.ones(2, dtype=np.int64),
        tone_port=np.zeros((2, 1), dtype=np.int64),
        tone_frequency_ghz=np.full((2, 1), 0.22),
        tone_amplitude=np.full((2, 1), 0.01),
        tone_theta_rad=np.array([[0.17], [1.3]]),
        tone_envelope=np.zeros((2, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((2, 1)),
        envelopes=(
            EnvelopeSpec(
                "sampled", np.array([0.3, 0.8], dtype=complex), np.array([0.0, 4.0])
            ),
        ),
    )
    solver = ShapedSambeKernel()
    plan = WorkPlan()

    def prepare(current_table, current_source):
        context = KernelContext(current_source, 1e-3, plan)
        keys = solver.prepare_keys(current_table, np.array([0]), context)
        return solver.prepare(keys, current_table, np.array([0]), context)

    def apply(prepared, current_table, work_plan):
        states = np.repeat(initial[None], 2, axis=0)
        solver.apply(
            prepared,
            np.zeros(2, dtype=np.int64),
            current_table,
            np.arange(2),
            states,
            ApplyContext(1e-3, work_plan=work_plan),
        )
        return states

    prepared = prepare(table, source)
    # WorkPlan has no public fact inventory; profile observes builds without
    # patching or calling a private numerical helper.
    profile = cProfile.Profile()
    with profile:
        actual = apply(prepared, table, plan)
        # Representative tables may renumber rows in the same evolve.
        repeated = apply(prepared, replace(table, timeline=np.array([7, 8])), plan)
    builds = sum(
        entry.callcount
        for entry in profile.getstats()
        if not isinstance(entry.code, str) and entry.code.co_name == "_spectral"
    )
    assert builds == 1, (
        "equal physical tails must build Fourier blocks only once per WorkPlan"
    )
    np.testing.assert_array_equal(actual, repeated)
    assert not np.allclose(actual[0], actual[1], atol=1e-8, rtol=0)

    if changed == "carrier":
        other_table = replace(table, tone_frequency_ghz=-table.tone_frequency_ghz)
    elif changed == "amplitude":
        other_table = replace(table, tone_amplitude=table.tone_amplitude * 2)
    elif changed == "duration":
        other_table = replace(table, duration_ns=table.duration_ns + 0.1)
    elif changed == "port":
        other_table = replace(table, tone_port=np.ones((2, 1), dtype=np.int64))
    else:
        other_table = table
        other_model = replace(
            model,
            identity="other-model",
            rates_per_ns=model.rates_per_ns * 2,
            energies_ghz=model.energies_ghz * 1.1,
        )
        if changed == "level":
            other_model = replace(other_model, identity=model.identity, level=(1,))
            other_table = replace(table, level=np.ones((2, 1), dtype=np.int64))
        source = ModelSource(initial, lambda _: other_model)
    other_prepared = prepare(other_table, source)
    shared = apply(other_prepared, other_table, plan)
    isolated_plan = WorkPlan()
    try:
        isolated = apply(other_prepared, other_table, isolated_plan)
        np.testing.assert_array_equal(shared, isolated)
        assert not np.allclose(actual, shared, atol=1e-8, rtol=0)
    finally:
        isolated_plan.close()
        plan.close()
