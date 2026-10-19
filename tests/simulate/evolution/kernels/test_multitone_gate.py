"""Covariant simultaneous gate products through Engine and the C5 seam."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
import scipy.integrate
from zcu_tools.simulate.evolution import (
    DriveTrack,
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    SecularDrive,
    SecularModel,
    TimelineBatch,
)
from zcu_tools.simulate.evolution.kernel import KernelContext, SegmentTable
from zcu_tools.simulate.evolution.kernels import REGISTRY
from zcu_tools.simulate.evolution.workplan import WorkPlan

from tests.simulate.evolution.contract.kernel_suite import KernelCase, KernelContract


def source(conflict=0.0):
    a, b = np.zeros((4, 4), complex), np.zeros((4, 4), complex)
    a[0, 2] = a[1, 3] = 0.002
    b[0, 1] = b[2, 3] = 0.002
    b[1, 2] = conflict
    model = SecularModel(
        (0,),
        f"joint-gate-{conflict}",
        np.array([0.0, 0.3, 0.6, 0.9]),
        np.eye(4, dtype=complex),
        (
            SecularDrive("a", a + a.T, np.zeros_like(a)),
            SecularDrive("b", b + b.T, np.zeros_like(b)),
        ),
        np.array([[0, 1e-5, 2e-5, 0], [0, 0, 0, 2e-5], [0, 0, 0, 1e-5], [0, 0, 0, 0]]),
        (np.ones((4, 4)) - np.eye(4)) * 1e-5,
    )
    v = np.array([1, 1j, -0.5, 0.3j], complex)
    v /= np.linalg.norm(v)
    return ModelSource(np.outer(v, v.conj()), lambda level: replace(model, level=level))


def envelope():
    t = np.linspace(0, 128, 129)
    return EnvelopeSpec(
        "sampled", np.exp(-0.5 * ((t - 64) / 32) ** 2).astype(complex), t
    )


def batch(phase=0.0, count=24):
    tracks = tuple(
        DriveTrack(
            port,
            np.arange(count, dtype=float) * 3.125,
            np.full(count, 128.0),
            np.full(count, frequency),
            np.linspace(-1, 2, count) + phase * (tone + 1),
            np.full(count, 0.08),
            (envelope(),),
            np.zeros(count, dtype=np.int64),
            np.ones(count, dtype=bool),
        )
        for tone, (port, frequency) in enumerate((("a", 0.6), ("b", 0.3)))
    )
    return TimelineBatch(
        np.arange(count),
        tracks,
        FluxSchedule(
            np.arange(count + 1), np.zeros(count), np.zeros((count, 1), dtype=np.int64)
        ),
        tracks[0].start_ns + 128.0,
    )


def family():
    found = next((k for k in REGISTRY if k.name == "multitone_gate"), None)
    assert found is not None, "repeated covariant joint gates need a cached family"
    return found


@pytest.mark.parametrize("conflict", [0.0, 1e-9])
def test_repeated_pair_builds_once_and_reuses_across_phase_changes(
    monkeypatch, conflict
):
    calls = []
    original = scipy.integrate.solve_ivp

    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(scipy.integrate, "solve_ivp", observed)
    with closing(Engine(1, 1_000_000)) as engine:
        for phase in (0.0, 0.713):
            result = engine.evolve(source(conflict), (batch(phase),), cancel=Event())
            assert result.paths.segments_by_kernel.get("multitone_gate") == 24
            with closing(
                Engine(
                    1,
                    0,
                    kernels=tuple(k for k in REGISTRY if k.name != "multitone_gate"),
                )
            ) as direct:
                reference = direct.evolve(
                    source(conflict), (batch(phase),), cancel=Event()
                )
            assert (
                np.max(
                    np.abs(
                        np.diagonal(result.densities, axis1=-2, axis2=-1)
                        - np.diagonal(reference.densities, axis1=-2, axis2=-1)
                    )
                )
                < 1e-2
            )
    assert len(calls) == 4, (
        "one canonical pair builds four Hilbert subsegments only once"
    )


@pytest.mark.parametrize("conflict,count", [(0.0004, 24), (0.0, 1)])
def test_inconsistent_or_unamortized_pairs_keep_row_rwa(conflict, count):
    family()
    with closing(Engine(1, 1_000_000)) as engine:
        result = engine.evolve(source(conflict), (batch(count=count),), cancel=Event())
    assert result.paths.segments_by_kernel.get("multitone_rwa") == count
    assert "multitone_gate" not in result.paths.segments_by_kernel


@pytest.fixture
def kernel_case():
    count = 24
    table = SegmentTable(
        timeline=np.arange(count),
        start_ns=np.arange(count, dtype=np.float64) * 3.125,
        duration_ns=np.full(count, 8.0),
        level=np.zeros((count, 1), dtype=np.int64),
        tones=np.full(count, 2),
        tone_port=np.tile([0, 1], (count, 1)),
        tone_frequency_ghz=np.tile([0.6, 0.3], (count, 1)),
        tone_amplitude=np.full((count, 2), 0.08),
        tone_theta_rad=np.arange(count * 2, dtype=np.float64).reshape(count, 2) * 0.13,
        tone_envelope=np.zeros((count, 2), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((count, 2)),
        envelopes=(
            EnvelopeSpec(
                "sampled", np.ones(3, dtype=complex), np.array([0.0, 4.0, 8.0])
            ),
        ),
    )
    src = source()
    return KernelCase(
        family(), src, table, np.repeat(src.initial_density[None], count, axis=0)
    )


class TestMultitoneGate(KernelContract):
    pass


def test_keys_remove_only_covariant_phase_and_start(kernel_case):
    case = kernel_case
    rows = np.arange(24)
    context = KernelContext(case.source, case.budget, WorkPlan())
    keys = case.kernel.prepare_keys(case.table, rows, context)
    assert len(np.unique(keys)) == 1
    for field in (
        "tone_amplitude",
        "tone_frequency_ghz",
        "tone_envelope_offset_ns",
        "duration_ns",
    ):
        values = getattr(case.table, field).copy()
        values[0] += 0.001
        changed = case.kernel.prepare_keys(
            replace(case.table, **{field: values}), rows, context
        )
        assert changed[0] != keys[0]
