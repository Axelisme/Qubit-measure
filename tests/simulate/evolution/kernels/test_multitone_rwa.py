"""Joint-basis multi-tone approximation at the Engine and C5 seams."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
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


def model_source() -> ModelSource:
    x = np.array([[0, 0.002, 0], [0.002, 0, 0.001], [0, 0.001, 0]], complex)
    model = SecularModel(
        (0,),
        "joint-three",
        np.array([0.0, 0.3, 0.9]),
        np.eye(3, dtype=complex),
        (
            SecularDrive("a", x, (x * 0.2).astype(np.complex128)),
            SecularDrive(
                "b", (x * 0.7).astype(np.complex128), (-x * 0.3).astype(np.complex128)
            ),
        ),
        np.array([[0, 0.0002, 0], [0, 0, 0.0001], [0, 0, 0]]),
        np.ones((3, 3)) * 0.0001 - np.eye(3) * 0.0001,
    )
    vector = np.array([1, 1j, -1], complex) / np.sqrt(3)
    return ModelSource(
        np.outer(vector, vector.conj()), lambda level: replace(model, level=level)
    )


def smooth() -> EnvelopeSpec:
    times = np.linspace(0, 128, 129)
    return EnvelopeSpec(
        "sampled", np.exp(-(((times - 64) / 32) ** 2) / 2).astype(complex), times
    )


def timeline(kind: str = "smooth", tones: int = 2) -> TimelineBatch:
    env = smooth()
    if kind == "complex":
        env = replace(env, samples=env.samples * (1 + 0.1j))
    if kind == "rough":
        env = replace(
            env, samples=np.where(np.arange(129) % 2, 1.0, 0.0).astype(complex)
        )
    frequencies = [0.3, -0.6, 0.9][:tones]
    if kind == "degenerate":
        frequencies[1] = -0.30001
    if kind == "dc":
        frequencies[1] = 0.0
    tracks = tuple(
        DriveTrack(
            "a" if slot == 0 else "b",
            np.array([7.0]),
            np.array([128.0]),
            np.array([f]),
            np.array([0.4 + slot]),
            np.array([100.0 if kind == "strong" else 0.08]),
            (env,),
            np.array([0]),
            np.array([True]),
        )
        for slot, f in enumerate(frequencies)
    )
    return TimelineBatch(
        np.array([0]),
        tracks,
        FluxSchedule(np.array([0, 1]), np.array([0.0]), np.array([[0]])),
        np.array([135.0]),
    )


def test_smooth_multiple_drives_route_with_explicit_admission():
    routes = {}
    with closing(Engine(1, 1000000)) as engine:
        for kind in ("smooth", "complex", "rough", "degenerate", "dc", "strong"):
            result = engine.evolve(model_source(), (timeline(kind),), cancel=Event())
            routes[kind] = dict(result.paths.segments_by_kernel)
        result = engine.evolve(model_source(), (timeline(tones=3),), cancel=Event())
        routes["three"] = dict(result.paths.segments_by_kernel)
    expected = {kind: {"t4": 1, "dop853": 1} for kind in routes}
    expected["smooth"] = expected["three"] = {"t4": 1, "multitone_rwa": 1}
    assert routes == expected


@pytest.fixture
def kernel_case() -> KernelCase:
    family = next((k for k in REGISTRY if k.name == "multitone_rwa"), None)
    assert family is not None, "smooth simultaneous tones need their own RWA family"
    source = model_source()
    table = SegmentTable(
        timeline=np.arange(3),
        start_ns=np.array([0.0, 7.0, 19.0]),
        duration_ns=np.array([8.0, 16.0, 12.0]),
        level=np.zeros((3, 1), dtype=np.int64),
        tones=np.array([2, 3, 2]),
        tone_port=np.array([[0, 1, -1], [0, 1, 1], [1, 0, -1]]),
        tone_frequency_ghz=np.tile([0.3, -0.6, 0.9], (3, 1)),
        tone_amplitude=np.tile([0.08, -0.06, 0.04], (3, 1)),
        tone_theta_rad=np.array([[0.4, 1.4, 0], [1.1, -0.8, 2.2], [-0.2, 0.7, 0]]),
        tone_envelope=np.array([[0, 0, -1], [0, 0, 0], [0, 0, -1]]),
        tone_envelope_offset_ns=np.tile([40.0, 48.0, 32.0], (3, 1)),
        envelopes=(smooth(),),
    )
    return KernelCase(
        family, source, table, np.repeat(source.initial_density[None], 3, axis=0)
    )


class TestMultitoneRWA(KernelContract):
    pass


@pytest.mark.parametrize(
    "field",
    [
        "tone_frequency_ghz",
        "tone_port",
        "tone_amplitude",
        "tone_theta_rad",
        "tone_envelope_offset_ns",
        "envelopes",
        "budget",
    ],
)
def test_numerical_keys_distinguish_all_drive_content(
    kernel_case: KernelCase, field: str
):
    case = kernel_case
    rows = np.array([0])
    table = case.table
    budget = case.budget
    context = KernelContext(case.source, budget, WorkPlan())
    original = case.kernel.prepare_keys(table, rows, context)
    if field == "envelopes":
        table = replace(
            table, envelopes=(replace(smooth(), samples=smooth().samples * 0.99),)
        )
    elif field == "budget":
        budget *= 2
    else:
        values = getattr(table, field).copy()
        values[0, 1] = 0 if field == "tone_port" else values[0, 1] + 0.01
        table = replace(table, **{field: values})
    changed = case.kernel.prepare_keys(
        table, rows, KernelContext(case.source, budget, WorkPlan())
    )
    assert original.tobytes() != changed.tobytes()


def test_tight_budget_declines_approximation(kernel_case: KernelCase):
    case = kernel_case
    assert not case.kernel.guard(
        case.table, np.arange(3), KernelContext(case.source, 1e-8, WorkPlan())
    ).any()


def test_third_carrier_changes_joint_density_within_lab_budget(kernel_case: KernelCase):
    from zcu_tools.simulate.evolution.kernel import ApplyContext

    from tests.simulate.evolution.contract.kernel_suite import assert_budget, reference

    case = kernel_case
    model = case.source.model((0,))
    coupling = model.drives[1].in_phase_ghz.copy()
    coupling[0, 2] = coupling[2, 0] = 0.004
    model = replace(
        model, drives=(model.drives[0], replace(model.drives[1], in_phase_ghz=coupling))
    )
    initial = np.diag([1.0, 0.0, 0.0]).astype(complex)
    source = ModelSource(initial, lambda level: replace(model, level=level))
    table = replace(
        case.table,
        tones=np.full(3, 3),
        tone_port=np.tile([0, 1, 1], (3, 1)),
        tone_envelope=np.zeros((3, 3), dtype=np.int64),
        duration_ns=np.full(3, 128.0),
        tone_envelope_offset_ns=np.zeros((3, 3)),
    )
    states = []
    rows = np.array([1])
    for enabled in (False, True):
        amplitude = table.tone_amplitude.copy()
        amplitude[1, 2] = 0.2 if enabled else 0.0
        selected = replace(table, tone_amplitude=amplitude)
        context = KernelContext(source, case.budget, WorkPlan())
        assert case.kernel.guard(selected, rows, context).all()
        keys = case.kernel.prepare_keys(selected, rows, context)
        prepared = case.kernel.prepare(keys, selected, rows, context)
        actual = initial[None].copy()
        case.kernel.apply(
            prepared, np.array([0]), selected, rows, actual, ApplyContext(case.budget)
        )
        expected = reference(replace(case, source=source, table=selected), 1, initial)
        assert_budget(actual[0], expected, case.budget)
        states.append(actual[0])
    assert np.max(np.abs(states[1] - states[0])) > 0.01


def test_continuous_flux_motion_declines_even_at_one_quantized_level(
    kernel_case: KernelCase,
):
    from zcu_tools.simulate.evolution import FluxWaveform

    case = kernel_case
    # The interior excursion returns to its original flux. Endpoint-only
    # inspection would incorrectly admit it while every level is still zero.
    waveform = FluxWaveform(
        np.array([0, 3, 6, 9]),
        np.tile([0.0, 10.0, 40.0], 3),
        np.tile([[0.2], [0.201], [0.2]], (3, 1)),
    )
    table = replace(case.table, flux_waveform=waveform)
    context = KernelContext(case.source, case.budget, WorkPlan())
    assert not case.kernel.guard(table, np.arange(3), context).any()
    fixed = replace(table, flux_waveform=replace(waveform, flux=np.full((9, 1), 0.2)))
    assert case.kernel.guard(fixed, np.arange(3), context).all()
