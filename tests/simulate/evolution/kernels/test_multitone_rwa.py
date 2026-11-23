"""Joint-basis multi-tone approximation at the Engine and C5 seams."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from scipy.linalg import expm
from zcu_tools.simulate.evolution import (
    DriveTrack,
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    NoValidKernel,
    SecularDrive,
    SecularModel,
    TimelineBatch,
)
from zcu_tools.simulate.evolution.kernel import KernelContext, SegmentTable
from zcu_tools.simulate.evolution.kernels import REGISTRY
from zcu_tools.simulate.evolution.workplan import WorkPlan

from tests.simulate.evolution.contract.interaction_reference import (
    interaction_reference,
)
from tests.simulate.evolution.contract.kernel_suite import KernelCase, KernelContract
from tests.simulate.evolution.contract.multitone_cases import accumulated_pair_case


def endpoint_pair():
    """Two constant tones with omitted diagonal and counter-rotating channels."""
    x = np.array([[0.012, 0.02], [0.02, -0.009]], dtype=np.complex128)
    b = np.array([[-0.008, 0.014], [0.014, 0.005]], dtype=np.complex128)
    model = SecularModel(
        (0,),
        "endpoint-pair",
        np.array([0.0, 0.6]),
        np.eye(2, dtype=complex),
        (
            SecularDrive("a", x, np.zeros_like(x)),
            SecularDrive("b", b, np.zeros_like(b)),
        ),
        np.array([[0.0, 0.00003], [0.00001, 0.0]]),
        np.array([[0.0, 0.00002], [0.00002, 0.0]]),
    )
    vector = np.array([1.0, 0.4j], dtype=complex) / np.sqrt(1.16)
    source = ModelSource(
        np.outer(vector, vector.conj()), lambda level: replace(model, level=level)
    )
    tracks = tuple(
        DriveTrack(
            name,
            np.array([7.0]),
            np.array([24.375]),
            np.array([frequency]),
            np.array([phase]),
            np.array([gain]),
            (EnvelopeSpec("constant"),),
            np.array([0]),
            np.array([True]),
        )
        for name, frequency, phase, gain in (
            ("a", 0.6, 0.3, 0.4),
            ("b", 0.63, 1.1, -0.2),
        )
    )
    batch = TimelineBatch(
        np.array([0]),
        tracks,
        FluxSchedule(np.array([0, 1]), np.array([0.0]), np.array([[0]])),
        np.array([33.515]),
    )
    return source, batch


def test_constant_pair_endpoint_density():
    source, batch = endpoint_pair()
    with (
        closing(source),
        closing(Engine(1, families=("multitone_rwa", "t4"))) as engine,
    ):
        expected = interaction_reference(source, batch)
        actual = engine.evolve(source, (batch,), cancel=Event())
    assert actual.paths.segments_by_kernel == {"multitone_rwa": 1, "t4": 2}
    error = float(np.max(np.abs(actual.densities - expected.density)))
    assert error <= 2e-3, f"endpoint full-density error {error:.9g} > 2e-3"


def test_constant_pair_row_splitting_keeps_endpoint_density():
    source, batch = endpoint_pair()
    start = float(batch.tracks[0].start_ns[0])
    duration = float(batch.tracks[0].duration_ns[0])
    errors = []
    with (
        closing(source),
        closing(Engine(1, families=("multitone_rwa", "t4"))) as engine,
    ):
        expected = interaction_reference(source, batch)
        for count in (1, 2, 4, 8):
            boundaries = tuple(np.linspace(start, start + duration, count + 1)[1:-1])
            split = replace(batch, segment_boundaries=(boundaries,))
            actual = engine.evolve(source, (split,), cancel=Event())
            assert actual.paths.segments_by_kernel == {"multitone_rwa": count, "t4": 2}
            error = float(np.max(np.abs(actual.densities - expected.density)))
            assert error <= 2e-3, f"{count} rows: full-density error {error:.9g}"
            errors.append(error)
    # Each row is independently dressed; extra cuts must not accumulate a
    # first-order endpoint defect. Compare every count to the same exact oracle.
    assert max(errors) <= 2 * errors[0], f"1/2/4/8-row errors: {errors}"
    print(f"1/2/4/8-row full-density errors: {errors}")


@pytest.mark.parametrize("duration", [1000.0, 2000.0])
@pytest.mark.parametrize("coherent", [False, True], ids=["thermal", "cross-block"])
def test_disconnected_drive_blocks_keep_full_bath_and_coherences(duration, coherent):
    """Resonant circular drives give an independent constant-generator oracle.

    Drive components {0,1} and {2,3} do not couple, but their populations
    exchange through the bath. Nonzero cross-component coherences must also
    survive. Neither component-local bath maps nor dropping cross blocks works.
    """
    energies = np.array([0.0, 0.3, 1.0, 1.7])
    drives = []
    h = np.zeros((4, 4), dtype=np.complex128)
    for port, (i, j) in enumerate(((0, 1), (2, 3))):
        x = np.zeros_like(h)
        y = np.zeros_like(h)
        x[i, j] = x[j, i] = 0.002
        y[i, j], y[j, i] = 0.002j, -0.002j
        drives.append(SecularDrive(str(port), x, y))
        h[i, j] = h[j, i] = 0.0002
    rates = np.array(
        [
            [0, 0.0002, 0.0003, 0],
            [0.0001, 0, 0, 0.0004],
            [0.00005, 0, 0, 0.0001],
            [0, 0.00002, 0.00003, 0],
        ]
    )
    gamma = 0.00001 * (np.ones((4, 4)) - np.eye(4))
    model = SecularModel(
        (0,),
        "disconnected-drives-full-bath",
        energies,
        np.eye(4, dtype=np.complex128),
        tuple(drives),
        rates,
        gamma,
    )
    initial = np.diag(np.array([0.55, 0.25, 0.15, 0.05])).astype(np.complex128)
    if coherent:
        vector = np.array([1, 1j, -1, 1j], dtype=np.complex128) / 2
        initial = 0.7 * initial + 0.3 * np.outer(vector, vector.conj())
    outgoing = rates.sum(axis=0)
    decay = (outgoing[:, None] + outgoing[None, :]) / 2 + gamma
    generator = np.empty((16, 16), dtype=np.complex128)
    for column in range(16):
        basis = np.zeros((4, 4), dtype=np.complex128)
        basis.flat[column] = 1
        derivative = -2j * np.pi * (h @ basis - basis @ h) - decay * basis
        derivative[np.diag_indices(4)] += rates @ np.diag(basis)
        generator[:, column] = derivative.ravel()
    expected = (expm(duration * generator) @ initial.ravel()).reshape(4, 4)
    phases = np.exp(-2j * np.pi * energies * duration)
    expected *= phases[:, None] * phases[None, :].conj()
    batch = TimelineBatch(
        np.array([0]),
        tuple(
            DriveTrack(
                str(port),
                np.array([0.0]),
                np.array([duration]),
                np.array([frequency]),
                np.array([0.0]),
                np.array([0.1]),
                (EnvelopeSpec("constant"),),
                np.array([0]),
                np.array([True]),
            )
            for port, frequency in enumerate((0.3, 0.7))
        ),
        FluxSchedule(np.array([0, 1]), np.array([0.0]), np.array([[0]])),
        np.array([duration]),
    )
    with (
        closing(
            ModelSource(initial, lambda level: replace(model, level=level))
        ) as source,
        closing(Engine(1)) as engine,
    ):
        result = engine.evolve(source, (batch,), cancel=Event())
    assert result.paths.segments_by_kernel == {"multitone_rwa": 1}
    np.testing.assert_allclose(result.densities[0], expected, atol=1e-9, rtol=0)


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


def timeline(tones: int = 2, *, second_frequency: float = -0.6) -> TimelineBatch:
    env = smooth()
    frequencies = [0.3, second_frequency, 0.9][:tones]
    tracks = tuple(
        DriveTrack(
            "a" if slot == 0 else "b",
            np.array([7.0]),
            np.array([128.0]),
            np.array([f]),
            np.array([0.4 + slot]),
            np.array([0.08]),
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


@pytest.fixture
def kernel_case() -> KernelCase:
    family = next((k for k in REGISTRY if k.name == "multitone_rwa"), None)
    assert family is not None, "constant simultaneous tones need their own RWA family"
    source = model_source()
    table = SegmentTable(
        timeline=np.arange(3),
        start_ns=np.array([0.0, 7.0, 19.0]),
        duration_ns=np.array([8.0, 16.0, 12.0]),
        level=np.zeros((3, 1), dtype=np.int64),
        tones=np.array([2, 2, 2]),
        tone_port=np.array([[0, 1, -1], [0, 1, -1], [1, 0, -1]]),
        tone_frequency_ghz=np.tile([0.3, -0.6, 0.9], (3, 1)),
        tone_amplitude=np.tile([0.08, -0.06, 0.04], (3, 1)),
        tone_theta_rad=np.array([[0.4, 1.4, 0], [1.1, -0.8, 2.2], [-0.2, 0.7, 0]]),
        tone_envelope=np.array([[0, 0, -1], [0, 0, -1], [0, 0, -1]]),
        tone_envelope_offset_ns=np.tile([40.0, 48.0, 32.0], (3, 1)),
        envelopes=(EnvelopeSpec("constant"),),
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
    if field == "budget":
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


def constant_pair(offsets=(0.0, 0.0)):
    """Small coupled pair with a 32.049 MHz conditional double excitation."""
    energies = np.array([0.0, 6.7, 7.5, 14.232049])
    drives = []
    for name, edges in (("a", ((0, 1), (2, 3))), ("b", ((0, 2), (1, 3)))):
        positive = np.zeros((4, 4), dtype=np.complex128)
        for i, j in edges:
            positive[i, j] = 0.01
        drives.append(
            SecularDrive(name, positive + positive.T, 1j * (positive - positive.T))
        )
    model = SecularModel(
        (0,),
        "conditional-pair",
        energies,
        np.eye(4, dtype=np.complex128),
        tuple(drives),
        np.zeros((4, 4)),
        np.zeros((4, 4)),
    )
    vector = np.array([1, 1j, -1, 1], dtype=np.complex128) / 2
    source = ModelSource(
        np.outer(vector, vector.conj()), lambda level: replace(model, level=level)
    )
    tracks = tuple(
        DriveTrack(
            name,
            np.array([0.0]),
            np.array([128.0]),
            np.array([frequency + offset]),
            np.array([0.3]),
            np.array([1.0]),
            (EnvelopeSpec("constant"),),
            np.array([0]),
            np.array([True]),
        )
        for name, frequency, offset in zip(("a", "b"), (6.7, 7.5), offsets, strict=True)
    )
    batch = TimelineBatch(
        np.array([0]),
        tracks,
        FluxSchedule(np.array([0, 1]), np.array([0.0]), np.array([[0]])),
        np.array([128.0]),
    )
    return source, batch


@pytest.mark.parametrize("port", ["a", "b"])
def test_same_port_constant_pair_retains_original_two_tone_routing(port):
    source = model_source()
    batch = timeline()
    batch = replace(
        batch,
        tracks=tuple(
            replace(track, port=port, envelopes=(EnvelopeSpec("constant"),))
            for track in batch.tracks
        ),
    )
    families = ("t4", "common_frame", "multitone_gate", "floquet", "dop853")
    with closing(Engine(1, families=families)) as engine:
        original = engine.evolve(source, (batch,), cancel=Event())
    with closing(Engine(1)) as engine:
        result = engine.evolve(source, (batch,), cancel=Event())
    assert set(original.paths.segments_by_kernel) & {"floquet", "dop853"}
    assert result.paths.segments_by_kernel == original.paths.segments_by_kernel
    np.testing.assert_array_equal(result.densities, original.densities)


@pytest.mark.parametrize("tones,frequency", [(2, 0.0), (3, -0.6)])
def test_sampled_multitone_rejects_outside_the_two_carrier_domain(
    tones: int, frequency: float
) -> None:
    batch = timeline(tones, second_frequency=frequency)
    with (
        closing(Engine(1, families=("t4", "multitone_rwa"))) as engine,
        pytest.raises(NoValidKernel) as rejected,
    ):
        engine.evolve(model_source(), (batch,))
    assert rejected.value.domain == "multiple"
    assert rejected.value.points == (0,)


@pytest.mark.parametrize(
    "frequency, driven_kernel",
    [
        (0.3, "single_tone_rwa"),
        (0.301437, "multitone_rwa"),
        (-0.301437, "multitone_rwa"),
    ],
)
def test_close_carriers_keep_joint_interference_and_coherent_density(
    frequency: float, driven_kernel: str
) -> None:
    batch = timeline()
    batch = replace(
        batch,
        tracks=tuple(
            replace(
                track,
                frequency_ghz=np.array([f]),
                amplitude=np.array([0.8]),
                envelopes=(EnvelopeSpec("constant"),),
            )
            for track, f in zip(batch.tracks, (0.3, frequency), strict=True)
        ),
    )
    source = model_source()
    with closing(Engine(1)) as engine:
        actual = engine.evolve(source, (batch,), cancel=Event())
        expected = engine.evolve(source, (batch,), budget=1e-7, cancel=Event())
    # Equal signed carriers with the same constant shape and support fold.
    assert actual.paths.segments_by_kernel == {"t4": 1, driven_kernel: 1}
    # RWA is approximate: use the existing constant-pair density tolerance,
    # still much tighter than the qualitative population allowance of 0.1.
    np.testing.assert_allclose(actual.densities, expected.densities, atol=0.01, rtol=0)
    assert np.max(np.abs(actual.densities[0] - source.initial_density)) > 0.01


def test_constant_pair_uses_joint_rwa_and_preserves_coherent_joint_input():
    source, batch = constant_pair()
    with closing(Engine(1)) as engine:
        result = engine.evolve(source, (batch,), cancel=Event())
    assert result.paths.segments_by_kernel == {"multitone_rwa": 1}
    # A tight-budget Engine must retain full-drive dynamics as the reference.
    with closing(Engine(1)) as engine:
        reference_result = engine.evolve(source, (batch,), budget=1e-7, cancel=Event())
    np.testing.assert_allclose(
        result.densities, reference_result.densities, atol=0.01, rtol=0
    )
    assert np.max(np.abs(result.densities[0] - source.initial_density)) > 0.01


@pytest.mark.parametrize("offsets", [(0.084, -0.052), (0.084, 0.084)])
def test_constant_combination_resonance_restores_contributing_edges(offsets):
    source, safe = constant_pair()
    _, risky = constant_pair(offsets)
    # The effective channel is unsafe to omit, but its four-state retained
    # graph fits. Restoring its paths must keep the full-drive joint dynamics.
    with closing(Engine(1)) as engine:
        admitted = engine.evolve(source, (safe,), cancel=Event())
        rejected = engine.evolve(source, (risky,), cancel=Event())
    assert (admitted.paths.segments_by_kernel, rejected.paths.segments_by_kernel) == (
        {"multitone_rwa": 1},
        {"multitone_rwa": 1},
    )
    with closing(Engine(1)) as engine:
        expected = engine.evolve(source, (risky,), budget=1e-7, cancel=Event())
    np.testing.assert_allclose(
        rejected.densities, expected.densities, atol=0.01, rtol=0
    )


@pytest.mark.parametrize("detuning", [0.0, 0.07])
def test_omission_admission_accepts_strong_retained_rows(detuning):
    from zcu_tools.simulate.evolution.kernel import ApplyContext
    from zcu_tools.simulate.evolution.kernels.multitone_rwa import MultitoneRWAKernel

    from tests.simulate.evolution.contract.kernel_suite import reference

    drives = []
    for port, target in (("a", 1), ("b", 2)):
        positive = np.zeros((3, 3), dtype=np.complex128)
        positive[0, target] = 0.04
        drives.append(
            SecularDrive(port, positive + positive.T, 1j * (positive - positive.T))
        )
    model = SecularModel(
        (0,),
        "strong-circular",
        np.array([0.0, 0.3, 0.9]),
        np.eye(3, dtype=np.complex128),
        tuple(drives),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
    )
    initial = np.full((3, 3), 1 / 3, dtype=np.complex128)
    source = ModelSource(initial, lambda level: replace(model, level=level))
    table = SegmentTable(
        timeline=np.array([0]),
        start_ns=np.array([0.0]),
        duration_ns=np.array([128.0]),
        level=np.zeros((1, 1), dtype=np.int64),
        tones=np.array([2]),
        tone_port=np.array([[0, 1]]),
        tone_frequency_ghz=np.array([[0.3 + detuning, 0.9]]),
        tone_amplitude=np.ones((1, 2)),
        tone_theta_rad=np.array([[0.7, -1.2]]),
        tone_envelope=np.zeros((1, 2), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((1, 2)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    kernel = MultitoneRWAKernel()
    budget = 1e-3
    with closing(WorkPlan()) as plan:
        context = KernelContext(source, budget, plan)
        rows = np.array([0])
        assert kernel.guard(table, rows, context).all()
        keys = kernel.prepare_keys(table, rows, context)
        prepared = kernel.prepare(keys, table, rows, context)
        actual = initial[None].copy()
        kernel.apply(prepared, rows, table, rows, actual, ApplyContext(budget))
    expected = reference(KernelCase(kernel, source, table, initial[None]), 0, initial)
    # Lab comparison includes the RWA DOP853 playback's budget-level error.
    np.testing.assert_allclose(actual[0], expected, atol=budget, rtol=0)


@pytest.mark.parametrize("frequencies", [(0.3, 0.9), (0.3, 0.301437)])
def test_connected_retained_graph_over_block_limit_declines(frequencies):
    from zcu_tools.simulate.evolution.kernels.multitone_rwa import MultitoneRWAKernel

    size = 17
    positive = np.diag(np.full(size - 1, 0.001), 1).astype(np.complex128)
    drive = SecularDrive("a", positive + positive.T, 1j * (positive - positive.T))
    model = SecularModel(
        (0,),
        "connected-chain",
        np.arange(size, dtype=np.float64) * 0.3,
        np.eye(size, dtype=np.complex128),
        (drive, replace(drive, name="b")),
        np.zeros((size, size)),
        np.zeros((size, size)),
    )
    source = ModelSource(
        np.eye(size, dtype=np.complex128) / size,
        lambda level: replace(model, level=level),
    )
    table = SegmentTable(
        timeline=np.array([0]),
        start_ns=np.array([0.0]),
        duration_ns=np.array([128.0]),
        level=np.zeros((1, 1), dtype=np.int64),
        tones=np.array([2]),
        tone_port=np.array([[0, 1]]),
        tone_frequency_ghz=np.array([frequencies]),
        tone_amplitude=np.ones((1, 2)),
        tone_theta_rad=np.zeros((1, 2)),
        tone_envelope=np.zeros((1, 2), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((1, 2)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    with closing(WorkPlan()) as plan:
        assert (
            not MultitoneRWAKernel()
            .guard(table, np.array([0]), KernelContext(source, 1e-3, plan))
            .any()
        )


def test_combination_restores_strongest_parent_without_fast_weak_paths(kernel_case):
    from zcu_tools.simulate.evolution.kernel import ApplyContext
    from zcu_tools.simulate.evolution.kernels.multitone_rwa import RWAPrepared

    from tests.simulate.evolution.contract.kernel_suite import reference

    case = kernel_case
    drives = []
    for name, edges in (("a", ((0, 1), (0, 2))), ("b", ((1, 3), (2, 3)))):
        positive = np.zeros((4, 4), dtype=np.complex128)
        for (i, j), coupling in zip(edges, (0.01, 0.00001), strict=True):
            positive[i, j] = coupling
        drives.append(
            SecularDrive(name, positive + positive.T, 1j * (positive - positive.T))
        )
    model = SecularModel(
        (0,),
        "dominant-raman-parent",
        np.array([0.0, 0.6, 0.7, 1.2]),
        np.eye(4, dtype=np.complex128),
        tuple(drives),
        np.zeros((4, 4)),
        np.zeros((4, 4)),
    )
    initial = np.diag([1.0, 0.0, 0.0, 0.0]).astype(np.complex128)
    source = ModelSource(initial, lambda level: replace(model, level=level))
    table = replace(
        case.table,
        tones=np.full(3, 2),
        tone_port=np.tile([0, 1, -1], (3, 1)),
        tone_frequency_ghz=np.tile([0.3, 0.9, 0.0], (3, 1)),
        tone_amplitude=np.ones((3, 3)),
        tone_envelope=np.tile([0, 0, -1], (3, 1)),
        envelopes=(EnvelopeSpec("constant"),),
        duration_ns=np.full(3, 128.0),
    )
    rows = np.array([0])
    with closing(WorkPlan()) as plan:
        context = KernelContext(source, case.budget, plan)
        assert case.kernel.guard(table, rows, context).all()
        keys = case.kernel.prepare_keys(table, rows, context)
        prepared = case.kernel.prepare(keys, table, rows, context)
        assert isinstance(prepared, RWAPrepared)
        assert np.count_nonzero(prepared.positive) == 2
        actual = initial[None].copy()
        case.kernel.apply(
            prepared, rows, table, rows, actual, ApplyContext(case.budget)
        )
    expected = reference(
        KernelCase(case.kernel, source, table, initial[None]), 0, initial
    )
    np.testing.assert_allclose(actual[0], expected, atol=case.budget, rtol=0)


def test_residual_phase_restores_edges_even_below_transfer_threshold(kernel_case):
    from zcu_tools.simulate.evolution.kernels.multitone_rwa import RWAPrepared

    case = kernel_case
    model = case.source.model((0,))
    positive = np.zeros((3, 3), dtype=np.complex128)
    positive[0, 1] = 0.001
    x, y = positive + positive.T, 1j * (positive - positive.T)
    model = replace(
        model,
        drives=(SecularDrive("a", x, y), SecularDrive("b", x * 0, y * 0)),
    )
    source = ModelSource(
        case.source.initial_density, lambda level: replace(model, level=level)
    )
    table = replace(
        case.table,
        tones=np.full(3, 2),
        tone_port=np.tile([0, 1, -1], (3, 1)),
        tone_frequency_ghz=np.tile([0.2, 0.9, 0.0], (3, 1)),
        tone_amplitude=np.ones((3, 3)),
        tone_envelope=np.tile([0, 0, -1], (3, 1)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    # g/d=.01 gives transfer <.0004 at every duration. Only accumulated
    # residual phase, after the explicit second-order shift, needs restoration.
    for duration, retained in ((10.0, False), (1e12, True)):
        selected = replace(table, duration_ns=np.full(3, duration))
        with closing(WorkPlan()) as plan:
            context = KernelContext(source, case.budget, plan)
            rows = np.array([0])
            assert case.kernel.guard(selected, rows, context).all()
            keys = case.kernel.prepare_keys(selected, rows, context)
            prepared = case.kernel.prepare(keys, selected, rows, context)
            assert isinstance(prepared, RWAPrepared)
            assert bool(np.any(prepared.positive)) == retained


def phase_proxy_pair(phase_rad, *, domain="constant"):
    """Three levels with analytically known retained norms of 0.025 GHz/gain."""
    drives = []
    for port, target in (("a", 1), ("b", 2)):
        positive = np.zeros((3, 3), dtype=np.complex128)
        positive[0, target] = 0.025
        drives.append(
            SecularDrive(port, positive + positive.T, 1j * (positive - positive.T))
        )
    model = SecularModel(
        (0,),
        "phase-proxy-pair",
        np.array([0.0, 0.3, 0.9]),
        np.eye(3, dtype=np.complex128),
        tuple(drives),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
    )
    initial = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    source = ModelSource(initial, lambda level: replace(model, level=level))
    # With distinct ports both resonant Fourier operators have norm 0.025.
    # For the same port only the 0.3 GHz edge is retained.
    inverse_frequencies = 1 / 0.3 + (0 if domain == "same_port" else 1 / 0.9)
    duration = phase_rad / (2 * np.pi * 0.025**2 * inverse_frequencies)
    tracks = tuple(
        DriveTrack(
            "a" if domain == "same_port" else port,
            np.array([0.0]),
            np.array([duration]),
            np.array([frequency]),
            np.array([0.3]),
            np.array([1.0]),
            (EnvelopeSpec("constant"),),
            np.array([0]),
            np.array([True]),
        )
        for port, frequency in (("a", 0.3), ("b", 0.9))
    )
    return source, TimelineBatch(
        np.array([0]),
        tracks,
        FluxSchedule(np.array([0, 1]), np.array([0.0]), np.array([[0]])),
        np.array([duration]),
    )


def test_retained_only_phase_does_not_reject_long_rows():
    routes = {}
    for domain in ("constant", "same_port"):
        for phase in (0.05, 0.07, 1.0):
            source, batch = phase_proxy_pair(phase, domain=domain)
            with closing(Engine(1)) as engine:
                result = engine.evolve(source, (batch,), cancel=Event())
            routes[domain, phase] = "multitone_rwa" in result.paths.segments_by_kernel
            if domain == "same_port":
                # Rejected rows retain the original two-tone routing and density.
                families = ("t4", "common_frame", "multitone_gate", "floquet", "dop853")
                with closing(Engine(1, families=families)) as engine:
                    original = engine.evolve(source, (batch,), cancel=Event())
                assert (
                    result.paths.segments_by_kernel == original.paths.segments_by_kernel
                )
                np.testing.assert_array_equal(result.densities, original.densities)
    assert routes == {
        (domain, phase): domain != "same_port"
        for domain in ("constant", "same_port")
        for phase in (0.05, 0.07, 1.0)
    }


def test_constant_resonant_pair_rejects_accumulated_error() -> None:
    """Pinned constant A16 must decline secular accumulation from ground."""
    source, batch = accumulated_pair_case()
    with (
        closing(source),
        closing(Engine(1, families=("multitone_rwa",))) as engine,
        pytest.raises(NoValidKernel),
    ):
        engine.evolve(source, (batch,), budget=1e-3)
