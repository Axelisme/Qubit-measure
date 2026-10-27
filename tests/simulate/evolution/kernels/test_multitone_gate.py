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
from zcu_tools.simulate.evolution.kernel import (
    ApplyContext,
    KernelContext,
    SegmentTable,
)
from zcu_tools.simulate.evolution.kernels import REGISTRY
from zcu_tools.simulate.evolution.kernels._rwa_gate import GATE_POLICY, GatePrepared
from zcu_tools.simulate.evolution.kernels.multitone_rwa import (
    interaction_hamiltonian,
    rwa_coefficients,
)
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
    conflict,
):
    with closing(Engine(1)) as engine:
        for phase in (0.0, 0.713):
            result = engine.evolve(source(conflict), (batch(phase),), cancel=Event())
            assert result.paths.segments_by_kernel.get("multitone_gate") == 24
            with closing(
                Engine(
                    1,
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


@pytest.mark.parametrize("conflict,count", [(0.0004, 24), (0.0, 1)])
def test_inconsistent_or_unamortized_pairs_keep_row_rwa(conflict, count):
    family()
    with closing(Engine(1)) as engine:
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


@pytest.mark.parametrize("duration", [254.0, 256.0])
@pytest.mark.parametrize("bath_scale", [1.0, 200.0])
def test_native_sample_support_and_adaptive_split_match_lab_density(
    kernel_case, duration, bath_scale
):
    from zcu_tools.simulate.evolution.kernels.dop853 import DOP853Plan

    # The finite pulse support, not the last DAC sample, ends playback. A
    # 254 ns row slices the grid; a 256 ns row holds the final sample.
    times = np.linspace(0.0, 255.045573, 257)
    env = EnvelopeSpec(
        "sampled", np.exp(-0.5 * ((times - 128.0) / 64.0) ** 2).astype(complex), times
    )
    table = replace(
        kernel_case.table,
        duration_ns=np.full(24, duration),
        tone_amplitude=np.full((24, 2), 0.5),
        envelopes=(env,),
    )
    src = source()
    model = src.model((0,))
    model = replace(
        model,
        identity=f"native-pair-bath-{bath_scale}",
        rates_per_ns=model.rates_per_ns * bath_scale,
        dephasing_per_ns=model.dephasing_per_ns * bath_scale,
    )
    src = ModelSource(src.initial_density, lambda level: replace(model, level=level))
    rows = np.arange(3, dtype=np.int64)
    with closing(WorkPlan()) as plan:
        context = KernelContext(src, 1e-3, plan)
        kernel = family()
        assert kernel.guard(table, rows, context).all(), (
            "native sample endpoints must not veto complete pulse playback"
        )
        keys = kernel.prepare_keys(table, rows[:1], context)
        prepared = kernel.prepare(keys, table, rows[:1], context)
        assert isinstance(prepared, GatePrepared)
        if bath_scale == 1.0:
            assert prepared.segments[0] == 4, (
                "small split proxy keeps the four-step path"
            )
        else:
            assert prepared.segments[0] > 4, "large split proxy must refine, not reject"
        states = np.repeat(src.initial_density[None], len(rows), axis=0)
        kernel.apply(
            prepared,
            np.zeros(3, dtype=np.int64),
            table,
            rows,
            states,
            ApplyContext(1e-3),
        )
        direct = next(k for k in REGISTRY if k.name == "dop853")
        reference = np.repeat(src.initial_density[None], len(rows), axis=0)
        direct_keys = direct.prepare_keys(table, rows, context)
        direct.apply(
            direct.prepare(direct_keys, table, rows, context),
            rows,
            table,
            rows,
            reference,
            ApplyContext(
                1e-8, DOP853Plan(np.full(3, "lab"), np.full(3, -1, dtype=np.int64))
            ),
        )
    np.testing.assert_allclose(states, reference, atol=1e-2, rtol=0)


@pytest.mark.parametrize("bath_scale", [1.0, 200.0])
@pytest.mark.parametrize("constant_companion", [False, True])
def test_compiled_joint_unitaries_match_callback(
    kernel_case, monkeypatch, bath_scale, constant_companion
):
    times = np.linspace(0.0, 80.0, 81)
    other_times = np.linspace(0.0, 39.0, 40)
    envs = (
        EnvelopeSpec(
            "sampled", np.exp(-0.5 * ((times - 40) / 32) ** 2).astype(complex), times
        ),
        EnvelopeSpec("constant")
        if constant_companion
        else EnvelopeSpec(
            "sampled",
            np.exp(-0.5 * ((other_times - 20) / 32) ** 2).astype(complex),
            other_times,
        ),
    )
    table = replace(
        kernel_case.table,
        duration_ns=np.full(24, 82.0),
        tone_frequency_ghz=np.tile([0.6002, 0.2997], (24, 1)),
        tone_amplitude=np.tile([0.4, -0.3], (24, 1)),
        tone_envelope=np.tile([0, 1], (24, 1)),
        tone_envelope_offset_ns=np.tile([3.0, 7.0], (24, 1)),
        envelopes=envs,
    )
    model = kernel_case.source.model((0,))
    model = replace(
        model,
        rates_per_ns=model.rates_per_ns * bath_scale,
        dephasing_per_ns=model.dephasing_per_ns * bath_scale,
    )
    src = ModelSource(kernel_case.source.initial_density, lambda level: model)
    rows = np.array([0], dtype=np.int64)
    calls = []
    solve = scipy.integrate.solve_ivp

    def observed(*args, **kwargs):
        calls.append(1)
        return solve(*args, **kwargs)

    with closing(WorkPlan()) as plan:
        context = KernelContext(src, 1e-3, plan)
        kernel = family()
        assert kernel.guard(table, rows, context).all()
        with monkeypatch.context() as patch:
            patch.setattr(scipy.integrate, "solve_ivp", observed)
            prepared = kernel.prepare(
                kernel.prepare_keys(table, rows, context), table, rows, context
            )
    assert isinstance(prepared, GatePrepared)
    energy = model.energies_ghz + prepared.charges[0] @ table.tone_frequency_ghz[0]
    energy -= energy[0]
    coefficients = rwa_coefficients(model, (0, 1), (0.6002, 0.2997))

    def rhs(time, state):
        amplitudes = np.array(
            [
                table.tone_amplitude[0, k]
                * (
                    1.0
                    if env.kind == "constant"
                    else np.interp(
                        time + table.tone_envelope_offset_ns[0, k],
                        env.sample_ns,
                        env.samples.real,
                    )
                )
                for k, env in enumerate(envs)
            ],
            dtype=np.complex128,
        )
        h = interaction_hamiltonian(time, energy, coefficients, amplitudes)
        return (-2j * np.pi * h @ state.reshape(4, 4)).ravel()

    boundaries = np.linspace(0.0, 82.0, prepared.segments[0] + 1)
    reference = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        solution = solve(
            rhs,
            (start, end),
            np.eye(4, dtype=complex).ravel(),
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
            max_step=4.0,
        )
        assert solution.success
        endpoint = solution.y[:, -1].reshape(4, 4)
        endpoint *= np.exp(-2j * np.pi * energy * end)[:, None]
        endpoint *= np.exp(2j * np.pi * energy * start)[None, :]
        reference.append(endpoint)
    # Allow accumulation over adaptive steps at the builder's local relative tolerance.
    np.testing.assert_allclose(
        prepared.unitaries[0], reference, atol=100 * GATE_POLICY[-2], rtol=0
    )
    assert not calls, "joint Hilbert builds must not call the Python solver callback"


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
