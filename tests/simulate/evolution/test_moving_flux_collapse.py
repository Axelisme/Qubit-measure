"""Driven continuous-flux Lindblad evolution through the public Engine."""

from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from zcu_tools.simulate.boundary import Cancelled
from zcu_tools.simulate.evolution import (
    CaptureBuffer,
    Engine,
    EnvelopeSpec,
    FluxExpansion,
    FluxSchedule,
    FluxTerm,
    FluxWaveform,
    ModelSource,
    NoValidKernel,
    SecularDrive,
    SecularModel,
    SegmentShift,
    TimelineBatch,
    read_frozen_manifest,
    read_physical_input,
    reference_key,
    write_capture,
)

from ._engine import batch, track

pytestmark = pytest.mark.sim

_START, _SWITCH, _END = 1.25, 3.45, 7.0
_SHIFT = np.array([0.019, -0.017, 0.025])
_RATE = 0.052
_DEPHASING = np.array([0.0, 0.12, -0.08])
_EXTRA_DEPHASING = np.array([0.0, 0.16, 0.08])


@dataclass(frozen=True)
class _Scene:
    timeline: TimelineBatch
    waveform: FluxWaveform
    expansion: FluxExpansion
    models: tuple[SecularModel, SecularModel]
    initial: NDArray[np.complex128]
    lab_jump: NDArray[np.complex128]
    lab_drives: tuple[SecularDrive, SecularDrive]

    def source(self, *, with_jump: bool = True) -> ModelSource:
        def build(level: tuple[int, ...]) -> SecularModel:
            model = self.models[level[0]]
            return model if with_jump else replace(model, collapse_operators=())

        return ModelSource(self.initial, build, self.expansion)


def _gap_decay(amplitudes: NDArray[np.float64]) -> NDArray[np.float64]:
    return (amplitudes[:, None] - amplitudes[None, :]) ** 2 / 2


def _flux_h(expansion: FluxExpansion, flux: float) -> NDArray[np.complex128]:
    return expansion.static + np.sin(2 * np.pi * flux) * expansion.terms[0].matrix


@pytest.fixture
def scene() -> _Scene:
    static = np.diag([0.0, 0.22, 0.53]).astype(complex)
    vertex = np.array([[0, 0.12j, 0.055], [-0.12j, 0.032, 0], [0.055, 0, 0]], complex)
    expansion = FluxExpansion(static, (FluxTerm(0, "sin_2pi", vertex),))
    waveform = FluxWaveform(
        np.array([0, 6]),
        np.array([0.0, 1.91, 2.6, _SWITCH, 4.87, 6.1]),
        np.array([[0.0], [0.08], [-0.046], [0.19], [0.06], [-0.09]]),
    )
    x = np.array([[0, 0.07, 0], [0.07, 0, 0.025j], [0, -0.025j, 0]], complex)
    y = np.array([[0, -0.04j, 0], [0.04j, 0, 0.02], [0, 0.02, 0]], complex)
    lab_drives = (
        SecularDrive("drive", x, y),
        SecularDrive("other", np.asarray(y * 1.5, dtype=complex), -x),
    )
    lab_jump = np.array([[0, 0.18j, 0.27], [0, 0, 0.32 - 0.11j], [0, 0, 0]], complex)
    rates = np.zeros((3, 3))
    rates[1, 2] = _RATE

    def at_level(level: int, flux: float) -> SecularModel:
        energies, basis = np.linalg.eigh(_flux_h(expansion, flux))

        def in_row(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
            return basis.conj().T @ matrix @ basis

        drives = tuple(
            SecularDrive(
                drive.name, in_row(drive.in_phase_ghz), in_row(drive.quadrature_ghz)
            )
            for drive in (lab_drives if level == 0 else lab_drives[::-1])
        )
        return SecularModel(
            (level,),
            f"flux-fixture-{level}",
            energies,
            basis,
            drives,
            rates,
            _gap_decay(_DEPHASING),
            collapse_operators=(in_row(lab_jump),),
        )

    models = (at_level(0, 0.0), at_level(1, 0.19))
    vector = np.array([0.23 - 0.07j, 0.34j, 1.0])
    vector /= np.linalg.norm(vector)
    initial = np.outer(vector, vector.conj())
    envelope = EnvelopeSpec(
        "sampled",
        np.array([0.3 + 0.1j, -0.2 + 0.8j, 0.65 - 0.4j, 0.1j]),
        np.array([0.0, 1.44, 3.81, 5.3]),
    )
    first = replace(
        track(1, envelope, start=0.4, duration=7.0),
        frequency_ghz=np.array([0.37]),
        phase_rad=np.array([-0.51]),
        amplitude=np.array([1.15]),
    )
    second = replace(
        track(1, EnvelopeSpec("constant"), duration=8.0),
        port="other",
        frequency_ghz=np.array([0.28]),
        phase_rad=np.array([0.77]),
        amplitude=np.array([-0.72]),
    )
    timeline = replace(
        batch(1, duration=_END),
        tracks=(first, second),
        flux=FluxSchedule(
            np.array([0, 2]), np.array([0.0, _SWITCH]), np.array([[0], [1]])
        ),
        flux_waveform=waveform,
        segment_boundaries=((_START,),),
        start_ns=np.array([_START]),
        initial_densities=initial[None],
        segment_shifts=(
            (SegmentShift(_SWITCH, _END, _SHIFT, _gap_decay(_EXTRA_DEPHASING)),),
        ),
    )
    return _Scene(timeline, waveform, expansion, models, initial, lab_jump, lab_drives)


def _lab_endpoints(
    scene: _Scene, *, rtol: float, max_step: float, with_jump: bool = True
) -> NDArray[np.complex128]:
    """Integrate in fixed coordinates, representing even the secular bath by jumps."""
    first, second = scene.timeline.tracks
    envelope = first.envelopes[0]

    def rhs(
        time: float,
        state: NDArray[np.complex128],
        shift: NDArray[np.complex128],
        jumps: list[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        rho = state.reshape(3, 3)
        phi = float(np.interp(time, scene.waveform.time_ns, scene.waveform.flux[:, 0]))
        h = _flux_h(scene.expansion, phi) + shift
        sampled = np.interp(
            time - first.start_ns[0], envelope.sample_ns, envelope.samples
        )
        for tone, value, drive in zip(
            (first, second), (sampled, 1.0), scene.lab_drives, strict=True
        ):
            angle = 2 * np.pi * tone.frequency_ghz[0] * time + tone.phase_rad[0]
            z = tone.amplitude[0] * value * np.exp(1j * angle)
            h += z.real * drive.in_phase_ghz + z.imag * drive.quadrature_ghz
        out = -2j * np.pi * (h @ rho - rho @ h)
        for jump in jumps:
            adjoint = jump.conj().T
            loss = adjoint @ jump
            out += jump @ rho @ adjoint - (loss @ rho + rho @ loss) / 2
        return out.ravel()

    initial_basis = scene.models[0].eigenvectors
    flat = (initial_basis @ scene.initial @ initial_basis.conj().T).ravel()
    endpoints = []
    for level, (start, end) in enumerate(((_START, _SWITCH), (_SWITCH, _END))):
        basis = scene.models[level].eigenvectors
        shift = (
            basis @ np.diag(_SHIFT) @ basis.conj().T if level else np.zeros_like(basis)
        )
        jumps = [
            np.sqrt(_RATE) * np.outer(basis[:, 1], basis[:, 2].conj()),
            basis @ np.diag(_DEPHASING) @ basis.conj().T,
        ]
        if level:
            jumps.append(basis @ np.diag(_EXTRA_DEPHASING) @ basis.conj().T)
        if with_jump:
            jumps.append(scene.lab_jump)
        # Resolve the waveform and envelope slope breaks by step refinement,
        # without sharing the Engine's knot partitioning or row transforms.
        solution = solve_ivp(
            rhs,
            (start, end),
            flat,
            args=(shift, jumps),
            rtol=rtol,
            atol=rtol / 100,
            max_step=max_step,
        )
        assert solution.success
        flat = solution.y[:, -1]
        endpoints.append(initial_basis.conj().T @ flat.reshape(3, 3) @ initial_basis)
    return np.stack(endpoints)


def test_driven_moving_flux_matches_phase_sensitive_lab_density(scene: _Scene) -> None:
    reference = _lab_endpoints(scene, rtol=2e-10, max_step=0.065)
    refined = _lab_endpoints(scene, rtol=2e-12, max_step=0.006)
    no_jump = _lab_endpoints(scene, rtol=2e-12, max_step=0.006, with_jump=False)
    budget = 1e-8
    prefix = replace(scene.timeline, end_ns=np.array([_SWITCH]), segment_shifts=())
    with (
        closing(scene.source()) as source,
        closing(scene.source(with_jump=False)) as control,
        closing(Engine(1, backend="cpu")) as engine,
    ):
        whole = engine.evolve(source, (scene.timeline,), budget=budget)
        first = engine.evolve(source, (prefix,), budget=budget)
        suffix = replace(
            scene.timeline,
            start_ns=np.array([_SWITCH]),
            initial_densities=first.densities,
        )
        continued = engine.evolve(source, (suffix,), budget=budget)
        without = engine.evolve(control, (scene.timeline,), budget=budget)

    actual = np.stack((first.densities[0], whole.densities[0]))
    coherence = actual[-1, 0, 1]
    print(
        f"reference refinement={np.max(np.abs(refined - reference)):.3e}; "
        f"density error={np.max(np.abs(actual - refined)):.3e}; "
        f"coherence={coherence:.10g} vs {refined[-1, 0, 1]:.10g}; "
        f"continuation error={np.max(np.abs(continued.densities - whole.densities)):.3e}; "
        f"no-L density delta={np.max(np.abs(without.densities - whole.densities)):.3e}; "
        f"no-L coherence delta={abs(coherence - without.densities[0, 0, 1]):.3e}"
    )
    np.testing.assert_allclose(refined, reference, rtol=0, atol=2e-9)
    np.testing.assert_allclose(actual, refined, rtol=0, atol=3e-8)
    assert coherence == pytest.approx(refined[-1, 0, 1], abs=3e-8, rel=0)
    np.testing.assert_allclose(continued.densities, whole.densities, rtol=0, atol=3e-8)
    np.testing.assert_allclose(without.densities[0], no_jump[-1], rtol=0, atol=3e-8)
    assert abs(refined[-1, 0, 1] - no_jump[-1, 0, 1]) > 0.01
    for result in (whole, without):
        assert result.paths.segments_by_kernel == {"flux_dop853": 2}
        assert result.paths.execution_device == "cpu"


def test_continuous_and_fixed_full_loss_select_only_their_cpu_families(
    scene: _Scene,
) -> None:
    fixed = replace(
        scene.timeline,
        points=np.array([8]),
        flux=batch(1).flux,
        flux_waveform=None,
        segment_shifts=(),
    )
    with closing(scene.source()) as source, closing(Engine(1)) as engine:
        report = engine.inspect_dispatch(source, (scene.timeline, fixed))
        result = engine.evolve(source, (scene.timeline, fixed))
    assert [row.family for row in report.rows] == [
        "flux_dop853",
        "flux_dop853",
        "dop853",
    ]
    assert result.paths.segments_by_kernel == {"flux_dop853": 2, "dop853": 1}
    assert result.paths.execution_device == "cpu"


@pytest.mark.parametrize("unavailable", ["inactive", "other-point-only", "excluded"])
def test_full_loss_flux_refuses_unavailable_routes(
    scene: _Scene, unavailable: str
) -> None:
    moving = scene.timeline
    batches = (moving,)
    families = None
    points = (0,)
    if unavailable == "inactive":
        tracks = tuple(
            replace(tone, active=np.array([False])) for tone in moving.tracks
        )
        batches = (replace(moving, tracks=tracks),)
    elif unavailable == "other-point-only":
        # A waveform provided by another batch must not opt this point in.
        batches = (moving, replace(moving, points=np.array([8]), flux_waveform=None))
        points = (8,)
    else:
        families = ("dop853", "multitone_rwa", "adiabatic_map")
    with (
        closing(scene.source()) as source,
        closing(Engine(1, backend="cpu", families=families)) as engine,
    ):
        for run in (engine.inspect_dispatch, engine.evolve):
            with pytest.raises(
                NoValidKernel, match="collapse operators.*flux_dop853"
            ) as error:
                run(source, batches)
            assert error.value.points == points
            assert error.value.attempted == ()


@pytest.mark.parametrize("expansion_error", ["missing", "dimension", "loop"])
def test_full_loss_waveform_requires_matching_expansion(
    scene: _Scene, expansion_error: str
) -> None:
    expansion = None
    if expansion_error == "dimension":
        expansion = FluxExpansion(np.zeros((2, 2), complex), ())
    elif expansion_error == "loop":
        (term,) = scene.expansion.terms
        expansion = replace(scene.expansion, terms=(replace(term, loop=1),))
    with (
        closing(
            ModelSource(scene.initial, lambda level: scene.models[level[0]], expansion)
        ) as source,
        closing(Engine(1, backend="cpu")) as engine,
    ):
        for run in (engine.inspect_dispatch, engine.evolve):
            with pytest.raises(NoValidKernel, match="matching expansion"):
                run(source, (scene.timeline,))


@dataclass
class _Checkpoints:
    stop_at: int | None = None
    seen: int = 0

    def is_set(self) -> bool:
        self.seen += 1
        return self.stop_at is not None and self.seen >= self.stop_at


def test_cancelled_moving_full_loss_has_no_completed_capture_or_reuse(
    scene: _Scene,
) -> None:
    # Measure checkpoint count through the public query on a successful call.
    # Cancelling at that last checkpoint also covers the result handoff.
    normal = _Checkpoints()
    with closing(scene.source()) as source, closing(Engine(1, backend="cpu")) as engine:
        engine.evolve(source, (scene.timeline,), cancel=normal, capture=CaptureBuffer())
    capture = CaptureBuffer()
    with closing(scene.source()) as source, closing(Engine(1, backend="cpu")) as engine:
        with pytest.raises(Cancelled):
            engine.evolve(
                source,
                (scene.timeline,),
                cancel=_Checkpoints(normal.seen),
                capture=capture,
                reuse_key=b"stop",
            )
        assert not engine.reuse(source, b"stop")
    (record,) = capture.records
    metadata = record.metadata
    assert isinstance(metadata, dict)
    assert metadata["complete"] is False
    assert {row["family"] for row in metadata["routes"]} == {"flux_dop853"}


def test_moving_full_loss_capture_replays_the_same_physics(
    scene: _Scene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))
    capture = CaptureBuffer()
    budget = 1e-8
    with (
        closing(scene.source()) as source,
        closing(scene.source(with_jump=False)) as control,
        closing(Engine(1, backend="cpu")) as engine,
    ):
        actual = engine.evolve(
            source, (scene.timeline,), budget=budget, capture=capture
        )
        path = write_capture(capture)
        assert path is not None
        (record,) = read_frozen_manifest(path)
        metadata = record.metadata
        assert isinstance(metadata, dict)
        assert metadata["frozen_physical_version"] == 2
        assert metadata["complete"] is True
        assert metadata["source_identity"] == source.model((0,)).content_identity
        assert [row["family"] for row in metadata["routes"]] == ["flux_dop853"] * 2
        replay = read_physical_input(record.physical)
        with closing(replay.source):
            key = reference_key(source, scene.timeline)
            assert reference_key(replay.source, replay.batch) == key
            assert reference_key(control, scene.timeline) != key
            for level in ((0,), (1,)):
                original, restored = source.model(level), replay.source.model(level)
                assert restored.content_identity == original.content_identity
                np.testing.assert_array_equal(
                    restored.collapse_operators, original.collapse_operators
                )
            waveform = replay.batch.flux_waveform
            assert waveform is not None
            np.testing.assert_array_equal(waveform.time_ns, scene.waveform.time_ns)
            np.testing.assert_array_equal(waveform.flux, scene.waveform.flux)
            rebuilt = engine.evolve(
                replay.source, (replay.batch,), budget=metadata["budget"]
            )
    assert rebuilt.paths.segments_by_kernel == actual.paths.segments_by_kernel
    np.testing.assert_array_equal(rebuilt.densities, actual.densities)
    print(
        f"capture={path}; physical_sha256={record.physical_sha256}; reference_key={key}"
    )
