"""Engine routing boundaries separate dispatch costs from numerical admission."""

import gc
import sys
import weakref
from collections.abc import Sequence
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    Engine,
    EnvelopeSpec,
    NoValidKernel,
    TimelineBatch,
    merge_path_diagnostics,
)

from .accuracy_cases import (
    AccuracyCase,
    accuracy_case,
    coherent_source,
    cpu_dp54_case,
    drive_track,
    linear_drive_model,
    timeline,
)
from .reference import converged_reference, density_error
from .synthetic_models import synthetic_model


@pytest.mark.parametrize(
    ("duration", "budget", "dp54"),
    [
        pytest.param(10_000.0, 1e-3, True, id="inclusive-duration-and-budget"),
        pytest.param(10_001.0, 1e-3, False, id="over-duration"),
        pytest.param(100.0, 0.999e-3, False, id="under-budget"),
    ],
)
def test_cpu_dop853_selects_tableau_by_row_duration_and_budget(
    duration: float, budget: float, dp54: bool
) -> None:
    case = accuracy_case("sampled_gaussian_9")
    end = np.full_like(case.batch.end_ns, duration)
    batch = replace(
        case.batch,
        end_ns=end,
        tracks=tuple(
            replace(
                track,
                duration_ns=end,
                envelopes=tuple(
                    replace(env, sample_ns=env.sample_ns * (duration / 100.0))
                    for env in track.envelopes
                ),
            )
            for track in case.batch.tracks
        ),
    )
    with closing(Engine(1, backend="cpu", families=case.families)) as engine:
        paths = engine.evolve(
            case.source, (batch,), budget=budget, telemetry=True
        ).paths

    assert paths.segments_by_kernel == {"dop853": len(batch.points)}
    report = paths.routing["dop853"]
    assert report.dp54_rows == (len(batch.points) if dp54 else 0)
    assert report.dop853_rows == (0 if dp54 else len(batch.points))
    merged = merge_path_diagnostics((paths, paths)).routing["dop853"]
    assert merged.dp54_rows == 2 * report.dp54_rows
    assert merged.dop853_rows == 2 * report.dop853_rows
    reused = paths.for_reuse(telemetry=True).routing["dop853"]
    assert (reused.dp54_rows, reused.dop853_rows) == (0, 0)


def test_dispatch_prices_the_short_cpu_tableau_below_dop853() -> None:
    case = accuracy_case("sampled_gaussian_9")
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=case.families)) as engine,
    ):
        tight = engine.inspect_dispatch(case.source, (case.batch,), budget=0.999e-3)
        loose = engine.inspect_dispatch(case.source, (case.batch,), budget=1e-3)
        result = engine.evolve(case.source, (case.batch,), budget=1e-3, telemetry=True)

    costs = []
    for inspection in (tight, loose):
        prices = [
            candidate.scheduling_cost
            for row in inspection.rows
            for candidate in row.candidates
            if row.family == candidate.family == "dop853"
        ]
        assert len(prices) == len(case.batch.points)
        costs.append(sum(price for price in prices if price is not None))
    assert 0 < costs[1] < costs[0]
    routing = result.paths.routing["dop853"]
    assert routing.dp54_rows == len(case.batch.points)
    assert routing.estimated_cost == pytest.approx(costs[1])


@pytest.fixture
def stub_opencl(monkeypatch: pytest.MonkeyPatch) -> str:
    """Install a working device at the optional OpenCL runner boundary."""
    from zcu_tools.simulate.evolution.kernels import _dop853_opencl

    class DeviceStub:
        execution_device = "opencl:stub/driver/kernel/pyopencl"

        def check_device(self) -> None:
            pass

        def cached_build_failure(self, dimension: int, *, dp54: bool) -> None:
            return None

        def compile_variant(self, dimension: int, *, dp54: bool) -> None:
            return None

        def advance(
            self,
            data: object,
            row_models: object,
            states: np.ndarray,
            budget: float,
            *,
            dp54: bool,
        ) -> np.ndarray:
            # Encode the device's actual tableau argument in a normalized endpoint.
            endpoints = np.zeros_like(states)
            marker = 1 if dp54 else 0
            endpoints[:, marker, marker] = 1
            return endpoints

        def close(self) -> None:
            pass

    # The optional GPU numerical collaborator is outside the routing contract.
    # CPU CI has no public way to supply a device, so replace just its runner.
    monkeypatch.setattr(_dop853_opencl, "OpenCLDOP853", DeviceStub)
    return DeviceStub.execution_device


@pytest.mark.parametrize(
    ("budget", "dp54"),
    [
        pytest.param(0.999e-3, False, id="under-budget"),
        pytest.param(1e-3, True, id="inclusive-budget"),
    ],
)
@pytest.mark.parametrize("case_name", ["sampled_gaussian_9", "cpu_dp54_two_tone_960"])
def test_opencl_dop853_selects_tableau_by_budget(
    stub_opencl: str, budget: float, dp54: bool, case_name: str
) -> None:
    case = (
        accuracy_case(case_name)
        if case_name == "sampled_gaussian_9"
        else cpu_dp54_case(case_name)
    )
    with (
        closing(case.source),
        closing(Engine(1, backend="opencl", families=case.families)) as engine,
    ):
        result = engine.evolve(
            case.source, (case.batch,), budget=budget, telemetry=True
        )

    assert result.paths.execution_device == stub_opencl
    report = result.paths.routing["dop853"]
    assert report.opencl_rows == len(case.batch.points)
    assert report.dp54_rows == (report.opencl_rows if dp54 else 0)
    assert report.dop853_rows == (0 if dp54 else report.opencl_rows)
    # Telemetry alone can agree with the plan while the runner gets the wrong flag.
    marker = 1 if dp54 else 0
    np.testing.assert_array_equal(result.densities[:, marker, marker], 1)


def test_explicit_opencl_raises_during_construction_without_a_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Replace only the optional loader's discovery boundary; no driver needed.
    try:
        import pyopencl as cl
    except ImportError:
        monkeypatch.setitem(
            sys.modules, "pyopencl", SimpleNamespace(get_platforms=list)
        )
    else:
        monkeypatch.setattr(cl, "get_platforms", list)
    with pytest.raises(RuntimeError, match="cl_khr_fp64"):
        Engine(1, backend="opencl")


@pytest.mark.requires_opencl
def test_explicit_opencl_raises_during_construction_when_probe_build_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pyopencl as cl

    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))

    def reject_build(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected probe compiler rejection")

    monkeypatch.setattr(cl.Program, "build", reject_build)
    with (
        pytest.raises(RuntimeError, match="probe build failed.*compiler rejection"),
        closing(Engine(1, backend="opencl")),
    ):
        pass


def test_auto_probes_only_on_eligible_work_and_retains_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def no_platforms() -> list[object]:
        nonlocal attempts
        attempts += 1
        return []

    # Observe discovery at the third-party boundary, without requiring pyopencl.
    monkeypatch.setitem(
        sys.modules, "pyopencl", SimpleNamespace(get_platforms=no_platforms)
    )
    case = accuracy_case("sampled_gaussian_9")
    clipped = replace(
        case.batch,
        tracks=tuple(
            replace(track, duration_ns=track.duration_ns + 10)
            for track in case.batch.tracks
        ),
    )
    with (
        closing(Engine(1, backend="cpu", families=case.families)) as cpu,
        closing(Engine(2, families=case.families)) as auto,
    ):
        auto.evolve(case.source, (clipped,), telemetry=True)
        expected = cpu.evolve(case.source, (case.batch,), telemetry=True)
        assert attempts == 0
        for _ in range(2):
            actual = auto.evolve(case.source, (case.batch,), telemetry=True)
            assert actual.densities.tobytes() == expected.densities.tobytes()
            report = actual.paths.routing["dop853"]
            assert report.opencl_rows == 0
            assert report.opencl_detection_error is not None
            assert "cl_khr_fp64" in report.opencl_detection_error
    assert attempts == 1


@pytest.mark.requires_opencl
def test_auto_rejects_a_wrong_device_endpoint_and_remembers_the_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pyopencl as cl

    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))
    case = accuracy_case("sampled_gaussian_9")
    copy = cl.enqueue_copy

    def corrupt_download(
        queue: cl.CommandQueue,
        dest: np.ndarray | cl.Buffer,
        src: np.ndarray | cl.Buffer,
        *,
        is_blocking: bool,
    ) -> cl.Event:
        if isinstance(dest, cl.Buffer) and isinstance(src, np.ndarray):
            return copy(queue, dest, src, is_blocking=is_blocking)
        if isinstance(dest, np.ndarray) and isinstance(src, cl.Buffer):
            event = copy(queue, dest, src, is_blocking=is_blocking)
            if dest.dtype == np.complex64:
                dest.fill(0)
                dest[:, 0, 0] = 1
            return event
        raise AssertionError("expected host/device transfer")

    with closing(Engine(1, backend="cpu", families=case.families)) as cpu:
        expected = cpu.evolve(case.source, (case.batch,))
    with monkeypatch.context() as driver:
        driver.setattr(cl, "enqueue_copy", corrupt_download)
        with closing(Engine(1, families=case.families)) as auto:
            first = auto.evolve(case.source, (case.batch,), telemetry=True)
    with closing(Engine(1, families=case.families)) as restarted:
        second = restarted.evolve(case.source, (case.batch,), telemetry=True)
    for result in (first, second):
        report = result.paths.routing["dop853"]
        assert report.opencl_detection_error is not None
        assert "self-test" in report.opencl_detection_error
        assert report.opencl_rows == 0
        assert result.densities.tobytes() == expected.densities.tobytes()
    assert (
        first.paths.routing["dop853"].opencl_detection_error
        == second.paths.routing["dop853"].opencl_detection_error
    )


@pytest.mark.requires_opencl
def test_auto_keeps_gpu_when_the_detection_result_cannot_be_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))
    replace_file = Path.replace
    blocked = 0

    def reject_detection_write(path: Path, target: str | Path) -> Path:
        nonlocal blocked
        destination = Path(target)
        if destination.parent.name == "detection" and destination.is_relative_to(
            tmp_path
        ):
            blocked += 1
            raise OSError("injected detection cache write failure")
        return replace_file(path, target)

    source = coherent_source(synthetic_model(32))
    batch = _dense_batch(1)
    with closing(source), closing(Engine(2, families=("dop853",))) as auto:
        with monkeypatch.context() as filesystem:
            filesystem.setattr(Path, "replace", reject_detection_write)
            device = auto.detect_execution_device()
        assert blocked == 1
        assert device.startswith("opencl:")
        result = auto.evolve(source, (batch,), telemetry=True)
    assert result.paths.execution_device == device
    assert result.paths.routing["dop853"].opencl_rows == 1


def _dense_batch(count: int) -> TimelineBatch:
    dense = timeline(count, 100.0)
    knots = np.linspace(0, 100.0, 960)
    envelope = EnvelopeSpec(
        "sampled", np.sin(np.pi * knots / 100).astype(complex), knots
    )
    return replace(
        dense,
        tracks=(
            drive_track(dense, envelope, frequency=0.129, gain=0.003),
            drive_track(dense, envelope, frequency=0.17, gain=-0.007),
        ),
    )


def test_auto_reprices_the_dp54_cpu_alternative(stub_opencl: str) -> None:
    # For one dense 13-level endpoint the GPU launch/trial floor dominates.
    # Its price sits between the DOP853 and calibrated DP5(4) CPU prices.
    batch = _dense_batch(1)
    with (
        closing(coherent_source(synthetic_model(13))) as source,
        closing(Engine(1, families=("dop853",), backend="auto")) as engine,
    ):
        tight = engine.evolve(source, (batch,), budget=0.999e-3, telemetry=True)
        loose = engine.evolve(source, (batch,), budget=1e-3, telemetry=True)

    assert tight.paths.execution_device == "cpu"
    assert tight.paths.routing["dop853"].dop853_rows == 1
    assert loose.paths.execution_device == stub_opencl
    assert loose.paths.routing["dop853"].opencl_rows == 1


def test_auto_reprices_the_dp54_opencl_alternative(stub_opencl: str) -> None:
    # A sparse 20-level endpoint is cheaper on CPU with the old GPU price,
    # even with CPU DP5(4). The lower GPU compute price tips the comparison.
    case = cpu_dp54_case("cpu_dp54_short_sampled")
    with (
        closing(case.source),
        closing(coherent_source(synthetic_model(20))) as source,
        closing(Engine(1, families=case.families)) as engine,
    ):
        tight = engine.evolve(source, (case.batch,), budget=0.999e-3, telemetry=True)
        loose = engine.evolve(source, (case.batch,), budget=1e-3, telemetry=True)

    assert tight.paths.routing["dop853"].opencl_rows == 0
    assert loose.paths.execution_device == stub_opencl
    assert loose.paths.routing["dop853"].opencl_rows == 1


@pytest.mark.requires_opencl
def test_auto_prices_small_and_wide_dense_acquires(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))
    small = accuracy_case("sampled_gaussian_9")
    dense = _dense_batch(12)
    source = coherent_source(synthetic_model(32))
    with (
        closing(Engine(2, backend="cpu", families=small.families)) as cpu,
        closing(Engine(2, families=small.families)) as auto,
    ):
        small_cpu = cpu.evolve(small.source, (small.batch,))
        small_auto = auto.evolve(small.source, (small.batch,), telemetry=True)
        assert small_auto.paths.routing["dop853"].opencl_rows == 0
        assert small_auto.paths.routing["dop853"].opencl_detection_error is None
        assert small_auto.densities.tobytes() == small_cpu.densities.tobytes()
        dense_cpu = cpu.evolve(source, (dense,))
        dense_auto = auto.evolve(source, (dense,), telemetry=True)
        assert dense_auto.paths.routing["dop853"].opencl_rows == len(dense.points)
        assert density_error(dense_auto.densities, dense_cpu.densities) < 1e-3


@pytest.mark.requires_opencl
def test_opencl_price_does_not_depend_on_engine_or_disk_warmth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pyopencl as cl

    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))
    source = coherent_source(synthetic_model(32))
    narrow, wide = _dense_batch(1), _dense_batch(12)
    with closing(Engine(2, families=("dop853",))) as auto:
        cold = auto.evolve(source, (narrow,), telemetry=True)
        auto.evolve(source, (wide,))
        warm = auto.evolve(source, (narrow,), telemetry=True)
    with closing(Engine(2, families=("dop853",))) as restarted:
        again = restarted.evolve(source, (narrow,), telemetry=True)
    for result in (cold, warm, again):
        assert result.paths.routing["dop853"].opencl_rows == 1
        assert result.densities.tobytes() == cold.densities.tobytes()

    def bad_launch(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected GPU execution failure")

    monkeypatch.setattr(cl, "Kernel", bad_launch)
    with (
        closing(Engine(2, families=("dop853",))) as restarted,
        pytest.raises(RuntimeError, match="injected GPU execution failure"),
    ):
        restarted.evolve(source, (wide,))


@pytest.mark.requires_opencl
def test_auto_routing_and_density_do_not_depend_on_chunk_count() -> None:
    source = coherent_source(synthetic_model(32))
    batch = _dense_batch(12)
    with (
        closing(Engine(2, chunks_per_worker=1, families=("dop853",))) as coarse,
        closing(Engine(2, chunks_per_worker=6, families=("dop853",))) as fine,
    ):
        coarse_result = coarse.evolve(source, (batch,), telemetry=True)
        fine_result = fine.evolve(source, (batch,), telemetry=True)
    for result in (coarse_result, fine_result):
        assert result.paths.routing["dop853"].opencl_rows == len(batch.points)
    assert fine_result.paths.execution_device == coarse_result.paths.execution_device
    assert fine_result.densities.tobytes() == coarse_result.densities.tobytes()


@pytest.mark.requires_opencl
def test_auto_isolates_a_failing_dimension_tableau_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pyopencl as cl

    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))
    build = cl.Program.build
    failures = 0

    def build_device_program(
        program: cl.Program,
        options: str | Sequence[str] | None = None,
        devices: Sequence[cl.Device] | None = None,
        cache_dir: str | None = None,
    ) -> cl.Program:
        nonlocal failures
        defines = options.split() if isinstance(options, str) else options or ()
        if "-DD=32" in defines and "-DDP54=1" in defines:
            failures += 1
            raise RuntimeError("injected d32 DP5(4) compiler rejection")
        return build(program, options=options, devices=devices, cache_dir=cache_dir)

    monkeypatch.setattr(cl.Program, "build", build_device_program)
    source = coherent_source(synthetic_model(32))
    good_source = coherent_source(synthetic_model(24))
    batch = _dense_batch(12)
    with (
        closing(Engine(2, families=("dop853",))) as auto,
        closing(Engine(2, families=("dop853",), backend="cpu")) as cpu,
    ):
        expected = cpu.evolve(source, (batch,))
        declined = auto.evolve(source, (batch,), telemetry=True)
        tight = auto.evolve(source, (batch,), budget=0.999e-3, telemetry=True)
        supported = auto.evolve(good_source, (_dense_batch(24),), telemetry=True)
    with closing(Engine(2, families=("dop853",))) as restarted:
        cached = restarted.evolve(source, (batch,), telemetry=True)
    for result in (declined, cached):
        assert result.paths.execution_device == "cpu"
        assert result.densities.tobytes() == expected.densities.tobytes()
        report = result.paths.routing["dop853"]
        assert report.opencl_rows == 0
        assert report.opencl_detection_error is not None
        assert "dimension 32" in report.opencl_detection_error
        assert "injected d32 DP5(4) compiler rejection" in report.opencl_detection_error
    assert failures == 1
    tight_report = tight.paths.routing["dop853"]
    assert tight_report.opencl_rows == tight_report.dop853_rows == len(batch.points)
    assert tight_report.opencl_detection_error is None
    assert supported.paths.routing["dop853"].opencl_rows == 24
    assert supported.paths.routing["dop853"].opencl_detection_error is None
    with (
        closing(Engine(2, families=("dop853",), backend="opencl")) as forced,
        pytest.raises(RuntimeError, match=r"injected d32 DP5\(4\) compiler rejection"),
    ):
        forced.evolve(source, (batch,))


@pytest.mark.requires_opencl
def test_opencl_engine_close_releases_its_context() -> None:
    import pyopencl as cl

    def contexts() -> set[weakref.ReferenceType[cl.Context]]:
        return {
            weakref.ref(obj) for obj in gc.get_objects() if isinstance(obj, cl.Context)
        }

    case = accuracy_case("sampled_gaussian_9")
    before = contexts()
    with closing(Engine(1, backend="opencl", families=case.families)) as gpu:
        result = gpu.evolve(case.source, (case.batch,), telemetry=True)
        assert result.paths.routing["dop853"].opencl_rows == len(case.batch.points)
        created = contexts() - before
        assert len(created) == 1
    gc.collect()
    assert all(context() is None for context in created)


@pytest.mark.requires_opencl
def test_opencl_limits_the_union_of_both_tone_grids() -> None:
    case = accuracy_case("sampled_gaussian_9")
    # The grids share only their endpoints. Each tone stays well below the
    # limit, but their union reaches 960 knots and then exceeds it by one.
    grid = np.linspace(0, 100.0, 481)
    other = np.r_[0.0, (grid[:-2] + grid[1:-1]) / 2, 100.0]

    def sampled(times: np.ndarray) -> EnvelopeSpec:
        return EnvelopeSpec(
            "sampled", np.sin(np.pi * times / 100).astype(np.complex128), times
        )

    first, second = case.batch.tracks
    batches = tuple(
        replace(
            case.batch,
            points=case.batch.points + 2 * i,
            tracks=(
                replace(first, envelopes=(sampled(grid),)),
                replace(second, envelopes=(sampled(times),)),
            ),
        )
        for i, times in enumerate((other, np.r_[other[:-1], 99.9, 100.0]))
    )
    with (
        closing(Engine(1, backend="cpu", families=case.families)) as cpu,
        closing(Engine(1, backend="opencl", families=case.families)) as gpu,
    ):
        expected = cpu.evolve(case.source, batches)
        actual = gpu.evolve(case.source, batches, telemetry=True)
    assert actual.paths.routing["dop853"].opencl_rows == len(case.batch.points)
    assert actual.densities[2:].tobytes() == expected.densities[2:].tobytes()
    assert density_error(actual.densities[:2], expected.densities[:2]) < 1e-3


@pytest.mark.requires_opencl
def test_opencl_keeps_clipped_pulses_and_touching_dop853_chains_on_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pyopencl as cl

    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))
    case = accuracy_case("sampled_gaussian_9")
    full = case.batch
    clipped = replace(
        full,
        points=full.points + 2,
        tracks=tuple(replace(t, duration_ns=t.duration_ns + 10) for t in full.tracks),
    )
    split = replace(full, points=full.points + 4, segment_boundaries=((50.0,), (50.0,)))
    batches = (full, clipped, split)
    with (
        closing(Engine(2, backend="cpu", families=case.families)) as cpu,
        closing(Engine(2, backend="opencl", families=case.families)) as gpu,
    ):
        expected = cpu.evolve(case.source, batches)
        actual = gpu.evolve(case.source, batches, telemetry=True)
    assert actual.paths.segments_by_kernel == {"dop853": 8}
    assert actual.paths.routing["dop853"].opencl_rows == len(full.points)
    np.testing.assert_array_equal(actual.points, expected.points)
    assert actual.densities[2:].tobytes() == expected.densities[2:].tobytes()
    assert density_error(actual.densities[:2], expected.densities[:2]) < 1e-3
    assert expected.paths.execution_device == "cpu"
    identity = actual.paths.execution_device
    assert identity.startswith("opencl:")
    name, driver, kernel, binding = identity.removeprefix("opencl:").rsplit("/", 3)
    assert (name, driver) in {
        (device.name, device.driver_version)
        for platform in cl.get_platforms()
        for device in platform.get_devices(device_type=cl.device_type.GPU)
    }
    assert kernel
    assert binding == cl.VERSION_TEXT
    reused = actual.paths.for_reuse(telemetry=False)
    assert reused.execution_device == actual.paths.execution_device
    assert (
        merge_path_diagnostics((expected.paths, actual.paths, reused)).execution_device
        == actual.paths.execution_device
    )


def test_sampled_multitone_reuses_one_spot_check_only_within_evolve() -> None:
    case = accuracy_case("cheap_sampled_nearby")
    batch = replace(
        case.batch,
        tracks=tuple(
            replace(track, phase_rad=np.repeat(track.phase_rad[:1], 2))
            for track in case.batch.tracks
        ),
    )
    with closing(Engine(1, backend="cpu", families=("multitone_rwa",))) as engine:
        inspection = engine.inspect_dispatch(case.source, (batch,))
        assert len(inspection.rows) == 2
        for row in inspection.rows:
            assert row.family == "multitone_rwa"
            assert any(
                candidate.family == "multitone_rwa" and candidate.guard == "accepted"
                for candidate in row.candidates
            )
        for _ in range(2):
            result = engine.evolve(case.source, (batch,), telemetry=True)
            assert result.paths.segments_by_kernel == {"multitone_rwa": 2}
            assert result.paths.routing["multitone_rwa"].admission_rows == 1


def test_sampled_multitone_rejection_does_not_cross_phase_groups() -> None:
    case = accuracy_case("sampled_multitone_phase_sequence")
    batch = timeline(3, 8.0)
    batch = replace(
        batch,
        tracks=tuple(
            replace(
                drive_track(batch, track.envelopes[0], port=track.port),
                frequency_ghz=np.full(3, track.frequency_ghz[0]),
                phase_rad=np.array([track.phase_rad[0], track.phase_rad[0], -0.9]),
                amplitude=track.amplitude[[1, 0, 0]],
            )
            for track in case.batch.tracks[:2]
        ),
    )
    families = ("multitone_rwa", "dop853")
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=families)) as engine,
    ):
        inspection = engine.inspect_dispatch(case.source, (batch,))
        assert [row.family for row in inspection.rows] == [
            "dop853",
            "dop853",
            "multitone_rwa",
        ]
        result = engine.evolve(case.source, (batch,), telemetry=True)
    assert result.paths.segments_by_kernel == {"dop853": 2, "multitone_rwa": 1}
    routing = result.paths.routing["multitone_rwa"]
    assert routing.admission_rows == 2
    assert routing.rejected_rows == 2


def _short_sampled_pair(case: AccuracyCase, count: int) -> TimelineBatch:
    batch = timeline(count, 1.5)
    envelope = EnvelopeSpec(
        "sampled", np.array([0.8, 1.0, 0.9], complex), np.array([0.0, 0.75, 1.5])
    )
    return replace(
        batch,
        tracks=tuple(
            replace(
                drive_track(
                    batch,
                    envelope,
                    frequency=float(track.frequency_ghz[0]),
                    port=track.port,
                ),
                phase_rad=np.full(count, track.phase_rad[0]),
            )
            for track in case.batch.tracks
        ),
    )


def test_sampled_multitone_probes_each_tones_amplitude_maximum() -> None:
    case = accuracy_case("joint_constant")
    scan = _short_sampled_pair(case, 3)
    batch = replace(
        scan,
        tracks=tuple(
            replace(track, amplitude=np.array(gains))
            for track, gains in zip(
                scan.tracks, ((0.03, 0.015, 0.002), (0.015, 0.031, 0.002)), strict=True
            )
        ),
    )
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=("multitone_rwa",))) as engine,
    ):
        result = engine.evolve(case.source, (batch,), telemetry=True)
    assert result.paths.segments_by_kernel == {"multitone_rwa": 3}
    assert result.paths.routing["multitone_rwa"].admission_rows == 2


@pytest.mark.parametrize("variation", ["scale", "one_ulp_sample", "sample_grid"])
def test_sampled_multitone_requires_exact_proportional_envelopes(
    variation: str,
) -> None:
    case = accuracy_case("joint_constant")
    scan = _short_sampled_pair(case, 2)
    samples = np.array([0.5 + 0.125j, 1.0, 0.75 - 0.25j], complex)
    scaled = np.asarray(samples * 1.5, dtype=np.complex128)
    if variation == "one_ulp_sample":
        scaled.real[0] = np.nextafter(scaled.real[0], np.inf)
    knots = np.array([0.0, 0.75, 1.5])
    scaled_knots = np.array([0.0, 0.6, 1.5]) if variation == "sample_grid" else knots
    envelopes = (
        EnvelopeSpec("sampled", samples, knots),
        EnvelopeSpec("sampled", scaled, scaled_knots),
    )
    batch = replace(
        scan,
        tracks=tuple(
            replace(
                track,
                amplitude=np.array([0.01, 0.02]),
                envelopes=envelopes,
                envelope_index=np.arange(2),
            )
            for track in scan.tracks
        ),
    )
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=("multitone_rwa",))) as engine,
    ):
        result = engine.evolve(case.source, (batch,), telemetry=True)
    assert result.paths.segments_by_kernel == {"multitone_rwa": 2}
    assert result.paths.routing["multitone_rwa"].admission_rows == (
        1 if variation == "scale" else 2
    )


def test_sampled_multitone_does_not_group_different_pulse_slices() -> None:
    case = accuracy_case("joint_constant")
    scan = _short_sampled_pair(case, 2)
    first, second = scan.tracks
    # Both points resume at 0.5 ns and play a 1 ns pair at identical phases.
    # Only the second point's first tone is already 0.5 ns into its envelope.
    batch = replace(
        scan,
        start_ns=np.full(2, 0.5),
        initial_densities=np.repeat(case.source.initial_density[None], 2, axis=0),
        tracks=(
            replace(
                first, start_ns=np.array([0.5, 0.0]), duration_ns=np.array([1.0, 1.5])
            ),
            replace(second, start_ns=np.full(2, 0.5), duration_ns=np.ones(2)),
        ),
    )
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=("multitone_rwa",))) as engine,
    ):
        result = engine.evolve(case.source, (batch,), telemetry=True)
    assert result.paths.segments_by_kernel == {"multitone_rwa": 2}
    assert result.paths.routing["multitone_rwa"].admission_rows == 2


def _joint_gate_case(count: int) -> AccuracyCase:
    case = accuracy_case("joint_gate_inventory")
    batch = timeline(count, 1.5)
    tracks = tuple(
        replace(
            drive_track(
                batch,
                track.envelopes[0],
                frequency=float(track.frequency_ghz[0]),
                gain=float(track.amplitude[0]),
                port=track.port,
            ),
            phase_rad=np.full(count, track.phase_rad[0]),
        )
        for track in case.batch.tracks
    )
    return replace(case, batch=replace(batch, tracks=tracks))


@pytest.mark.parametrize("count", [19, 20])
def test_sampled_multitone_admission_is_independent_of_reuse(count: int) -> None:
    case = _joint_gate_case(count)
    reference = converged_reference(case.source, case.batch)
    with closing(Engine(1, backend="cpu", families=("multitone_rwa",))) as engine:
        result = engine.evolve(case.source, (case.batch,), telemetry=True)
    assert result.paths.segments_by_kernel == {"multitone_rwa": count}
    assert result.paths.routing["multitone_rwa"].admission_rows == 1
    assert density_error(result.densities, reference.density) <= 2e-3

    with closing(Engine(1, backend="cpu")) as engine:
        result = engine.evolve(case.source, (case.batch,))
    assert density_error(result.densities, reference.density) <= 2e-3


def test_pinned_multitone_gate_below_cost_break_even() -> None:
    """Pinning does not bypass the independent multitone_gate rejection."""
    case = _joint_gate_case(19)
    with closing(Engine(1, backend="cpu")) as engine:
        engine.evolve(case.source, (case.batch,))
    with (
        closing(Engine(1, backend="cpu", families=("multitone_gate",))) as engine,
        pytest.raises(NoValidKernel) as rejected,
    ):
        engine.evolve(case.source, (case.batch,))
    assert rejected.value.domain == "multiple"
    assert rejected.value.attempted == ("multitone_gate",)


def _carrier_case(frequency: float) -> AccuracyCase:
    # Near-resonant strength excludes weak-drive Dyson above the boundary.
    model = replace(linear_drive_model(), energies_ghz=np.array([0.0, 0.05]))
    batch = timeline(2, 1.0)
    batch = replace(
        batch,
        tracks=(
            drive_track(batch, EnvelopeSpec("constant"), frequency=frequency, gain=20),
        ),
    )
    return AccuracyCase("carrier_boundary", coherent_source(model), batch, ())


@pytest.mark.parametrize(
    "frequency,family",
    [
        (sign * carrier, family)
        for sign in (1, -1)
        for carrier, family in (
            (0.029999, "slow_envelope"),
            (0.03, "floquet_period"),
            (0.030001, "floquet_period"),
        )
    ],
)
def test_default_absolute_carrier_boundary(frequency: float, family: str) -> None:
    case = _carrier_case(frequency)
    reference = converged_reference(case.source, case.batch)
    with closing(Engine(1)) as engine:
        result = engine.evolve(case.source, (case.batch,))
    assert dict(result.paths.segments_by_kernel) == {family: 2}
    assert density_error(result.densities, reference.density) <= 2e-3


def test_pinned_slow_envelope_rejects_exact_carrier_boundary() -> None:
    for frequency in (0.03, -0.03):
        case = _carrier_case(frequency)
        with (
            closing(Engine(1, families=("slow_envelope",))) as engine,
            pytest.raises(RuntimeError, match=r"domain plateau: points \[0, 1\]"),
        ):
            engine.evolve(case.source, (case.batch,))
