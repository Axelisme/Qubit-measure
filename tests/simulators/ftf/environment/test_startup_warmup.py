"""Startup preparation is observable and does not consume acquisition identity."""

import sys
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import replace
from importlib.machinery import ModuleSpec
from pathlib import Path
from threading import Event, Thread
from types import ModuleType

import numpy as np
import pytest
from tqdm.auto import tqdm
from zcu_tools.experiment import simulation
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.simulate.adc import ConditionalReadout, ConditionalSums, ReadoutBatch
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec
from zcu_tools.simulators.ftf.config import parse_ftf
from zcu_tools.simulators.ftf.environment import ftf
from zcu_tools.simulators.ftf.environment import warmup as ftf_warmup

from tests.program._profiles import profile_soccfg
from tests.program._timeline import compile_modules
from tests.simulate.evolution.contract.accuracy_cases import (
    coherent_source,
    drive_track,
    timeline,
)
from tests.simulate.evolution.contract.synthetic_models import synthetic_model
from tests.simulators.ftf.config._data import ftf_document

pytestmark = pytest.mark.sim


def warmup_type():
    assert hasattr(simulation, "EnvironmentWarmup"), (
        "environment needs observable startup warmup"
    )
    return simulation.EnvironmentWarmup


def program(env):
    readout = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=6000.0,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.02),
        ro_length=0.02,
    )
    return compile_modules(
        env.soccfg,
        modules=[readout.build("ro")],
        reps=3,
    )


@pytest.fixture
def isolated_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    # The public interface has no reset for process-wide single-flight state.
    monkeypatch.setattr(ftf_warmup, "_startup", {})
    monkeypatch.setattr(ftf_warmup, "_cpu_prepared", set())
    monkeypatch.setattr(ftf_warmup, "_cpu_failures", {})
    # GPU tests need no CPU workloads unless they check the CPU/GPU handoff.
    monkeypatch.setattr(ftf_warmup, "COVERAGE", ())


def install_fake_opencl(
    monkeypatch: pytest.MonkeyPatch, get_platforms: Callable[[], list[object]]
) -> None:
    pyopencl = ModuleType("pyopencl")
    pyopencl.__spec__ = ModuleSpec("pyopencl", loader=None)
    pyopencl.__dict__["get_platforms"] = get_platforms
    monkeypatch.setitem(sys.modules, "pyopencl", pyopencl)


def test_invalid_warmup_dimension_is_rejected():
    with pytest.raises(ValueError, match="dimension must be at least one"):
        ftf_warmup.startup_warmup(0, families=("t4",))


def test_gpu_warmup_does_not_block_construction_or_duplicate_discovery(
    monkeypatch: pytest.MonkeyPatch,
    isolated_startup: None,
):
    entered, release = Event(), Event()
    calls = 0

    def no_gpu() -> list[object]:
        nonlocal calls
        calls += 1
        entered.set()
        # The constructor caller releases discovery after both builds return.
        assert release.wait(30), "test watchdog: release device discovery"
        return []

    install_fake_opencl(monkeypatch, no_gpu)
    settings = parse_ftf(ftf_document())
    soccfg = profile_soccfg("ftf")

    with ExitStack() as environments:
        try:
            first = environments.enter_context(
                simulation.closing_environment(
                    ftf(settings, soccfg=soccfg, warmup=True)
                )
            )
            assert first.warmup is not None
            # Construction starts the worker, whose first item calls discovery.
            assert entered.wait(30), "test watchdog: startup must reach discovery"
            second = environments.enter_context(
                simulation.closing_environment(
                    ftf(settings, soccfg=soccfg, warmup=True)
                )
            )
            assert second.warmup is first.warmup
            with pytest.raises(TimeoutError):
                first.wait_ready(timeout=0)
        finally:
            release.set()
        # Released discovery returns no devices and the only item finishes.
        status = first.wait_ready(timeout=30)
    assert status is not None and status.state == "ready"
    assert status.completed == ("dop853_opencl",)
    assert calls == 1


def test_warmup_skips_gpu_without_the_extra(
    monkeypatch: pytest.MonkeyPatch,
    isolated_startup: None,
) -> None:
    monkeypatch.setitem(sys.modules, "pyopencl", None)
    warmup = ftf_warmup.startup_warmup(8, families=None)
    # With an empty CPU manifest and no GPU item, the worker finishes immediately.
    status = warmup.wait_ready(timeout=30)
    assert status.state == "ready"
    assert not status.completed


def test_warmup_skips_gpu_when_dop853_is_excluded(
    monkeypatch: pytest.MonkeyPatch,
    isolated_startup: None,
) -> None:
    def unexpected_discovery() -> list[object]:
        pytest.fail("excluded dop853 must not discover an OpenCL device")

    install_fake_opencl(monkeypatch, unexpected_discovery)
    warmup = ftf_warmup.startup_warmup(8, families=("t4",))
    # Excluding dop853 leaves the worker no items to run.
    status = warmup.wait_ready(timeout=30)
    assert status.state == "ready"
    assert not status.completed


def test_cpu_failure_stops_gpu_warmup_across_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    isolated_startup: None,
) -> None:
    discoveries = cpu_calls = 0

    def no_gpu() -> list[object]:
        nonlocal discoveries
        discoveries += 1
        return []

    def reject_readout(
        readout: ConditionalReadout, batch: ReadoutBatch
    ) -> ConditionalSums:
        nonlocal cpu_calls
        cpu_calls += 1
        raise ValueError("synthetic CPU readout failure")

    install_fake_opencl(monkeypatch, no_gpu)
    monkeypatch.setattr(ftf_warmup, "COVERAGE", ("readout_integrated",))
    monkeypatch.setattr(ConditionalReadout, "responses", reject_readout)
    for dimension in (8, 9):
        warmup = ftf_warmup.startup_warmup(dimension, families=None)
        # The readout boundary rejects immediately (or reuses that failure).
        status = warmup.wait_ready(timeout=30)
        assert status.state == "failed"
        assert status.error == "ValueError: synthetic CPU readout failure"
        assert not status.completed
    assert cpu_calls == 1
    assert discoveries == 0


@pytest.mark.requires_opencl
def test_gpu_warmup_prepares_each_dimension_and_shares_cpu_work(
    monkeypatch: pytest.MonkeyPatch,
    isolated_startup: None,
    tmp_path: Path,
) -> None:
    import pyopencl as cl

    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ftf_warmup, "COVERAGE", ("readout_integrated",))
    responses = ConditionalReadout.responses
    cpu_calls = 0

    def count_readout(
        readout: ConditionalReadout, batch: ReadoutBatch
    ) -> ConditionalSums:
        nonlocal cpu_calls
        cpu_calls += 1
        return responses(readout, batch)

    builds: Counter[str] = Counter()
    build = cl.Program.build

    def count_build(
        program: cl.Program,
        options: str | Sequence[str] | None = None,
        devices: Sequence[cl.Device] | None = None,
        cache_dir: str | None = None,
    ) -> cl.Program:
        defines = options.split() if isinstance(options, str) else options or ()
        builds.update(define for define in defines if define.startswith("-DD="))
        return build(program, options=options, devices=devices, cache_dir=cache_dir)

    monkeypatch.setattr(ConditionalReadout, "responses", count_readout)
    monkeypatch.setattr(cl.Program, "build", count_build)
    first = ftf_warmup.startup_warmup(8, families=None)
    second = ftf_warmup.startup_warmup(9, families=None)
    assert first is not second
    for warmup in (first, second):
        # Each controller runs a finite solve with no parked collaborators.
        status = warmup.wait_ready(timeout=30)
        assert status.state == "ready"
        assert status.completed == ("readout_integrated", "dop853_opencl")
    assert cpu_calls == 1
    assert builds["-DD=8"] == builds["-DD=9"] == 1


@pytest.mark.requires_opencl
def test_gpu_warmup_builds_the_dimension_before_acquire(
    monkeypatch: pytest.MonkeyPatch,
    isolated_startup: None,
    tmp_path: Path,
) -> None:
    import pyopencl as cl

    cache = tmp_path / "cache"
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(cache))
    progressed, release = Event(), Event()
    settings = parse_ftf(ftf_document())
    dimension = settings.numerics.truncation.retained_levels
    build = cl.Program.build
    variant_builds: list[Path] = []

    def files() -> set[Path]:
        return {path for path in cache.rglob("*") if path.is_file()}

    def pause_build(
        program: cl.Program,
        options: str | Sequence[str] | None = None,
        devices: Sequence[cl.Device] | None = None,
        cache_dir: str | None = None,
    ) -> cl.Program:
        defines = options.split() if isinstance(options, str) else options or ()
        if f"-DD={dimension}" in defines:
            assert cache_dir is not None
            variant_builds.append(Path(cache_dir))
            progressed.set()
            # The test releases the build after taking the pre-build snapshot.
            assert release.wait(30), "test watchdog: release the OpenCL compiler"
        return build(program, options=options, devices=devices, cache_dir=cache_dir)

    monkeypatch.setattr(cl.Program, "build", pause_build)
    with simulation.closing_environment(
        ftf(settings, soccfg=profile_soccfg("ftf"), warmup=True)
    ) as env:
        try:
            assert env.warmup is not None, "expected background GPU preparation"
            env.warmup.start().add_done_callback(lambda _: progressed.set())
            # The worker either reaches Program.build or publishes terminal status.
            assert progressed.wait(30), "test watchdog: startup must advance"
            assert len(variant_builds) == 1, "startup must build for retained_levels"
            before_build = files()
        finally:
            release.set()
        # The released compiler and the finite synthetic solve finish the item.
        status = env.wait_ready(timeout=30)
        assert status is not None and status.state == "ready"
        prepared = files()
        (variant_dir,) = variant_builds
        assert any(
            path.is_relative_to(variant_dir)
            and path.name == "binary"
            and path.stat().st_size
            for path in prepared - before_build
        ), "startup must persist the retained_levels binary in the configured cache"
        program(env).acquire(env.soc, progress=False)
        assert files() == prepared

    # Use a fresh Engine with eligible work to check reuse of the persisted
    # program rather than resident programs.
    batch = timeline(16, 120.0)
    times = np.linspace(0, 120.0, 129)
    envelope = EnvelopeSpec(
        "sampled", np.sin(np.pi * times / 120).astype(complex), times
    )
    batch = replace(
        batch,
        tracks=(
            drive_track(batch, envelope, frequency=0.129, gain=0.003),
            drive_track(batch, envelope, frequency=0.17, gain=-0.007),
        ),
    )
    with closing(Engine(1, families=("dop853",))) as engine:
        result = engine.evolve(
            coherent_source(synthetic_model(dimension)), (batch,), telemetry=True
        )
    assert result.paths.routing["dop853"].opencl_rows == len(batch.points)
    assert files() == prepared


@pytest.mark.requires_opencl
def test_failed_gpu_build_reports_error_but_allows_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    isolated_startup: None,
    tmp_path: Path,
) -> None:
    import pyopencl as cl

    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))
    settings = parse_ftf(ftf_document())
    dimension = settings.numerics.truncation.retained_levels
    build = cl.Program.build
    reason = "synthetic OpenCL compiler rejection"

    def reject_variant(
        program: cl.Program,
        options: str | Sequence[str] | None = None,
        devices: Sequence[cl.Device] | None = None,
        cache_dir: str | None = None,
    ) -> cl.Program:
        defines = options.split() if isinstance(options, str) else options or ()
        if f"-DD={dimension}" in defines:
            raise RuntimeError(reason)
        return build(program, options=options, devices=devices, cache_dir=cache_dir)

    monkeypatch.setattr(cl.Program, "build", reject_variant)
    with simulation.closing_environment(
        ftf(settings, soccfg=profile_soccfg("ftf"), warmup=True)
    ) as env:
        # The worker encounters the injected rejection and finishes the manifest.
        status = env.wait_ready(timeout=30)
        assert status is not None
        assert status.state == "failed"
        assert "dop853_opencl" not in status.completed
        assert status.error is not None and reason in status.error
        program(env).acquire(env.soc, progress=False)
        assert env.soc.binding.last_request is not None
        assert env.soc.binding.last_request.sequence == 0


def test_start_is_single_flight_and_progress_is_observable():
    cls = warmup_type()
    entered, release = Event(), Event()
    calls = []

    def first():
        calls.append("first")

    def second():
        entered.set()
        # The test releases the worker after observing the active item.
        assert release.wait(30), "test watchdog: main thread must release warmup"
        calls.append("second")

    warmup = cls((("first", first), ("second", second)))
    try:
        future = warmup.start()
        # The first item only appends, so the worker proceeds to the second.
        assert entered.wait(30), "test watchdog: second item must start"
        assert warmup.start() is future
        status = warmup.status
        assert status.state == "running"
        assert status.completed == ("first",)
        assert status.current == "second"
        assert status.total == 2
        assert status.error is None
    finally:
        release.set()
        # The released second item only appends before the worker finishes.
        result = warmup.wait_ready(timeout=30)
    assert result.state == "ready"
    assert warmup.status.completed == ("first", "second")
    assert calls == ["first", "second"]


def test_acquire_waits_for_startup_future_before_consuming_sequence():
    cls = warmup_type()
    entered, release, waiting = Event(), Event(), Event()

    class ObservedWarmup(cls):
        def wait_ready(self, timeout=None):
            waiting.set()
            return super().wait_ready(timeout)

    def prepare():
        entered.set()
        # The test releases preparation after observing acquire at readiness.
        assert release.wait(30), "test watchdog: main thread must release warmup"

    warmup = ObservedWarmup((("synthetic", prepare),))
    with (
        simulation.closing_environment(
            ftf(parse_ftf(ftf_document()), soccfg=profile_soccfg("ftf"), warmup=warmup)
        ) as env,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        try:
            prog = program(env)
            # The factory starts the one-item manifest, which signals entry.
            assert entered.wait(30), "test watchdog: startup must start"
            assert env.warmup is warmup
            shared = warmup.start()
            with pytest.raises(TimeoutError):
                env.wait_ready(timeout=0)
            waiting.clear()

            acquisition = executor.submit(prog.acquire, env.soc, progress=False)
            # The free binding admits the only caller and calls wait_ready.
            assert waiting.wait(30), (
                "test watchdog: acquire must reach readiness boundary"
            )
            assert not acquisition.done()
            assert env.soc.binding.last_request is None
        finally:
            release.set()
        # Releasing preparation opens the binding; the finite acquire can finish.
        acquisition.result(timeout=30)
        assert shared is warmup.start()
        assert env.soc.binding.last_request is not None
        assert env.soc.binding.last_request.sequence == 0
        assert warmup.status.state == "ready"


def test_close_backend_waits_for_pending_preparation() -> None:
    entered, release, waiting, progressed = Event(), Event(), Event(), Event()

    class ObservedWarmup(simulation.EnvironmentWarmup):
        def wait_ready(self, timeout: float | None = None) -> simulation.WarmupStatus:
            waiting.set()
            progressed.set()
            return super().wait_ready(timeout)

    def prepare() -> None:
        entered.set()
        # The test releases preparation after observing the close attempt.
        assert release.wait(30), "test watchdog: release preparation"

    env = ftf(
        parse_ftf(ftf_document()),
        soccfg=profile_soccfg("ftf"),
        warmup=ObservedWarmup((("synthetic", prepare),)),
    )
    try:
        close_backend = env.close_backend
        assert close_backend is not None
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                # The single startup item signals entry before it parks.
                assert entered.wait(30), "test watchdog: startup must start"
                closed = executor.submit(close_backend)
                closed.add_done_callback(lambda _: progressed.set())
                # The close call either reaches wait_ready or returns.
                assert progressed.wait(30), "test watchdog: close must advance"
                assert waiting.is_set(), "backend close must wait for startup work"
                assert not closed.done()
            finally:
                release.set()
            # Released preparation completes so backend close can finish.
            closed.result(timeout=30)
        assert env.warmup is not None and env.warmup.status.state == "ready"
    finally:
        release.set()
        with simulation.closing_environment(env):
            pass


def test_failed_warmup_reports_error_but_allows_acquisition():
    cls = warmup_type()

    def broken():
        raise ValueError("synthetic compilation failure")

    warmup = cls((("broken", broken),))
    env = ftf(parse_ftf(ftf_document()), soccfg=profile_soccfg("ftf"), warmup=warmup)
    try:
        # The only item raises immediately and publishes failed status.
        status = env.wait_ready(timeout=30)
        assert status is not None
        assert status.state == "failed"
        assert status.current is None
        assert status.error is not None
        assert "synthetic compilation failure" in status.error
        program(env).acquire(env.soc, progress=False)
        assert env.soc.binding.last_request is not None
        assert env.soc.binding.last_request.sequence == 0
    finally:
        env.devices.close_all_devices()


def test_worker_start_failure_is_terminal_and_allows_on_demand(monkeypatch):
    cls = warmup_type()
    warmup = cls((("synthetic", lambda: None),))

    def unavailable_thread(self):
        raise RuntimeError("cannot start synthetic thread")

    with monkeypatch.context() as patch:
        # A global Thread.start failure would also make tqdm disable its monitor globally.
        patch.setattr(tqdm, "monitor_interval", 0)
        patch.setattr(Thread, "start", unavailable_thread)
        try:
            env = ftf(
                parse_ftf(ftf_document()), soccfg=profile_soccfg("ftf"), warmup=warmup
            )
        except RuntimeError as error:
            pytest.fail(
                f"worker startup failure must allow on-demand acquisition: {error}"
            )
    try:
        status = env.wait_ready(timeout=0)
        assert status is not None
        assert status.state == "failed"
        assert status.error is not None
        assert "cannot start synthetic thread" in status.error
        future = warmup.start()
        assert future.done()
        assert future.result().state == "failed"
        program(env).acquire(env.soc, progress=False)
        assert env.soc.binding.last_request is not None
        assert env.soc.binding.last_request.sequence == 0
    finally:
        env.devices.close_all_devices()


def test_warmup_does_not_change_rng_or_live_bias():
    cls = warmup_type()
    settings = parse_ftf(ftf_document())
    prepared = ftf(
        settings,
        soccfg=profile_soccfg("ftf"),
        warmup=cls((("synthetic", lambda: None),)),
    )
    plain = ftf(settings, soccfg=profile_soccfg("ftf"))
    try:
        # The no-op startup item returns without waiting on a collaborator.
        prepared.wait_ready(timeout=30)
        assert prepared.devices.get_info("bias") == plain.devices.get_info("bias")
        a, b = program(prepared), program(plain)
        a.acquire(prepared.soc, progress=False)
        b.acquire(plain.soc, progress=False)
        import numpy as np

        raw_a, raw_b = a.get_raw(), b.get_raw()
        assert raw_a is not None and raw_b is not None
        for x, y in zip(raw_a, raw_b, strict=True):
            np.testing.assert_array_equal(x, y)
        assert prepared.soc.binding.last_request is not None
        assert plain.soc.binding.last_request is not None
        assert (
            prepared.soc.binding.last_request.sequence
            == plain.soc.binding.last_request.sequence
            == 0
        )
    finally:
        prepared.devices.close_all_devices()
        plain.devices.close_all_devices()
