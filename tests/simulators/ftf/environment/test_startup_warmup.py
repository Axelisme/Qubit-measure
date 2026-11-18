"""Startup preparation is observable and does not consume acquisition identity."""

import logging
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
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
    # Restore the process-wide registries after each built-in warmup test.
    monkeypatch.setattr(ftf_warmup, "_startup", {})
    monkeypatch.setattr(ftf_warmup, "_cpu_prepared", {})
    # The GPU seam needs no CPU manifest; the measurement exercises all CPU items.
    monkeypatch.setattr(ftf_warmup, "COVERAGE", ())


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
        assert release.wait(15), "test watchdog: release device discovery"
        return []

    # OpenCL discovery is the external boundary; the worker and factory run normally.
    pyopencl = ModuleType("pyopencl")
    pyopencl.__spec__ = ModuleSpec("pyopencl", loader=None)
    pyopencl.__dict__["get_platforms"] = no_gpu
    monkeypatch.setitem(sys.modules, "pyopencl", pyopencl)
    settings = parse_ftf(ftf_document())
    soccfg = profile_soccfg("ftf")

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_build = executor.submit(ftf, settings, soccfg=soccfg, warmup=True)
        try:
            first = first_build.result(timeout=10)
            assert first.warmup is not None, (
                "GPU warmup must be started at construction"
            )
            assert entered.wait(10)
            second = executor.submit(ftf, settings, soccfg=soccfg, warmup=True).result(
                timeout=10
            )
            try:
                assert second.warmup is first.warmup
                with pytest.raises(TimeoutError):
                    first.wait_ready(timeout=0)
            finally:
                second.devices.close_all_devices()
        finally:
            release.set()
            first = first_build.result()
            try:
                status = first.wait_ready()
            finally:
                first.devices.close_all_devices()
    assert status is not None and status.state == "ready"
    assert calls == 1


@pytest.mark.parametrize("has_extra", [False, True])
def test_gpu_only_warmup_is_inactive_when_no_extra_or_no_dop853(
    monkeypatch: pytest.MonkeyPatch,
    isolated_startup: None,
    has_extra: bool,
) -> None:
    pyopencl = ModuleType("pyopencl")
    pyopencl.__spec__ = ModuleSpec("pyopencl", loader=None)
    monkeypatch.setitem(sys.modules, "pyopencl", pyopencl if has_extra else None)
    settings = parse_ftf(ftf_document())
    families = ("t4",) if has_extra else None
    with simulation.closing_environment(
        ftf(settings, soccfg=profile_soccfg("ftf"), warmup=True, families=families)
    ) as env:
        status = env.wait_ready()
        assert status is not None and status.state == "ready"
        assert not status.completed


@pytest.mark.requires_opencl
def test_gpu_warmup_builds_the_dimension_before_acquire(
    monkeypatch: pytest.MonkeyPatch,
    isolated_startup: None,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import pyopencl as cl

    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path / "opencl"))
    compiling, progressed, release, waiting = Event(), Event(), Event(), Event()
    settings = parse_ftf(ftf_document())
    dimension = settings.numerics.truncation.retained_levels
    build = cl.Program.build
    wait_ready = simulation.EnvironmentWarmup.wait_ready

    def pause_build(
        program: cl.Program,
        options: str | Sequence[str] | None = None,
        devices: Sequence[cl.Device] | None = None,
        cache_dir: str | None = None,
    ) -> cl.Program:
        defines = options.split() if isinstance(options, str) else options or ()
        if f"-DD={dimension}" in defines:
            compiling.set()
            progressed.set()
            assert release.wait(30), "test watchdog: release the OpenCL compiler"
        return build(program, options=options, devices=devices, cache_dir=cache_dir)

    def observe_wait(
        warmup: simulation.EnvironmentWarmup, timeout: float | None = None
    ) -> simulation.WarmupStatus:
        waiting.set()
        return wait_ready(warmup, timeout)

    monkeypatch.setattr(cl.Program, "build", pause_build)
    # Observe the public readiness boundary: acquisition has no wait-state field.
    monkeypatch.setattr(simulation.EnvironmentWarmup, "wait_ready", observe_wait)
    caplog.set_level(logging.DEBUG, logger="pyopencl.cache")

    def acquire(env: simulation.SimulationEnvironment) -> int:
        with env.soc.binding.acquire() as sequence:
            return sequence

    with (
        simulation.closing_environment(
            ftf(settings, soccfg=profile_soccfg("ftf"), warmup=True)
        ) as env,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        try:
            assert env.warmup is not None, "expected background GPU preparation"
            env.warmup.start().add_done_callback(lambda _: progressed.set())
            assert progressed.wait(30), "test watchdog: startup must advance"
            assert compiling.is_set(), "startup must build for retained_levels"
            caplog.clear()  # Exclude the small device self-test's program.
            admission = executor.submit(acquire, env)
            assert waiting.wait(10), "acquire must reach the readiness boundary"
            assert not admission.done()
        finally:
            release.set()
        assert admission.result(timeout=10) == 0
        status = env.wait_ready()
        assert status is not None and status.state == "ready"
        assert caplog.text.count("from-source build complete") == 1

    # Use a fresh Engine with eligible work to check reuse of the persisted
    # binary, not resident programs. The optional runtime reports cache hits.
    caplog.clear()
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
    assert "binary cache hit" in caplog.text
    assert "from-source build complete" not in caplog.text


def test_start_is_single_flight_and_progress_is_observable():
    cls = warmup_type()
    entered, release = Event(), Event()
    calls = []

    def first():
        calls.append("first")

    def second():
        entered.set()
        assert release.wait(30), "test watchdog: main thread must release warmup"
        calls.append("second")

    warmup = cls((("first", first), ("second", second)))
    try:
        future = warmup.start()
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
        warmup.wait_ready()
    assert future.result().state == "ready"
    assert warmup.status.completed == ("first", "second")
    assert calls == ["first", "second"]


def test_acquire_waits_for_startup_future_before_consuming_sequence():
    cls = warmup_type()
    entered, release, attempted, waiting = Event(), Event(), Event(), Event()

    class ObservedWarmup(cls):
        def wait_ready(self, timeout=None):
            waiting.set()
            return super().wait_ready(timeout)

    def prepare():
        entered.set()
        assert release.wait(30), "test watchdog: main thread must release warmup"

    warmup = ObservedWarmup((("synthetic", prepare),))
    env = ftf(parse_ftf(ftf_document()), soccfg=profile_soccfg("ftf"), warmup=warmup)
    prog = program(env)
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            assert entered.wait(30), "test watchdog: startup must start"
            assert env.warmup is warmup
            shared = warmup.start()
            with pytest.raises(TimeoutError):
                env.wait_ready(timeout=0)
            waiting.clear()

            def acquire():
                attempted.set()
                return prog.acquire(env.soc, progress=False)

            acquisition = executor.submit(acquire)
            assert attempted.wait(30), "test watchdog: acquire caller must start"
            assert waiting.wait(30), (
                "test watchdog: acquire must reach readiness boundary"
            )
            assert not acquisition.done()
            assert env.soc.binding.last_request is None
        finally:
            release.set()
        acquisition.result(timeout=30)
        assert shared is warmup.start()
        assert env.soc.binding.last_request is not None
        assert env.soc.binding.last_request.sequence == 0
        assert warmup.status.state == "ready"
    env.devices.close_all_devices()


def test_close_backend_waits_for_pending_preparation() -> None:
    entered, release, waiting, progressed = Event(), Event(), Event(), Event()

    class ObservedWarmup(simulation.EnvironmentWarmup):
        def wait_ready(self, timeout: float | None = None) -> simulation.WarmupStatus:
            waiting.set()
            progressed.set()
            return super().wait_ready(timeout)

    def prepare() -> None:
        entered.set()
        assert release.wait(15), "test watchdog: release preparation"

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
                assert entered.wait(10)
                closed = executor.submit(close_backend)
                closed.add_done_callback(lambda _: progressed.set())
                assert progressed.wait(10)
                assert waiting.is_set(), "backend close must wait for startup work"
                assert not closed.done()
            finally:
                release.set()
            closed.result(timeout=10)
        assert env.warmup is not None and env.warmup.status.state == "ready"
    finally:
        release.set()
        env.wait_ready()
        env.devices.close_all_devices()


def test_failed_warmup_reports_error_but_allows_acquisition():
    cls = warmup_type()

    def broken():
        raise ValueError("synthetic compilation failure")

    warmup = cls((("broken", broken),))
    env = ftf(parse_ftf(ftf_document()), soccfg=profile_soccfg("ftf"), warmup=warmup)
    try:
        status = env.wait_ready()
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
        prepared.wait_ready()
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
