"""Startup preparation is observable and does not consume acquisition identity."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

import pytest
from tqdm.auto import tqdm
from zcu_tools.experiment import simulation
from zcu_tools.experiment.ftf import ftf
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.resources.ftf import parse_ftf

from tests.resources._ftf_data import ftf_document


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
    return ModularProgramV2(
        env.soccfg, ProgramV2Cfg(reps=3), modules=[readout.build("ro")]
    )


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
    env = ftf(parse_ftf(ftf_document()), warmup=warmup)
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


def test_failed_warmup_reports_error_but_allows_acquisition():
    cls = warmup_type()

    def broken():
        raise ValueError("synthetic compilation failure")

    warmup = cls((("broken", broken),))
    env = ftf(parse_ftf(ftf_document()), warmup=warmup)
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
            env = ftf(parse_ftf(ftf_document()), warmup=warmup)
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
    prepared = ftf(settings, warmup=cls((("synthetic", lambda: None),)))
    plain = ftf(settings)
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
