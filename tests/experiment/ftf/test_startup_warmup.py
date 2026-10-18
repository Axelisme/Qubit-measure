"""Startup preparation is observable and does not consume acquisition identity."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from zcu_tools.experiment import simulation
from zcu_tools.experiment.ftf import ftf
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.resources.ftf import parse_ftf

from tests.resources._ftf_data import ftf_document


def warmup_type():
    assert hasattr(simulation, "EnvironmentWarmup"), "environment needs observable startup warmup"
    return simulation.EnvironmentWarmup


def program(env):
    readout = PulseReadoutCfg(
        gen_ch=0, ro_ch=0, nqz=1, freq=6000.0, gain=0.1,
        waveform=ConstWaveformCfg(length=0.02), ro_length=0.02,
    )
    return ModularProgramV2(env.soccfg, ProgramV2Cfg(reps=3), modules=[readout.build("ro")])


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
    entered, release, attempted = Event(), Event(), Event()

    def prepare():
        entered.set()
        assert release.wait(30), "test watchdog: main thread must release warmup"

    warmup = cls((("synthetic", prepare),))
    env = ftf(parse_ftf(ftf_document()), warmup=warmup)
    prog = program(env)
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            assert entered.wait(30), "test watchdog: startup must start"
            assert env.warmup is warmup
            shared = warmup.start()
            with pytest.raises(TimeoutError):
                env.wait_ready(timeout=0)

            def acquire():
                attempted.set()
                return prog.acquire(env.soc, progress=False)

            acquisition = executor.submit(acquire)
            assert attempted.wait(30), "test watchdog: acquire caller must start"
            # No public queue-arrival hook exists. This short negative window only
            # detects a bypass; final sequence/output assertions are deterministic.
            assert not acquisition.done()
            assert env.soc.binding.last_request is None
        finally:
            release.set()
        acquisition.result(timeout=30)
        assert shared is warmup.start()
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
        assert status.state == "failed"
        assert status.current is None
        assert "synthetic compilation failure" in status.error
        program(env).acquire(env.soc, progress=False)
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

        for x, y in zip(a.get_raw(), b.get_raw(), strict=True):
            np.testing.assert_array_equal(x, y)
        assert prepared.soc.binding.last_request.sequence == plain.soc.binding.last_request.sequence == 0
    finally:
        prepared.devices.close_all_devices()
        plain.devices.close_all_devices()
