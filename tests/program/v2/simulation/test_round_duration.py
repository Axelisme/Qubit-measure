"""Physical round pacing at the public binding seam with virtual time."""

import time
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.core.simulation import (
    AcquireRequest,
    EnvironmentSnapshot,
    RoundSignals,
    TimelineTemplate,
)
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2, make_mock_soccfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import (
    SimulationBinding,
    SimulationClock,
    SimulationSoc,
    SimulationTiming,
)
from zcu_tools.program.v2.simulation.frontend import build_request


class Clock:
    """Monotonic virtual seconds; sleeping advances without blocking."""

    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        assert duration > 0
        self.waits.append(duration)
        self.now += duration


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    return clock


def request_for(binding: SimulationBinding, sequence: int) -> AcquireRequest:
    """Three points in two structural classes, totaling 20 ms per repetition."""
    return AcquireRequest(
        "decimated",
        (10, 3),
        0,
        (),
        1,
        sequence,
        binding.rng_version,
        point_dims=(3,),
        point_names=("scan",),
        reps=10,
        templates=(
            TimelineTemplate(np.array([0, 2]), (), (), np.array([2000.0, 15000.0])),
            TimelineTemplate(np.array([1]), (), (), np.array([3000.0])),
        ),
    )


@pytest.mark.parametrize(
    "physical_s, elapsed_s",
    [
        (-1.0, 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (float("-inf"), 1.0),
        (3.0, -1.0),
        (3.0, float("nan")),
        (3.0, float("inf")),
        (3.0, float("-inf")),
    ],
)
def test_simulation_clock_rejects_invalid_durations_without_advancing(
    physical_s: float, elapsed_s: float
) -> None:
    wall_clock = Clock()
    sim_clock = SimulationClock(monotonic=wall_clock.monotonic)
    sim_clock.complete_round(3.0, 1.0)
    wall_clock.now = 10.0
    before = sim_clock.now()

    with pytest.raises(ValueError):
        sim_clock.complete_round(physical_s, elapsed_s)

    assert sim_clock.now() == before


@pytest.mark.parametrize(
    "scale, work", [(1.0, 0.02), (0.1, 0.002), (0.0, 0.02), (0.0, 0.3)]
)
def test_simulation_clock_preserves_physical_rounds_and_idle(clock: Clock, scale, work):
    import zcu_tools.program.v2.simulation as simulation

    clock_type = getattr(simulation, "SimulationClock", None)
    assert clock_type is not None, "simulation must expose an environment-owned clock"
    sim_clock = clock_type(monotonic=clock.monotonic)

    def execute(_request):
        clock.now += work
        return RoundSignals(())

    binding = SimulationBinding(
        lambda: EnvironmentSnapshot("synthetic", "test", ()),
        execute,
        backend_version="test",
        timing=SimulationTiming(time_scale=scale, clock=sim_clock),
    )
    sequence = binding.begin()
    try:
        request = request_for(binding, sequence)
        binding.sample_round(request, 0)
        assert sim_clock.now() == pytest.approx(max(0.2, work))
        binding.sample_round(request, 1)
        assert sim_clock.now() == pytest.approx(2 * max(0.2, work))
    finally:
        binding.end()
    clock.now += 10
    assert sim_clock.now() == pytest.approx(10 + 2 * max(0.2, work))


@pytest.mark.parametrize("separate_pacing", [False, True])
def test_simulation_clock_does_not_credit_cancelled_or_failed_rounds(
    clock: Clock, separate_pacing
):
    from zcu_tools.program.v2.simulation import SimulationClock

    sim_clock = SimulationClock(monotonic=clock.monotonic)
    cancel = Event()

    def execute(_request):
        clock.now += 0.03
        cancel.set()
        return RoundSignals(())

    binding = SimulationBinding(
        lambda: EnvironmentSnapshot("synthetic", "test", ()),
        execute,
        backend_version="test",
        timing=SimulationTiming(time_scale=0, clock=sim_clock),
    )
    binding.begin()
    try:
        binding.sample_round(
            request_for(binding, 0),
            0,
            cancel=cancel,
            pacing_cancel=Event() if separate_pacing else None,
        )
        assert sim_clock.now() == pytest.approx(0.03)
    finally:
        binding.end()

    def fail(_request):
        clock.now += 0.02
        raise RuntimeError("numeric failure")

    binding.execute = fail
    binding.begin()
    try:
        with pytest.raises(RuntimeError, match="numeric failure"):
            binding.sample_round(request_for(binding, 1), 0)
        assert sim_clock.now() == pytest.approx(0.05)
    finally:
        binding.end()


def test_physical_target_scales_with_reps_and_time_scale(clock: Clock):
    binding = SimulationBinding(
        lambda: EnvironmentSnapshot("synthetic", "test", ()),
        lambda _: RoundSignals(()),
        backend_version="test",
        seed=1,
        timing=SimulationTiming(time_scale=0.5),
    )
    sequence = binding.begin()
    try:
        request = request_for(binding, sequence)
        binding.sample_round(request, 0)
        binding.sample_round(replace(request, reps=100, loop_dims=(100, 3)), 1)
        assert clock.waits == pytest.approx([0.1, 1.0])
    finally:
        binding.end()


def test_each_round_includes_snapshot_time_and_only_waits_for_remainder(clock: Clock):
    durations = iter((0.05, 0.3, 0.0))
    result = RoundSignals(())

    def snapshot() -> EnvironmentSnapshot:
        clock.now += 0.02
        return EnvironmentSnapshot("synthetic", "test", ())

    def execute(_request) -> RoundSignals:
        clock.now += next(durations)
        return result

    binding = SimulationBinding(
        snapshot,
        execute,
        backend_version="test",
        seed=1,
        timing=SimulationTiming(time_scale=1),
    )
    sequence = binding.begin()
    try:
        request = request_for(binding, sequence)
        completed: list[float] = []
        for index in range(3):
            assert binding.sample_round(request, index) is result
            completed.append(clock.now)
        assert clock.waits == pytest.approx([0.13, 0.18])
        assert completed == pytest.approx([0.2, 0.52, 0.72])
        assert [r.index for r in binding.rounds] == [0, 1, 2]
    finally:
        binding.end()


@pytest.mark.parametrize("adc_length, expected", [(4.0, 180e-6), (12.0, 200e-6)])
def test_compiled_timeline_includes_readout_delays_and_per_point_relaxation(
    clock: Clock,
    adc_length: float,
    expected: float,
):
    binding = SimulationBinding(
        lambda: EnvironmentSnapshot("synthetic", "test", ()),
        lambda _: RoundSignals(()),
        backend_version="test",
        timing=SimulationTiming(time_scale=1),
    )
    program = ModularProgramV2(
        make_mock_soccfg(),
        modules=[
            PulseReadoutCfg(
                gen_ch=0,
                ro_ch=0,
                nqz=1,
                freq=100,
                gain=0.1,
                waveform=ConstWaveformCfg(length=2),
                pre_delay=3,
                post_delay=5,
                ro_length=adc_length,
            ).build("readout")
        ],
        reps=10,
        initial_delay=0,
        relax_delay=8,
    )
    sequence = binding.begin()
    try:
        binding.sample_round(build_request(program, "decimated", binding, sequence), 0)
        # Allow 10 ns of DAC/ADC rounding per repetition of this nominal sequence.
        assert clock.waits == pytest.approx([expected], abs=100e-9)
    finally:
        binding.end()


def test_stop_during_wait_returns_at_next_checkpoint(
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
):
    cancel = Event()

    def stop_on_sleep(duration: float) -> None:
        clock.sleep(duration)
        cancel.set()

    monkeypatch.setattr(time, "sleep", stop_on_sleep)
    result = RoundSignals(())
    binding = SimulationBinding(
        lambda: EnvironmentSnapshot("synthetic", "test", ()),
        lambda _: result,
        backend_version="test",
        timing=SimulationTiming(time_scale=1),
    )
    sequence = binding.begin()
    try:
        assert (
            binding.sample_round(request_for(binding, sequence), 0, cancel=cancel)
            is result
        )
        assert cancel.is_set()
        assert len(clock.waits) == 1
        assert 0 < clock.waits[0] <= 0.01
    finally:
        binding.end()


def test_decimated_stop_interrupts_wait_and_keeps_completed_trace(
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
):
    cancel = Event()

    def stop_on_sleep(duration: float) -> None:
        clock.sleep(duration)
        cancel.set()

    monkeypatch.setattr(time, "sleep", stop_on_sleep)
    env = white_noise(std_adc=0, seed=1)
    binding = SimulationBinding(
        env.soc.binding.capture_snapshot,
        env.soc.binding.execute,
        backend_version="synthetic",
        seed=1,
        timing=SimulationTiming(time_scale=1),
    )
    soc = SimulationSoc(env.soccfg, binding, routes=((0, 0),))
    program = ModularProgramV2(
        env.soccfg,
        modules=[
            PulseReadoutCfg(
                gen_ch=0,
                ro_ch=0,
                nqz=1,
                freq=100,
                gain=0.1,
                waveform=ConstWaveformCfg(length=0.2),
                ro_length=0.2,
            ).build("readout")
        ],
        reps=1,
        relax_delay=50_000,
    )
    try:
        data = program.acquire_decimated(
            soc,
            cancel_flag=cancel,
            progress=False,
            rounds=3,
        )
        assert data is not None
        assert np.isfinite(data).all()
        assert len(binding.rounds) == 1
        assert binding.rounds[0].cancel is None
        assert len(clock.waits) == 1
        assert 0 < clock.waits[0] <= 0.01
        # Decimated acquisition still keeps the completed trace of that round.
        assert program.rounds_buf is not None and len(program.rounds_buf) == 1
    finally:
        env.devices.close_all_devices()


def test_zero_scale_never_sleeps_or_reads_clock(monkeypatch: pytest.MonkeyPatch):
    def forbidden(*_args):
        pytest.fail("zero pacing must not read the clock or sleep")

    monkeypatch.setattr(time, "monotonic", forbidden)
    monkeypatch.setattr(time, "sleep", forbidden)
    binding = SimulationBinding(
        lambda: EnvironmentSnapshot("synthetic", "test", ()),
        lambda _: RoundSignals(()),
        backend_version="test",
        timing=SimulationTiming(time_scale=0),
    )
    sequence = binding.begin()
    try:
        binding.sample_round(request_for(binding, sequence), 0)
    finally:
        binding.end()


def test_backend_failure_propagates_without_waiting(clock: Clock):
    def fail(_request):
        raise RuntimeError("backend failed")

    binding = SimulationBinding(
        lambda: EnvironmentSnapshot("synthetic", "test", ()),
        fail,
        backend_version="test",
        timing=SimulationTiming(time_scale=1),
    )
    sequence = binding.begin()
    try:
        with pytest.raises(RuntimeError, match="backend failed"):
            binding.sample_round(request_for(binding, sequence), 0)
        assert clock.waits == []
    finally:
        binding.end()


@pytest.mark.parametrize("scale", [-0.1, float("nan"), float("inf")])
def test_binding_rejects_invalid_time_scale(scale: float):
    with pytest.raises(ValueError, match="time_scale"):
        SimulationBinding(
            lambda: EnvironmentSnapshot("synthetic", "test", ()),
            lambda _: RoundSignals(()),
            backend_version="test",
            timing=SimulationTiming(time_scale=scale),
        )
