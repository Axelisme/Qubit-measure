"""Round pacing at the public binding seam, using a virtual time boundary."""

import time
from collections.abc import Iterator

import pytest
from zcu_tools.core.simulation import AcquireRequest, EnvironmentSnapshot, RoundSignals
from zcu_tools.program.v2.simulation import SimulationBinding


class Clock:
    """Monotonic virtual seconds; waiting advances without blocking."""

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


def test_each_round_includes_snapshot_time_and_only_waits_for_remainder(clock: Clock):
    durations: Iterator[float] = iter((0.05, 0.3, 0.0))
    result = RoundSignals(())

    def snapshot() -> EnvironmentSnapshot:
        clock.now += 0.02
        return EnvironmentSnapshot("synthetic", "test", ())

    def execute(_request) -> RoundSignals:
        clock.now += next(durations)
        return result

    binding = SimulationBinding(
        snapshot, execute, backend_version="test", seed=1, min_round_duration_s=0.2
    )
    sequence = binding.begin()
    request = AcquireRequest("decimated", (1,), 0, (), 1, sequence, binding.rng_version)
    try:
        completed: list[float] = []
        for index in range(3):
            assert binding.sample_round(request, index) is result
            completed.append(clock.now)
        assert clock.waits == pytest.approx([0.13, 0.18])
        assert completed == pytest.approx([0.2, 0.52, 0.72])
        assert [r.index for r in binding.rounds] == [0, 1, 2]
    finally:
        binding.end()


@pytest.mark.parametrize("duration", [-0.1, float("nan"), float("inf")])
def test_binding_rejects_invalid_round_duration(duration: float):
    with pytest.raises(ValueError, match="min_round_duration_s"):
        SimulationBinding(
            lambda: EnvironmentSnapshot("synthetic", "test", ()),
            lambda _: RoundSignals(()),
            backend_version="test",
            min_round_duration_s=duration,
        )
