"""Public prepared-session timing contracts with synthetic callbacks."""

from contextlib import contextmanager

import pytest

from tools import simulator_measurement as measurement


@pytest.mark.parametrize(
    ("wall_elapsed", "process_elapsed"), [(4, 2), (0, 2), (4, 0), (0, 0), (1, 4)]
)
def test_one_evolve_records_both_clocks_and_excludes_inspection(
    wall_elapsed, process_elapsed
):
    wall, process = [10.0], [20.0]
    calls = []

    def invoke(telemetry):
        calls.append(telemetry)
        wall[0] += wall_elapsed
        process[0] += process_elapsed
        return "density"

    def inspect(result, telemetry):
        assert (result, telemetry) == ("density", True)
        wall[0] += 1000
        process[0] += 500
        return "checked"

    measured = measurement.measure_sample(
        measurement.Session(invoke, inspect, lambda: None),
        telemetry=True,
        wall_clock=lambda: wall[0],
        process_clock=lambda: process[0],
    )
    assert calls == [True]
    assert (measured.wall_seconds, measured.process_seconds) == (
        wall_elapsed,
        process_elapsed,
    )
    assert measured.observation == "checked"


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
@pytest.mark.parametrize("clock", ["wall", "process"])
def test_invalid_elapsed_is_rejected_at_sample_construction(duration, clock):
    bad_ticks = iter([0, duration])
    good_ticks = iter([0, 1])
    clocks = {
        clock: lambda: next(bad_ticks),
        ("process" if clock == "wall" else "wall"): lambda: next(good_ticks),
    }
    with pytest.raises(ValueError, match=f"{clock}_seconds"):
        measurement.measure_sample(
            measurement.Session(lambda _: None, lambda *_: "checked", lambda: None),
            telemetry=False,
            wall_clock=clocks["wall"],
            process_clock=clocks["process"],
        )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("first", [[0], [0]]),
        ("warm_clear_history", [[0, 1, 0, 0]]),
        ("warm_retain_history", [[0, 1, 2, 3]]),
    ],
)
def test_session_lifetime_and_history_are_outside_sample_clocks(mode, expected):
    clock = [0.0]
    histories, closed = [], []

    @contextmanager
    def factory():
        clock[0] += 1000
        seen, history = [], [0]
        histories.append(seen)

        def invoke(telemetry):
            seen.append(history[0])
            history[0] += 1
            clock[0] += 1

        def clear():
            history[0] = 0
            clock[0] += 500

        try:
            yield measurement.Session(invoke, lambda *_: "checked", clear)
        finally:
            closed.append(True)
            clock[0] += 1000

    with measurement.SampleRunner(
        factory,
        mode=mode,
        warmups=2,
        wall_clock=lambda: clock[0],
        process_clock=lambda: clock[0],
    ) as runner:
        samples = [runner.sample(telemetry=False) for _ in range(2)]
    assert histories == expected
    assert len(closed) == len(expected)
    assert [(s.wall_seconds, s.process_seconds) for s in samples] == [(1, 1)] * 2


@pytest.mark.parametrize(
    ("mode", "warmups", "message"),
    [
        ("unknown", 2, "mode"),
        ("first", -1, "warmups"),
        ("warm_retain_history", -1, "warmups"),
    ],
)
def test_invalid_configuration_fails_before_opening(mode, warmups, message):
    opened = []

    @contextmanager
    def factory():
        opened.append(True)
        yield measurement.Session(lambda _: None, lambda *_: "checked", lambda: None)

    with pytest.raises(ValueError, match=message):
        measurement.SampleRunner(factory, mode=mode, warmups=warmups)
    assert opened == []


def test_empty_history_diagnostic_clears_once_without_rewarming():
    history = [0]
    seen = []
    ticks = iter(range(100))

    def invoke(telemetry):
        seen.append((history[0], telemetry))
        history[0] += 1

    def clear():
        history[0] = 0

    @contextmanager
    def factory():
        yield measurement.Session(invoke, lambda *_: "checked", clear)

    with measurement.SampleRunner(
        factory,
        mode="warm_retain_history",
        wall_clock=lambda: next(ticks),
        process_clock=lambda: next(ticks),
    ) as runner:
        runner.sample(telemetry=False)
        runner.sample(telemetry=True, empty_history=True)
        runner.sample(telemetry=False)
    assert seen == [(0, False), (1, False), (2, False), (0, True), (1, False)]


@pytest.mark.parametrize("mode", ["first", "warm_clear_history", "warm_retain_history"])
@pytest.mark.parametrize("failure", ["invoke", "inspect", "observer", "cleanup"])
def test_callback_failure_propagates_and_context_releases_session(mode, failure):
    closed = []
    ticks = iter(range(100))

    def invoke(_telemetry):
        if failure == "invoke":
            raise RuntimeError("invoke failed")

    def inspect(_result, _telemetry):
        if failure == "inspect":
            raise RuntimeError("inspect failed")
        return "checked"

    def observe(_mode, _sample):
        if failure == "observer":
            raise RuntimeError("observer failed")

    @contextmanager
    def factory():
        try:
            yield measurement.Session(invoke, inspect, lambda: None, observe)
        finally:
            closed.append(True)
            if failure == "cleanup":
                raise RuntimeError("cleanup failed")

    with (
        pytest.raises(RuntimeError, match=f"{failure} failed"),
        measurement.SampleRunner(
            factory,
            mode=mode,
            warmups=0,
            wall_clock=lambda: next(ticks),
            process_clock=lambda: next(ticks),
        ) as runner,
    ):
        runner.sample(telemetry=True)
    assert closed == [True]


def test_runner_close_releases_and_next_sample_reopens_and_rewarms():
    seen, closed = [], []
    ticks = iter(range(100))

    @contextmanager
    def factory():
        seen.append([])
        calls = seen[-1]
        try:
            yield measurement.Session(
                lambda flag: calls.append(flag), lambda *_: "checked", lambda: None
            )
        finally:
            closed.append(True)

    with measurement.SampleRunner(
        factory,
        mode="warm_retain_history",
        wall_clock=lambda: next(ticks),
        process_clock=lambda: next(ticks),
    ) as runner:
        runner.sample(telemetry=False)
        runner.close()
        runner.close()
        runner.sample(telemetry=True)
    assert seen == [[False, False, False], [True, True, True]]
    assert closed == [True, True]
