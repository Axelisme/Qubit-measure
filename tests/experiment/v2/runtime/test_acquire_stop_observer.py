"""Author acquisition observers report external stops, not data completeness."""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.v2.runtime import (
    AcquireOptions,
    RunOptions,
    Schedule,
    SignalBuffer,
)
from zcu_tools.program.acquisition import (
    CancelFlagProtocol,
    ShotSumProcessor,
    StoppedPartialAcquireError,
)
from zcu_tools.progress_bar import use_pbar_factory

from tests.experiment.v2.runtime._flow_fakes import (
    FakeModule,
    RecordingProgressBar,
    recording_pbar_factory,
)

Entry = Literal[
    "builder",
    "builder_decimated",
    "builder_summed",
    "direct",
    "direct_decimated",
    "direct_summed",
]
ENTRIES: tuple[Entry, ...] = (
    "builder",
    "builder_decimated",
    "builder_summed",
    "direct",
    "direct_decimated",
    "direct_summed",
)


class Producer:
    """Deterministic transport boundary with one externally selected ending."""

    def __init__(self, root: Scope, ending: str) -> None:
        self.root = root
        self.ending = ending
        self.calls = 0
        self.error = RuntimeError("producer failed")

    def acquire(
        self,
        soc: object,
        *,
        rounds: int,
        round_hook: Callable[
            [int, list[NDArray[np.float64]], CancelFlagProtocol], object
        ],
        cancel_flag: CancelFlagProtocol,
        **_kwargs: object,
    ) -> list[NDArray[np.float64]]:
        self.calls += 1
        if self.ending == "before_data":
            assert self.root.cancel("acquire").cancelled
            raise StoppedPartialAcquireError("no completed round")
        if self.ending == "interrupt":
            raise KeyboardInterrupt("producer interrupted")
        raw = [np.array([7.0])]
        round_hook(1, raw, cancel_flag)
        if self.ending in {"finite", "failure"}:
            assert self.root.cancel("acquire").cancelled
        elif self.ending == "final_hook":
            round_hook(rounds, raw, cancel_flag)
            assert self.root.cancel("acquire").cancelled
        if self.ending == "failure":
            raise self.error
        return raw

    acquire_decimated = acquire

    def acquire_summed(
        self,
        soc: object,
        *,
        processor: ShotSumProcessor,
        rounds: int = 1,
        step_rounds: bool = False,
        **qick_options: Any,
    ) -> NDArray[np.float64]:
        # The native summed boundary accepts dynamically typed QICK keywords.
        return self.acquire(
            soc,
            rounds=rounds,
            cancel_flag=qick_options["cancel_flag"],
            round_hook=lambda ir, raw, flag: qick_options["round_hook"](
                ir, raw[0], flag
            ),
        )[0]


def run_entry(
    entry: Entry,
    sched: Schedule[Any],
    producer: Producer,
    options: AcquireOptions,
    convert: Callable[[NDArray[np.float64]], NDArray[np.float64]] = lambda raw: raw,
) -> NDArray[np.float64]:
    """Exercise either public options family through its acquisition entry."""
    if entry.startswith("builder"):
        builder = sched.prog_builder(object(), object())
        if entry == "builder_summed":
            return builder.run_program_summed(
                producer,
                processor=lambda iq: iq[0],
                sums2signal_fn=convert,
                rounds=2,
                options=options,
            )
        method = (
            builder.run_program_decimated
            if entry == "builder_decimated"
            else builder.run_program
        )
        return method(
            producer,
            rounds=2,
            raw2signal_fn=lambda raw: convert(raw[0]),
            options=options,
        )
    if entry == "direct_summed":
        direct_sums = RunOptions(
            retry=options.retry,
            converter=convert,
            stop_condition=options.stop_condition,
            on_stop=options.on_stop,
        )
        return sched.run_acquire_summed(
            producer,
            object(),
            processor=lambda iq: iq[0],
            rounds=2,
            options=direct_sums,
        )
    direct: RunOptions[list[NDArray[np.float64]]] = RunOptions(
        retry=options.retry,
        converter=lambda raw: convert(raw[0]),
        stop_condition=options.stop_condition,
        on_stop=options.on_stop,
    )
    method = (
        sched.run_acquire_decimated
        if entry == "direct_decimated"
        else sched.run_acquire
    )
    return method(producer, object(), rounds=2, options=direct)


@pytest.mark.parametrize("decimated", [False, True])
@pytest.mark.parametrize("callback_fails", [False, True])
def test_pre_stopped_build_shortcut_notifies_once_without_build_or_retry(
    decimated: bool, callback_fails: bool
) -> None:
    builds: list[object] = []
    observed: list[bool] = []
    cause = ValueError("stop observer failed")

    def factory(soccfg: object, **kwargs: Any) -> Producer:
        builds.append(soccfg)
        return Producer(root, "complete")

    def on_stop() -> None:
        observed.append(True)
        if callback_fails:
            raise cause

    options = AcquireOptions(retry=2, on_stop=on_stop)
    root = Scope("run")
    buffer = SignalBuffer((1,), dtype=np.float64)
    buffer.set(np.array([19.0]))
    with Schedule( buffer, scope=root) as sched:
        builder = sched.prog_builder(object(), object(), program_cls=factory).add(
            FakeModule("readout")
        )
        run = (
            builder.build_and_acquire_decimated
            if decimated
            else builder.build_and_acquire
        )
        sched.set_stop()
        if callback_fails:
            with pytest.raises(ValueError, match="stop observer failed") as caught:
                run(options=options)
            assert caught.value is cause
        else:
            np.testing.assert_array_equal(run(options=options), [19.0])
        assert observed == [True]
        assert builds == []
        np.testing.assert_array_equal(buffer.data, [19.0])
        assert sched.outcome.status == "stopped"
    # Reusing the options on a fresh run must not replay a previous stop.
    with Schedule( SignalBuffer((1,), dtype=np.float64), scope=root) as sched:
        result = run_entry("builder", sched, Producer(root, "complete"), options)
        np.testing.assert_array_equal(result, [7.0])
        assert observed == [True]
        assert sched.outcome.status == "completed"


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize(
    "ending",
    ["before_data", "finite", "final_hook", "pre_stopped", "complete", "data_stop"],
)
def test_external_stop_observer_runs_after_cleanup_once_and_isolates_next_acquire(
    entry: Entry,
    ending: str,
) -> None:
    root = Scope("run")
    bars: list[RecordingProgressBar] = []
    observed: list[int] = []
    producer = Producer(root, ending)
    with (
        use_pbar_factory(recording_pbar_factory(bars)),
        Schedule( SignalBuffer((1,), dtype=np.float64), scope=root) as sched,
    ):

        def on_stop() -> None:
            assert root.active_path == sched.scope.path
            assert bars and all(bar.closed for bar in bars)
            observed.append(producer.calls)

        options = AcquireOptions(
            retry=2,
            on_stop=on_stop,
            stop_condition=lambda: ending == "data_stop",
        )
        if ending == "pre_stopped":
            sched.set_stop()
        result = run_entry(entry, sched, producer, options)
        cancelled = ending not in {"complete", "data_stop"}
        assert observed == ([producer.calls] if cancelled else [])
        assert sched.outcome.status == (
            "stopped" if ending == "pre_stopped" else "completed"
        )
        if ending in {"before_data", "pre_stopped"}:
            assert np.isnan(result).all()
        else:
            np.testing.assert_array_equal(result, [7.0])
        if ending != "pre_stopped":
            producer.ending = "complete"
            ending = "complete"
            again = run_entry(entry, sched, producer, options)
            np.testing.assert_array_equal(again, [7.0])
            assert observed == ([1] if cancelled else [])
            assert producer.calls == 2
            assert not sched.is_stop()
    assert root.active_path == "run"
    assert not root.stop.is_set()


@pytest.mark.parametrize("mode", ["integrated", "decimated", "summed"])
def test_reusing_run_options_does_not_replay_previous_stop(mode: str) -> None:
    root = Scope("run")
    producer = Producer(root, "finite")
    observed: list[bool] = []
    with Schedule( SignalBuffer((1,), dtype=np.float64), scope=root) as sched:
        if mode == "summed":
            sums_options: RunOptions[NDArray[np.float64]] = RunOptions(
                on_stop=lambda: observed.append(True)
            )

            def run() -> NDArray[np.float64]:
                return sched.run_acquire_summed(
                    producer,
                    object(),
                    processor=lambda iq: iq[0],
                    rounds=2,
                    options=sums_options,
                )
        else:
            iq_options: RunOptions[list[NDArray[np.float64]]] = RunOptions(
                converter=lambda raw: raw[0], on_stop=lambda: observed.append(True)
            )
            method = (
                sched.run_acquire_decimated
                if mode == "decimated"
                else sched.run_acquire
            )

            def run() -> NDArray[np.float64]:
                return method(producer, object(), rounds=2, options=iq_options)

        np.testing.assert_array_equal(run(), [7.0])
        assert observed == [True]
        producer.ending = "complete"
        np.testing.assert_array_equal(run(), [7.0])
        assert observed == [True]
        assert producer.calls == 2
        assert sched.outcome.status == "completed"
        assert not sched.is_stop()


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("ending", ["failure", "interrupt", "converter", "update"])
def test_exception_does_not_report_cooperative_stop(entry: Entry, ending: str) -> None:
    root = Scope("run")
    producer = Producer(root, ending)
    observed: list[bool] = []
    cause = producer.error
    bars: list[RecordingProgressBar] = []

    def convert(raw: NDArray[np.float64]) -> NDArray[np.float64]:
        if ending == "converter":
            raise cause
        return raw

    update_failed = False

    def update(_data: NDArray[np.float64]) -> None:
        nonlocal update_failed
        if ending == "update" and not update_failed:
            update_failed = True
            raise cause

    with (
        use_pbar_factory(recording_pbar_factory(bars)),
        Schedule(
            
            SignalBuffer(
                (1,), dtype=np.float64, on_update=update, update_interval=None
            ),
            scope=root,
        ) as sched,
    ):
        with pytest.raises(
            KeyboardInterrupt if ending == "interrupt" else RuntimeError
        ) as caught:
            run_entry(
                entry,
                sched,
                producer,
                AcquireOptions(retry=2, on_stop=lambda: observed.append(True)),
                convert,
            )
        if ending != "interrupt":
            assert caught.value is cause
        assert producer.calls == (3 if ending == "failure" else 1)
        assert observed == []
        assert root.active_path == sched.scope.path
        assert all(bar.closed for bar in bars)


@pytest.mark.parametrize("entry", ENTRIES)
def test_stop_callback_failure_propagates_without_retry_or_outcome_change(
    entry: Entry,
) -> None:
    root = Scope("run")
    producer = Producer(root, "finite")
    cause = ValueError("observer failed")
    bars: list[RecordingProgressBar] = []

    def on_stop() -> None:
        assert bars and all(bar.closed for bar in bars)
        raise cause

    with (
        use_pbar_factory(recording_pbar_factory(bars)),
        Schedule( SignalBuffer((1,), dtype=np.float64), scope=root) as sched,
    ):
        with pytest.raises(ValueError, match="observer failed") as caught:
            run_entry(entry, sched, producer, AcquireOptions(retry=2, on_stop=on_stop))
        assert caught.value is cause
        assert producer.calls == 1
        assert sched.outcome.status == "completed"
        assert not sched.is_stop()
        assert root.active_path == sched.scope.path
    assert root.active_path == "run"
