import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.v2.runner.base import AbsTask
from zcu_tools.experiment.v2.runner.repeat import (
    RepeatOverTime,
    ReTryIfFail,
    run_with_retries,
)
from zcu_tools.experiment.v2.runner.state import Result, TaskState

from .conftest import DictCfg


def _mock_subtask():
    m = MagicMock(spec=AbsTask)
    m.get_default_result.return_value = np.zeros(1)
    return m


def test_repeat_over_time_updates_repeat_idx(monkeypatch):
    sub = _mock_subtask()
    seen_idx: list[int] = []

    def _record_run(state: TaskState[Result, Result, DictCfg]) -> None:
        seen_idx.append(state.env["repeat_idx"])

    sub.run.side_effect = _record_run

    r = RepeatOverTime("r", num_times=3, task=sub, interval=0.0)
    default = r.get_default_result()
    state: TaskState[list[Result], list[Result], DictCfg] = TaskState(
        root_data=default, cfg=DictCfg()
    )
    r.init()
    r.run(state)
    r.cleanup()

    assert seen_idx == [0, 1, 2]


def test_run_with_retries_succeeds_after_failures():
    sub = _mock_subtask()
    attempts = {"n": 0}

    def flaky(state: TaskState[Result, Result, DictCfg]) -> None:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("boom")

    sub.run.side_effect = flaky

    state: TaskState[Result, Result, DictCfg] = TaskState(
        root_data=sub.get_default_result(), cfg=DictCfg()
    )
    run_with_retries(sub, state, retry_time=5)

    assert attempts["n"] == 3
    # cleanup+init called between retries (twice)
    assert sub.init.call_count == 2
    assert sub.cleanup.call_count == 2


def test_run_with_retries_raises_when_exhausted():
    sub = _mock_subtask()
    sub.run.side_effect = RuntimeError("always")

    state: TaskState[Result, Result, DictCfg] = TaskState(
        root_data=sub.get_default_result(), cfg=DictCfg()
    )
    with pytest.raises(RuntimeError):
        run_with_retries(sub, state, retry_time=2)

    assert sub.run.call_count == 3


def test_retryiffail_delegates_default_result():
    sub = _mock_subtask()
    r = ReTryIfFail(task=sub, max_retries=1)
    assert r.get_default_result() is sub.get_default_result.return_value


def test_repeat_stops_early_when_flag_set_before_run():
    flag = threading.Event()
    sub = _mock_subtask()

    r = RepeatOverTime("r", num_times=5, task=sub, interval=0.0)
    default = r.get_default_result()
    state: TaskState[list[Result], list[Result], DictCfg] = TaskState(
        root_data=default, cfg=DictCfg(), _stop_flag=flag
    )

    flag.set()
    r.init()
    r.run(state)
    r.cleanup()

    sub.run.assert_not_called()


def test_repeat_stops_early_after_first_iteration():
    flag = threading.Event()
    sub = _mock_subtask()
    run_count = {"n": 0}

    def _run_and_cancel(state: TaskState) -> None:  # type: ignore[type-arg]
        run_count["n"] += 1
        flag.set()

    sub.run.side_effect = _run_and_cancel

    r = RepeatOverTime("r", num_times=5, task=sub, interval=0.0)
    default = r.get_default_result()
    state: TaskState[list[Result], list[Result], DictCfg] = TaskState(
        root_data=default, cfg=DictCfg(), _stop_flag=flag
    )

    r.init()
    r.run(state)
    r.cleanup()

    assert run_count["n"] == 1


def test_repeat_dynamic_pbar_closed_after_run():
    sub = _mock_subtask()
    r = RepeatOverTime("r", num_times=2, task=sub, interval=0.0)
    default = r.get_default_result()
    state: TaskState[list[Result], list[Result], DictCfg] = TaskState(
        root_data=default, cfg=DictCfg()
    )

    r.init(dynamic_pbar=True)
    r.run(state)
    assert r.iter_pbar is None
    assert r.time_pbar is None


def test_retry_if_fail_init_run_cleanup():
    sub = _mock_subtask()
    r = ReTryIfFail(task=sub, max_retries=2)

    default = r.get_default_result()
    state: TaskState[Result, Result, DictCfg] = TaskState(
        root_data=default, cfg=DictCfg()
    )

    r.init(dynamic_pbar=True)
    r.run(state)
    r.cleanup()

    sub.init.assert_called_once_with(dynamic_pbar=True)
    sub.run.assert_called_once()
    sub.cleanup.assert_called_once()


def test_run_with_retries_no_raise_on_exhaustion():
    sub = _mock_subtask()
    sub.run.side_effect = RuntimeError("always")

    state: TaskState[Result, Result, DictCfg] = TaskState(
        root_data=sub.get_default_result(), cfg=DictCfg()
    )
    # Should not raise
    run_with_retries(sub, state, retry_time=1, raise_error=False)
    assert sub.run.call_count == 2


def test_repeat_force_triggers_hook_when_interval_set(monkeypatch):
    """Verify update hook is force-triggered after sub-task when interval > 0."""
    sub = _mock_subtask()
    hook_calls: list[int] = []

    r = RepeatOverTime("r", num_times=1, task=sub, interval=10.0)
    default = r.get_default_result()

    def _on_update(snap: TaskState) -> None:  # type: ignore[type-arg]
        hook_calls.append(1)

    state: TaskState[list[Result], list[Result], DictCfg] = TaskState(
        root_data=default, cfg=DictCfg(), on_update=_on_update
    )

    # Make time.time() return values so interval wait is skipped but post-task
    # check (time.time() - start_t < interval) is True, triggering the hook.
    times = iter(
        [
            0.0,  # start_t = time.time() - 2*interval  → 0 - 20 = -20
            0.0,  # while condition: 0 - (-20) = 20 >= 10, skip wait
            0.0,  # start_t = time.time()
            0.0,  # post-task: 0 - 0 = 0 < 10  → trigger hook
        ]
    )
    monkeypatch.setattr(
        "zcu_tools.experiment.v2.runner.repeat.time.time", lambda: next(times)
    )

    r.init(dynamic_pbar=False)
    r.run(state)
    r.cleanup()

    assert len(hook_calls) >= 1
