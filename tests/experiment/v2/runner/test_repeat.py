from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.v2.runner.base import AbsTask
from zcu_tools.experiment.v2.runner.repeat import (
    RepeatOverTime,
    ReTryIfFail,
    run_with_retries,
)
from zcu_tools.experiment.v2.runner.state import TaskState


def _mock_subtask():
    m = MagicMock(spec=AbsTask)
    m.get_default_result.return_value = np.zeros(1)
    return m


def test_repeat_over_time_updates_repeat_idx(monkeypatch):
    sub = _mock_subtask()
    seen_idx = []

    def _record_run(state):
        seen_idx.append(state.env["repeat_idx"])

    sub.run.side_effect = _record_run

    r = RepeatOverTime("r", num_times=3, task=sub, interval=0.0)
    state = TaskState(root_data=r.get_default_result(), cfg={})
    r.init()
    r.run(state)
    r.cleanup()

    assert seen_idx == [0, 1, 2]


def test_run_with_retries_succeeds_after_failures():
    sub = _mock_subtask()
    attempts = {"n": 0}

    def flaky(state):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("boom")

    sub.run.side_effect = flaky

    state = TaskState(root_data=sub.get_default_result(), cfg={})
    run_with_retries(sub, state, retry_time=5)

    assert attempts["n"] == 3
    # cleanup+init called between retries (twice)
    assert sub.init.call_count == 2
    assert sub.cleanup.call_count == 2


def test_run_with_retries_raises_when_exhausted():
    sub = _mock_subtask()
    sub.run.side_effect = RuntimeError("always")

    state = TaskState(root_data=sub.get_default_result(), cfg={})
    with pytest.raises(RuntimeError):
        run_with_retries(sub, state, retry_time=2)

    assert sub.run.call_count == 3


def test_retryiffail_delegates_default_result():
    sub = _mock_subtask()
    r = ReTryIfFail(task=sub, max_retries=1)
    assert r.get_default_result() is sub.get_default_result.return_value
