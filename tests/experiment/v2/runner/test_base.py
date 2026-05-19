import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.v2.runner.base import (
    AbsTask,
    ActiveTask,
    TaskHandle,
    run_task,
)
from zcu_tools.experiment.v2.runner.state import Result, TaskState

from .conftest import DictCfg


def _mock_task():
    m = MagicMock(spec=AbsTask)
    m.get_default_result.return_value = np.zeros(2)
    return m


def test_run_task_success_calls_init_run_cleanup_in_order():
    t = _mock_task()
    calls = []
    t.init.side_effect = lambda *a, **kw: calls.append("init")
    t.run.side_effect = lambda *a, **kw: calls.append("run")
    t.cleanup.side_effect = lambda: calls.append("cleanup")

    run_task(t, init_cfg=DictCfg.model_validate({"k": 1}))
    assert calls == ["init", "run", "cleanup"]


def test_run_task_keyboard_interrupt_swallowed_and_cleanup_called():
    t = _mock_task()
    t.run.side_effect = KeyboardInterrupt
    run_task(t, init_cfg=DictCfg())
    t.cleanup.assert_called_once()


def test_run_task_exception_reraised_after_cleanup():
    t = _mock_task()
    t.run.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        run_task(t, init_cfg=DictCfg())
    t.cleanup.assert_called_once()


def test_run_task_on_update_wired_through_state():
    t = _mock_task()
    seen: list[str] = []

    def _run(state: TaskState[Result, Result, DictCfg]) -> None:
        if state.on_update is not None:
            state.on_update(state)  # min_interval wrapper should forward

    t.run.side_effect = _run

    run_task(
        t,
        init_cfg=DictCfg(),
        on_update=lambda _snap: seen.append("hit"),
        update_interval=None,
    )
    assert seen == ["hit"]


def test_run_task_deepcopies_init_cfg():
    t = _mock_task()
    init_cfg = DictCfg()
    seen_cfg: list[DictCfg] = []

    def _run(state: TaskState[Result, Result, DictCfg]) -> None:
        seen_cfg.append(state.cfg)

    t.run.side_effect = _run
    run_task(t, init_cfg=init_cfg)
    # The cfg passed into the state must be a deepcopy, not the original object.
    assert seen_cfg and seen_cfg[0] is not init_cfg


def test_task_handle_cancel_and_is_cancelled():
    flag = threading.Event()
    handle = TaskHandle(flag)
    assert not handle.is_cancelled()
    handle.cancel()
    assert handle.is_cancelled()
    assert flag.is_set()


def test_active_task_provides_handle_and_clears_on_exit():
    import zcu_tools.experiment.v2.runner.base as base_mod

    event = threading.Event()
    with ActiveTask(event) as handle:
        assert base_mod._current_stop_flag is event
        assert not handle.is_cancelled()

    assert base_mod._current_stop_flag is None


def test_active_task_nested_raises():
    with ActiveTask(threading.Event()):
        with pytest.raises(RuntimeError, match="nested"):
            with ActiveTask(threading.Event()):
                pass


def test_run_task_uses_active_task_stop_flag():
    t = _mock_task()
    seen_stop: list[bool] = []

    def _run(state: TaskState[Result, Result, DictCfg]) -> None:
        seen_stop.append(state.is_stop())

    t.run.side_effect = _run

    event = threading.Event()
    with ActiveTask(event) as handle:
        handle.cancel()
        run_task(t, init_cfg=DictCfg())

    assert seen_stop == [True]


def test_run_task_explicit_stop_flag_overrides_active_task():
    t = _mock_task()
    seen_stop: list[bool] = []

    def _run(state: TaskState[Result, Result, DictCfg]) -> None:
        seen_stop.append(state.is_stop())

    t.run.side_effect = _run

    explicit_flag = threading.Event()
    with ActiveTask(threading.Event()):
        # explicit_flag is not set; ActiveTask's flag is irrelevant
        run_task(t, init_cfg=DictCfg(), stop_flag=explicit_flag)

    assert seen_stop == [False]


# ------------------------------------------------------------------
# AbsTask factory methods — need a real (non-mocked) AbsTask subclass
# ------------------------------------------------------------------


class _ConcreteTask(AbsTask):  # type: ignore[type-arg]
    def run(self, state) -> None:  # type: ignore[override]
        pass

    def get_default_result(self):
        import numpy as np

        return np.zeros(1)


def test_abstask_scan_returns_scan_instance():
    from zcu_tools.experiment.v2.runner.soft import Scan

    t = _ConcreteTask()
    result = t.scan("freq", [1, 2, 3], before_each=lambda i, s, v: None)
    assert isinstance(result, Scan)


def test_abstask_repeat_returns_repeat_instance():
    from zcu_tools.experiment.v2.runner.repeat import RepeatOverTime

    t = _ConcreteTask()
    result = t.repeat("T1", times=5, interval=0.0)
    assert isinstance(result, RepeatOverTime)


def test_abstask_auto_retry_returns_retry_instance():
    from zcu_tools.experiment.v2.runner.repeat import ReTryIfFail

    t = _ConcreteTask()
    result = t.auto_retry(max_retries=3)
    assert isinstance(result, ReTryIfFail)
