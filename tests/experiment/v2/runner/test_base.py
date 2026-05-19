from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.v2.runner.base import AbsTask, run_task
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
