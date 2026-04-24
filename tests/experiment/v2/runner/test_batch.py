from unittest.mock import MagicMock

import numpy as np
from zcu_tools.experiment.v2.runner.base import AbsTask
from zcu_tools.experiment.v2.runner.batch import BatchTask
from zcu_tools.experiment.v2.runner.state import TaskState


def _mock_subtask(default):
    m = MagicMock(spec=AbsTask)
    m.get_default_result.return_value = default
    return m


def test_batch_default_result_aggregates_per_child():
    a = _mock_subtask(np.zeros(2))
    b = _mock_subtask(np.zeros(3))
    bt = BatchTask({"a": a, "b": b})
    out = bt.get_default_result()
    assert set(out.keys()) == {"a", "b"}
    assert out["a"].shape == (2,)
    assert out["b"].shape == (3,)


def test_batch_run_dispatches_to_children():
    a = _mock_subtask(np.zeros(2))
    b = _mock_subtask(np.zeros(2))
    bt = BatchTask({"a": a, "b": b})

    root = bt.get_default_result()
    state = TaskState(root_data=root, cfg={})

    bt.init(state)
    bt.run(state)
    bt.cleanup()

    a.init.assert_called_once()
    b.init.assert_called_once()
    a.run.assert_called_once()
    b.run.assert_called_once()
    a.cleanup.assert_called_once()
    b.cleanup.assert_called_once()
