from unittest.mock import MagicMock

import numpy as np
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.runner.base import AbsTask
from zcu_tools.experiment.v2.runner.soft import Scan
from zcu_tools.experiment.v2.runner.state import TaskState


def _mock_subtask():
    m = MagicMock(spec=AbsTask)
    m.get_default_result.return_value = np.zeros(1)
    return m


def test_scan_before_each_called_per_value_in_order():
    calls = []
    sub = _mock_subtask()
    values = [10, 20, 30]
    scan = Scan(
        "sw", values, before_each=lambda i, s, v: calls.append((i, v)), task=sub
    )

    root = scan.get_default_result()
    state = TaskState(root_data=root, cfg=ExpCfgModel())

    scan.init(dynamic_pbar=False)
    scan.run(state)
    scan.cleanup()

    assert calls == [(0, 10), (1, 20), (2, 30)]
    assert sub.run.call_count == 3


def test_scan_empty_values_skips_sub_run():
    sub = _mock_subtask()
    scan = Scan("sw", [], before_each=lambda i, s, v: None, task=sub)
    state = TaskState(root_data=scan.get_default_result(), cfg=ExpCfgModel())
    scan.init(dynamic_pbar=False)
    scan.run(state)
    scan.cleanup()
    sub.run.assert_not_called()


def test_scan_default_result_length_matches_values():
    sub = _mock_subtask()
    scan = Scan("sw", [1, 2, 3, 4], before_each=lambda i, s, v: None, task=sub)
    assert len(scan.get_default_result()) == 4
