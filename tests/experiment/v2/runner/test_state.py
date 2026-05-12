import numpy as np
import pytest
from zcu_tools.experiment.v2.runner.state import Result, TaskState

from .conftest import DictCfg


def test_child_extends_path_and_deepcopies_cfg():
    cfg = DictCfg.model_validate({"nested": {"b": 1}})
    root: dict[str, Result] = {"x": np.zeros(3)}
    s: TaskState[dict[str, Result], dict[str, Result], DictCfg] = TaskState(
        root_data=root, cfg=cfg
    )
    c = s.child("x", child_type=Result)
    assert c.path == ("x",)
    # Mutating child cfg does not affect parent (deepcopy check via model identity)
    assert c.cfg is not cfg


def test_set_value_ndarray_copies_in_place():
    arr = np.zeros(3)
    root: dict[str, Result] = {"x": arr}
    s: TaskState[dict[str, Result], dict[str, Result], DictCfg] = TaskState(
        root_data=root, cfg=DictCfg()
    )
    c = s.child("x", child_type=Result)
    c.set_value(np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(arr, [1.0, 2.0, 3.0])


def test_set_value_dict_updates_in_place():
    d: dict[str, Result] = {"a": np.zeros(1)}
    root: dict[str, Result] = {"d": d}
    s: TaskState[dict[str, Result], dict[str, Result], DictCfg] = TaskState(
        root_data=root, cfg=DictCfg()
    )
    c = s.child("d", child_type=Result)
    c.set_value({"b": np.zeros(2)})
    assert "a" in d and "b" in d


def test_set_value_list_replaces_contents():
    lst: list[Result] = [np.array(1), np.array(2), np.array(3)]
    root: dict[str, Result] = {"l": lst}
    s: TaskState[dict[str, Result], dict[str, Result], DictCfg] = TaskState(
        root_data=root, cfg=DictCfg()
    )
    c = s.child("l", child_type=Result)
    c.set_value([np.array(9), np.array(8)])
    assert len(lst) == 2


def test_set_value_type_mismatch_rejected():
    d: dict[str, Result] = {"d": {}}
    s: TaskState[dict[str, Result], dict[str, Result], DictCfg] = TaskState(
        root_data=d, cfg=DictCfg()
    )
    c = s.child("d", child_type=Result)
    with pytest.raises(ValueError):
        c.set_value([1, 2])  # type: ignore[arg-type]


def test_trigger_update_calls_hook():
    calls: list[tuple] = []
    root: dict[str, Result] = {"x": np.zeros(2)}
    s: TaskState[dict[str, Result], dict[str, Result], DictCfg] = TaskState(
        root_data=root, cfg=DictCfg(), on_update=lambda snap: calls.append(snap.path)
    )
    s.child("x", child_type=Result).set_value(np.array([1.0, 2.0]))
    assert calls == [("x",)]
