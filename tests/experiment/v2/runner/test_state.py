import numpy as np
import pytest

from zcu_tools.experiment.v2.runner.state import TaskState


def test_child_extends_path_and_deepcopies_cfg():
    cfg = {"a": {"b": 1}}
    root = {"x": np.zeros(3)}
    s = TaskState(root_data=root, cfg=cfg)
    c = s.child("x")
    assert c.path == ("x",)
    c.cfg["a"]["b"] = 99
    assert cfg["a"]["b"] == 1


def test_set_value_ndarray_copies_in_place():
    arr = np.zeros(3)
    root = {"x": arr}
    s = TaskState(root_data=root, cfg={}).child("x")
    s.set_value(np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(arr, [1.0, 2.0, 3.0])


def test_set_value_dict_updates_in_place():
    d = {"a": 1}
    s = TaskState(root_data={"d": d}, cfg={}).child("d")
    s.set_value({"b": 2})
    assert d == {"a": 1, "b": 2}


def test_set_value_list_replaces_contents():
    lst = [1, 2, 3]
    s = TaskState(root_data={"l": lst}, cfg={}).child("l")
    s.set_value([9, 8])
    assert lst == [9, 8]


def test_set_value_type_mismatch_rejected():
    s = TaskState(root_data={"d": {}}, cfg={}).child("d")
    with pytest.raises(ValueError):
        s.set_value([1, 2])


def test_trigger_update_calls_hook():
    calls = []
    root = {"x": np.zeros(2)}
    s = TaskState(root_data=root, cfg={}, on_update=lambda snap: calls.append(snap.path))
    s.child("x").set_value(np.array([1.0, 2.0]))
    assert calls == [("x",)]
