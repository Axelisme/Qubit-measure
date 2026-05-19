import threading

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


def test_is_stop_returns_false_when_no_flag():
    s: TaskState[Result, Result, DictCfg] = TaskState(
        root_data=np.zeros(1), cfg=DictCfg()
    )
    assert s.is_stop() is False


def test_is_stop_returns_false_when_flag_not_set():
    flag = threading.Event()
    s: TaskState[Result, Result, DictCfg] = TaskState(
        root_data=np.zeros(1), cfg=DictCfg(), _stop_flag=flag
    )
    assert s.is_stop() is False


def test_is_stop_returns_true_when_flag_set():
    flag = threading.Event()
    flag.set()
    s: TaskState[Result, Result, DictCfg] = TaskState(
        root_data=np.zeros(1), cfg=DictCfg(), _stop_flag=flag
    )
    assert s.is_stop() is True


def test_set_stop_sets_underlying_event():
    flag = threading.Event()
    s: TaskState[Result, Result, DictCfg] = TaskState(
        root_data=np.zeros(1), cfg=DictCfg(), _stop_flag=flag
    )
    assert not flag.is_set()
    s.set_stop()
    assert flag.is_set()


def test_stop_flag_shared_across_child():
    flag = threading.Event()
    root: dict[str, Result] = {"x": np.zeros(1)}
    s: TaskState[dict[str, Result], dict[str, Result], DictCfg] = TaskState(
        root_data=root, cfg=DictCfg(), _stop_flag=flag
    )
    c = s.child("x", child_type=Result)
    flag.set()
    assert c.is_stop() is True


def test_child_with_cfg_replaces_cfg_and_shares_stop_flag():
    flag = threading.Event()
    root: dict[str, Result] = {"x": np.zeros(1)}
    s: TaskState[dict[str, Result], dict[str, Result], DictCfg] = TaskState(
        root_data=root, cfg=DictCfg(), _stop_flag=flag
    )
    new_cfg = DictCfg.model_validate({"k": 99})
    c = s.child_with_cfg("x", new_cfg, child_type=Result)
    assert c.cfg is not new_cfg  # deepcopy
    flag.set()
    assert c.is_stop() is True


def test_value_property_returns_current_node_data():
    arr = np.array([1.0, 2.0, 3.0])
    root: dict[str, Result] = {"x": arr}
    s: TaskState[dict[str, Result], dict[str, Result], DictCfg] = TaskState(
        root_data=root, cfg=DictCfg()
    )
    c = s.child("x", child_type=Result)
    assert np.array_equal(c.value, arr)  # type: ignore[arg-type]


def test_get_target_navigates_list_by_int_index():
    inner = np.zeros(2)
    lst: list[Result] = [np.ones(2), inner]
    root: list[Result] = lst
    s: TaskState[list[Result], list[Result], DictCfg] = TaskState(
        root_data=root, cfg=DictCfg()
    )
    c = s.child(1)
    assert c.value is inner  # type: ignore[comparison-overlap]


def test_get_target_raises_on_non_container():
    import dataclasses

    arr = np.zeros(2)
    root: dict[str, Result] = {"x": arr}
    s: TaskState[dict[str, Result], dict[str, Result], DictCfg] = TaskState(
        root_data=root, cfg=DictCfg()
    )
    # path descends into an ndarray (not Mapping or list) — should raise
    bad_child = dataclasses.replace(s, path=("x", 0))
    with pytest.raises(ValueError, match="Expected Mapping or list"):
        _ = bad_child.value
