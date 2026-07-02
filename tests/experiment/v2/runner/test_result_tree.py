from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.v2.runner import ResultTree, ResultUpdateEvent, Schedule
from zcu_tools.experiment.v2.utils import Result

LeafResult: TypeAlias = dict[str, NDArray[np.float64]]
TreeRow: TypeAlias = dict[str, LeafResult]


@dataclass(frozen=True)
class TreeEnv:
    label: str
    flux_values: np.ndarray


def _row() -> TreeRow:
    return {
        "freq": {
            "raw": np.full((2,), np.nan, dtype=np.float64),
            "fit": np.array(np.nan, dtype=np.float64),
        },
        "t1": {
            "value": np.array(np.nan, dtype=np.float64),
        },
    }


def _tree() -> ResultTree[TreeEnv]:
    data = cast(list[dict[str, Result]], [_row(), _row()])
    return ResultTree[TreeEnv](data, outer_values=np.array([0.25, 0.5]))


def test_result_tree_node_set_updates_nested_data() -> None:
    tree = _tree()

    tree.at(0).child("freq").child("fit").set(np.array(7.0), flush=True)

    freq_row = cast(LeafResult, tree.data[0]["freq"])
    np.testing.assert_allclose(freq_row["fit"], np.array(7.0))
    result = cast(LeafResult, tree.measurement_result("freq"))
    np.testing.assert_allclose(result["fit"], np.array([7.0, np.nan]))


def test_result_tree_schedule_set_emits_per_measurement_event() -> None:
    tree = _tree()
    env = TreeEnv(label="run", flux_values=np.array([0.25, 0.5]))
    events: list[ResultUpdateEvent[TreeEnv, Any]] = []
    tree.measurement_node("freq").subscribe(events.append)

    cfg: dict[str, object] = {}
    with Schedule(cfg, tree, env=env) as sched:
        for _, flux_step in sched.scan("flux", env.flux_values):
            if flux_step.index == 1:
                flux_step.child("freq").set_data(
                    {
                        "raw": np.array([3.0, 4.0]),
                        "fit": np.array(12.0),
                    },
                    flush=True,
                )

    assert len(events) == 1
    event = events[0]
    assert event.measurement_name == "freq"
    assert event.outer_index == 1
    assert event.outer_value == 0.5
    assert event.env is env
    assert event.flush is True
    np.testing.assert_allclose(event.result["fit"], np.array([np.nan, 12.0]))
    np.testing.assert_allclose(event.result["raw"][1], np.array([3.0, 4.0]))


def test_result_tree_child_buffer_writes_leaf_and_flushes_node() -> None:
    tree = _tree()
    env = TreeEnv(label="run", flux_values=np.array([0.25, 0.5]))
    events: list[ResultUpdateEvent[TreeEnv, Any]] = []
    tree.measurement_node("freq").subscribe(events.append)

    cfg: dict[str, object] = {}
    with Schedule(cfg, tree, env=env) as sched:
        _, flux_step = next(sched.scan("flux", env.flux_values))
        raw_step = flux_step.child("freq").child("raw")
        local_buffer = raw_step.buffer((2,), dtype=np.float64)
        local_buffer.set(np.array([1.0, 2.0]))
        raw_step.trigger_update(flush=True)

    freq_row = cast(LeafResult, tree.data[0]["freq"])
    np.testing.assert_allclose(freq_row["raw"], np.array([1.0, 2.0]))
    assert [event.flush for event in events] == [False, True]
    assert [event.outer_index for event in events] == [0, 0]


def test_result_tree_invalidates_only_updated_measurement_cache() -> None:
    tree = _tree()
    env = TreeEnv(label="run", flux_values=np.array([0.25, 0.5]))
    freq_before = tree.measurement_result("freq")
    t1_before = tree.measurement_result("t1")

    cfg: dict[str, object] = {}
    with Schedule(cfg, tree, env=env) as sched:
        _, flux_step = next(sched.scan("flux", env.flux_values))
        flux_step.child("freq").child("fit").set_data(np.array(9.0), flush=True)

    assert tree.measurement_result("freq") is not freq_before
    assert tree.measurement_result("t1") is t1_before
    freq_result = cast(LeafResult, tree.measurement_result("freq"))
    np.testing.assert_allclose(freq_result["fit"][0], 9.0)


def test_signal_buffer_flush_keeps_public_update_shape() -> None:
    from zcu_tools.experiment.v2.runner import SignalBuffer

    updates: list[np.ndarray] = []
    buffer = SignalBuffer(
        (2,),
        dtype=np.float64,
        on_update=lambda data: updates.append(data.copy()),
        update_interval=None,
    )

    cfg: dict[str, object] = {}
    with Schedule(cfg, buffer) as sched:
        for value, step in sched.scan("point", [0.0, 1.0]):
            step.set_data(value + 1.0, flush=True)

    assert len(updates) == 2
    np.testing.assert_allclose(updates[-1], np.array([1.0, 2.0]))
