"""Explicit scope lifetime, naming and cancellation through public contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from zcu_tools import experiment
from zcu_tools.experiment.stop_signal import ScheduleOutcomeError
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.program.v2 import ProgramV2Cfg

if TYPE_CHECKING:
    from zcu_tools.experiment import Scope


def make_root() -> Scope:
    assert hasattr(experiment, "Scope"), (
        "experiment must expose the explicit Scope contract"
    )
    from zcu_tools.experiment import Scope

    return Scope(kind="run")


def test_child_lifetime_and_birth_numbers_survive_exception() -> None:
    root = make_root()
    assert root.active_path == "run"
    with root.child("calibration") as child:
        assert child.active_path == "run/calibration#1"
        assert root.active_path == child.active_path
        with child.child("fit"):
            assert root.active_path == "run/calibration#1/fit#1"
    assert root.active_path == "run"
    with pytest.raises(ValueError, match="author failure"):
        with root.child("calibration") as child:
            assert child.active_path == "run/calibration#2"
            raise ValueError("author failure")
    assert root.active_path == "run"
    assert not root.cancel("calibration").cancelled


@pytest.mark.parametrize(
    "label",
    [
        "",
        "bad/name",
        "bad#name",
        "bad:name",
        "operation",
        "schedule",
        "acquire",
        "iteration",
        "workflow",
        "recipe",
        "run",
    ],
)
def test_author_labels_reject_path_syntax_and_reserved_kinds(label: str) -> None:
    root = make_root()
    with pytest.raises(ValueError):
        with root.child(label):
            pass
    assert root.active_path == "run"


def test_named_operation_selector_requires_sequence_for_live_siblings() -> None:
    root = make_root()
    with root.branch("operation", name="T1") as first:
        assert first.active_path == "run/operation:T1#1"
        with root.branch("operation", name="T1") as second:
            result = root.cancel("operation:T1")
            assert not result.cancelled
            assert result.reason == "ambiguous"
            assert result.candidates == ["run/operation:T1#1", "run/operation:T1#2"]
            assert root.active_path == "run"
            assert root.live_paths == result.candidates
            assert not first.cancelled and not second.cancelled
            assert root.cancel("operation:T1#1").cancelled
            assert first.cancelled and not second.cancelled
            assert not root.cancel("operation:T1#3").cancelled
        assert root.active_path == "run/operation:T1#1"
    assert root.active_path == "run"


def test_full_path_disambiguates_equal_sequences_under_different_parents() -> None:
    root = make_root()
    with root.child("left") as left, root.child("right") as right:
        with left.branch("operation"), right.branch("operation") as selected:
            assert root.cancel("operation#1").reason == "ambiguous"
            assert root.live_paths == [
                "run/left#1/operation#1",
                "run/right#1/operation#1",
            ]
            assert root.cancel("run/right#1/operation#1").cancelled
            assert selected.cancelled
            assert not left.stop.partial_result


def test_release_is_idempotent_and_closed_scope_rejects_new_children() -> None:
    root = make_root()
    child = root.child("point")
    child.release()
    child.release()
    with pytest.raises(RuntimeError):
        child.child("nested")
    assert root.live_paths == ["run"]


def test_cancel_is_descendant_only_and_born_cancelled_is_author_visible() -> None:
    root = make_root()
    with root.child("point") as point:
        assert root.cancel("point#1", expected=1).cancelled
        assert point.cancelled
        assert not root.cancelled
        with point.child("measure") as measure:
            assert measure.cancelled
    with root.child("point") as following:
        assert not following.cancelled
        refused = root.cancel("point", expected=1)
        assert not refused.cancelled
        assert refused.reason == "sequence mismatch"


def test_error_and_partial_release_stop_at_operation_boundary() -> None:
    root = make_root()
    cause = OSError("readout failed")
    with root.child("point") as point:
        with point.branch("operation") as operation:
            with operation.branch("schedule") as schedule:
                with schedule.branch("acquire") as acquire:
                    acquire.stop.set_error("failed", "readout failed", cause)
                assert schedule.stop.partial_result
                assert not schedule.cancelled
            assert operation.stop.partial_result
            assert not operation.cancelled
            with pytest.raises(ScheduleOutcomeError) as caught:
                operation.stop.raise_if_error()
            assert caught.value.exception is cause
        assert not point.stop.partial_result
        assert point.stop.error is None
    assert not root.stop.partial_result
    assert root.stop.error is None


def test_schedule_scope_and_acquire_cancel_preserve_next_acquisition() -> None:
    root = make_root()

    class Program:
        cfg_model = ProgramV2Cfg()
        calls = 0

        def acquire(self, _soc: object, **_kwargs: object) -> np.ndarray:
            self.calls += 1
            assert root.active_path == f"run/schedule:readout#1/acquire#{self.calls}"
            if self.calls == 1:
                assert root.cancel("acquire").cancelled
            return np.array([float(self.calls)])

        acquire_decimated = acquire

    program = Program()
    with Schedule(
        ProgramV2Cfg(), SignalBuffer(1), scope=root, name="readout"
    ) as schedule:
        assert schedule.scope.active_path == "run/schedule:readout#1"
        builder = schedule.prog_builder(object(), object())
        builder.run_program(program, raw2signal_fn=lambda value: value)
        assert not schedule.scope.cancelled
        builder.run_program(program, raw2signal_fn=lambda value: value)
        assert schedule.data[0] == 2.0
    assert root.active_path == "run"
    assert root.stop.partial_result
    assert not root.cancelled


def test_gui_run_root_receives_schedule_error_without_cancellation() -> None:
    root = make_root()
    with Schedule(ProgramV2Cfg(), scope=root) as schedule:
        schedule.mark_failed(OSError("readout failed"))
    assert not root.cancelled
    assert root.stop.partial_result
    with pytest.raises(ScheduleOutcomeError, match="readout failed"):
        root.stop.raise_if_error()
