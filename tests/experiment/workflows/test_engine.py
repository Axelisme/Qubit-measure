"""Public Engine control, isolation, and journal-authoritative commit contracts."""

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Event

import pytest
from pydantic import ValidationError
from zcu_tools.experiment.workflows import (
    Actor,
    DeviceSnapshot,
    Done,
    Effect,
    Engine,
    EnginePorts,
    InitEnv,
    InvalidRunState,
    MissingCapability,
    Next,
    RevisionConflict,
    RunIdentity,
    RunMismatch,
    RunPaths,
    RunStatus,
    Step,
    TunableChange,
    WorkflowEnv,
    workflow,
)

from ._engine_fakes import START, Plan, RecordingBar, Rig, State, Tunables


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


class ReadCallbackBar(RecordingBar):
    """Read real progress with an optional one-shot ``on_read`` host callback.

    None leaves reads unchanged. The getter clears the callback before calling
    it, so control methods may take nested progress snapshots safely.
    """

    on_read: Callable[[], None] | None = None

    @property
    def n(self) -> int | float:
        callback = self.on_read
        self.on_read = None
        if callback is not None:
            callback()
        return super().n


@pytest.fixture
def capture_bar(rig: Rig) -> ReadCallbackBar:
    bar = ReadCallbackBar(total=2, desc="work")

    def progress(**_kwargs: object) -> ReadCallbackBar:
        return bar

    rig.bars.append(bar)
    rig.engine = Engine(
        EnginePorts(rig.plots, progress, rig.clock, context=7, devices=rig.devices)
    )
    return bar


def points(
    env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
) -> Step[State, int]:
    if state.index == plan.count:
        return Done()
    yield from env.wait_until(START)
    state.index += 1
    return Next(tun.value, state)


def start_replacement(rig: Rig) -> RunStatus:
    """Start offline points on the same Engine with fresh paths and run ID new."""

    def replacement(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        return points(env, plan, tun, state)

    declared = workflow(
        "replacement",
        plan=Plan,
        tunables=Tunables,
        state=State,
        record=int,
        init=rig.init,
    )(replacement)
    return rig.engine.start(
        declared,
        plan=Plan(count=1),
        tunables=Tunables(),
        paths=RunPaths(rig.root / "new-meta", rig.root / "new-data"),
        identity=RunIdentity("new", DeviceSnapshot(()), "offline"),
    )


def test_start_is_deferred_and_commits_precede_done(rig: Rig) -> None:
    assert rig.engine.status() is None
    rig.start(points)
    assert rig.initialized == []
    assert rig.kinds() == ("run_started",)
    initial = rig.engine.status()
    assert initial is not None
    assert (
        initial.lifecycle,
        initial.call_seq,
        initial.committed_seq,
        initial.revision,
    ) == ("running", None, 0, 0)

    final = rig.engine.execute("run")
    assert rig.initialized == [START]
    assert (final.lifecycle, final.call_seq, final.committed_seq) == ("done", 3, 2)
    assert rig.kinds() == (
        "run_started",
        "iteration_started",
        "step_committed",
        "iteration_started",
        "step_committed",
        "iteration_started",
        "step_finished",
        "run_ended",
    )
    assert [record.call_seq for record in rig.engine.records("run")] == [1, 2]
    assert [record.commit_seq for record in rig.engine.records("run", since=1)] == [2]
    assert rig.engine.records("run", since=2) == ()
    assert len(tuple((rig.paths.data_root / "iter").iterdir())) == 3
    assert (rig.paths.data_root / "iter/000003/files").is_dir()
    manifest = json.loads((rig.paths.metadata_root / "manifest.json").read_text())
    assert manifest["lifecycle"] == "done"
    assert manifest["ended_at"] is not None
    assert rig.engine.stop("run") == final


def test_revision_is_captured_once_and_nested_inputs_are_detached(rig: Rig) -> None:
    seen: list[tuple[list[int], list[int], int]] = []
    caller_plan = Plan()

    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        if state.index == plan.count:
            return Done()
        seen.append((plan.values.copy(), state.values.copy(), tun.value))
        plan.values.append(99)
        state.values.append(tun.value)
        if state.index == 0:
            changed = rig.engine.update_tunables(
                "run",
                (TunableChange("value", 8),),
                expected_revision=0,
                actor=Actor("agent", "test"),
            )
            assert changed.revision == 1
        yield from env.wait_until(START)
        state.index += 1
        return Next(tun.value, state)

    rig.start(step, plan=caller_plan)
    caller_plan.values.append(42)
    assert rig.engine.execute("run").lifecycle == "done"
    assert seen == [([10], [], 1), ([10], [1], 8)]
    records = rig.engine.records("run")
    assert [(record.revision, record.encoded_record.value) for record in records] == [
        (0, 1),
        (1, 8),
    ]
    assert rig.kinds().index("tunables_changed") < rig.kinds().index("step_committed")


def test_pause_discards_mutated_copy_and_resume_repeats_deadline(rig: Rig) -> None:
    deadline = START + timedelta(seconds=10)
    seen: list[tuple[int, list[int], int]] = []
    closed: list[int] = []

    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        if state.index == plan.count:
            return Done()
        env.pbar("points", plan.count).set_progress(state.index)
        seen.append((state.index, state.values.copy(), tun.value))
        state.values.append(tun.value)
        try:
            yield from env.wait_until(deadline)
        finally:
            closed.append(state.index)
        state.index += 1
        return Next(tun.value, state)

    rig.start(step)
    rig.clock.on_wait = lambda _signal: rig.engine.pause("run")
    paused = rig.engine.execute("run")
    assert (paused.lifecycle, paused.committed_seq, paused.call_seq) == ("paused", 0, 1)
    assert seen == [(0, [], 1)]
    assert closed == [0]
    assert rig.bars[0].closed is False
    assert rig.kinds().count("step_discarded") == 1
    assert rig.engine.pause("run") == paused
    rig.engine.update_tunables(
        "run",
        (TunableChange("value", 3),),
        expected_revision=0,
        actor=Actor("user", "test"),
    )
    rig.clock.on_wait = None
    assert rig.engine.resume("run", devices=DeviceSnapshot(())).lifecycle == "running"
    final = rig.engine.execute("run")
    assert final.lifecycle == "done"
    assert final.call_seq == 4
    assert rig.initialized == [START]
    assert rig.clock.targets == [deadline, deadline]
    assert seen == [(0, [], 1), (0, [], 3), (1, [3], 3)]
    assert [record.call_seq for record in rig.engine.records("run")] == [2, 3]
    assert rig.bars[0].closed is True
    assert len(rig.bars) == 1
    assert final.progress[0].completed == 1


@pytest.mark.parametrize("control", ["pause", "stop"])
def test_next_return_is_committed_before_pending_control(
    rig: Rig, control: str
) -> None:
    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        if state.index == plan.count:
            return Done()
        yield from env.wait_until(START)
        if state.index == 0:
            if control == "pause":
                rig.engine.pause("run")
            else:
                rig.engine.stop("run")
        state.index += 1
        return Next(tun.value, state)

    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == ("paused" if control == "pause" else "stopped")
    assert status.committed_seq == 1
    assert rig.engine.records("run")[0].call_seq == 1
    assert "step_discarded" not in rig.kinds()
    if control == "pause":
        rig.engine.resume("run", devices=DeviceSnapshot(()))
        assert rig.engine.execute("run").lifecycle == "done"
        assert [record.call_seq for record in rig.engine.records("run")] == [1, 2]


def test_thread_safe_requests_do_not_wait_and_stop_upgrades_pause(rig: Rig) -> None:
    entered, release = Event(), Event()
    signals: list[bool] = []

    def wait(signal: object) -> None:
        entered.set()
        assert release.wait(5), "test did not release the engine thread"
        signals.append(True)

    rig.clock.on_wait = wait

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        yield from env.wait_until(START + timedelta(seconds=20))
        return Next(1, state)

    rig.start(step)
    with ThreadPoolExecutor(max_workers=1) as pool:
        segment = pool.submit(rig.engine.execute, "run")
        try:
            assert entered.wait(5), "engine did not reach the injected wait seam"
            assert rig.engine.pause("run").lifecycle == "pausing"
            assert rig.engine.pause("run").lifecycle == "pausing"
            assert not segment.done()
            with pytest.raises(InvalidRunState):
                rig.engine.execute("run")
            with pytest.raises(InvalidRunState):
                rig.engine.resume("run", devices=DeviceSnapshot(()))
            assert rig.engine.stop("run").lifecycle == "stopping"
            assert rig.engine.stop("run").lifecycle == "stopping"
            with pytest.raises(InvalidRunState):
                rig.engine.pause("run")
            with pytest.raises(InvalidRunState):
                rig.engine.update_tunables(
                    "run",
                    (TunableChange("value", 2),),
                    expected_revision=0,
                    actor=Actor("user", "test"),
                )
        finally:
            release.set()
        assert segment.result(timeout=5).lifecycle == "stopped"
    assert signals == [True]
    assert rig.kinds().count("pause_requested") == 1
    assert rig.kinds().count("stop_requested") == 1
    assert rig.engine.records("run") == ()
    discarded = next(
        event for event in rig.events() if event["kind"] == "step_discarded"
    )
    assert discarded["reason"] == "stop"


def test_final_capture_rejects_resume_until_paused_segment_returns(
    rig: Rig, capture_bar: ReadCallbackBar
) -> None:
    attempts: list[str] = []

    def reject_takeover() -> None:
        attempts.append("resume")
        with pytest.raises(InvalidRunState):
            rig.engine.resume("run", devices=DeviceSnapshot(()))
        with pytest.raises(InvalidRunState):
            rig.engine.execute("run")

    def pause(_signal: object) -> None:
        rig.engine.pause("run")

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        bar = env.pbar("work", 2)
        bar.set_progress(1)
        try:
            yield from env.wait_until(START + timedelta(seconds=20))
            return Next(1, state)
        finally:
            capture_bar.on_read = reject_takeover

    rig.clock.on_wait = pause
    rig.start(step)
    paused = rig.engine.execute("run")
    assert attempts == ["resume"]
    assert (
        paused.lifecycle,
        paused.call_seq,
        paused.committed_seq,
        paused.revision,
    ) == ("paused", 1, 0, 0)
    assert paused.progress[0].completed == 1

    rig.engine.update_tunables(
        "run",
        (TunableChange("value", 9),),
        expected_revision=0,
        actor=Actor("user", "test"),
    )
    assert rig.engine.resume("run", devices=DeviceSnapshot(())).lifecycle == "running"
    resumed = rig.engine.execute("run")
    assert (resumed.lifecycle, resumed.call_seq, resumed.revision) == ("paused", 2, 1)
    assert attempts == ["resume", "resume"]
    capture_bar.set_progress(2)
    assert (paused.lifecycle, paused.call_seq, paused.revision) == ("paused", 1, 0)
    assert paused.progress[0].completed == 1


def test_final_capture_rejects_replacement_until_terminal_segment_returns(
    rig: Rig, capture_bar: ReadCallbackBar
) -> None:
    attempts: list[str] = []

    def reject_takeover() -> None:
        attempts.append("start")
        with pytest.raises(InvalidRunState):
            start_replacement(rig)

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        bar = env.pbar("work", 2)
        bar.set_progress(1)
        yield from env.wait_until(START)
        capture_bar.on_read = reject_takeover
        return Done()

    rig.start(step)
    old = rig.engine.execute("run")
    assert attempts == ["start"]
    assert (old.run_id, old.lifecycle, old.call_seq, old.committed_seq) == (
        "run",
        "done",
        1,
        0,
    )
    assert old.progress[0].completed == 1
    assert capture_bar.closed
    assert start_replacement(rig).lifecycle == "running"
    new = rig.engine.execute("new")
    assert (new.run_id, new.lifecycle, new.committed_seq) == ("new", "done", 1)
    assert (old.run_id, old.lifecycle, old.call_seq, old.committed_seq) == (
        "run",
        "done",
        1,
        0,
    )
    assert old.progress[0].completed == 1


@pytest.mark.parametrize(
    ("ending", "lifecycle", "reason"),
    [
        ("pause", "paused", None),
        ("done", "done", None),
        ("failure", "failed", "RuntimeError: producer failed"),
    ],
)
def test_final_capture_failure_releases_reservation_and_preserves_producer_cause(
    rig: Rig,
    capture_bar: ReadCallbackBar,
    ending: str,
    lifecycle: str,
    reason: str | None,
) -> None:
    def fail_capture() -> None:
        raise LookupError("capture failed")

    def pause(_signal: object) -> None:
        rig.engine.pause("run")

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        env.pbar("work", 2).set_progress(1)
        try:
            if ending == "failure":
                raise RuntimeError("producer failed")
            yield from env.wait_until(START + timedelta(seconds=20))
            return Done()
        finally:
            capture_bar.on_read = fail_capture

    if ending == "pause":
        rig.clock.on_wait = pause
    rig.start(step)
    with pytest.raises(LookupError, match="capture failed"):
        rig.engine.execute("run")
    status = rig.engine.status()
    assert status is not None
    assert (status.lifecycle, status.call_seq, status.committed_seq, status.reason) == (
        lifecycle,
        1,
        0,
        reason,
    )
    assert status.progress[0].completed == 1
    assert rig.engine.records("run") == ()
    if ending == "failure":
        event = next(e for e in rig.events() if e["kind"] == "step_failed")
        error = event["error"]
        assert isinstance(error, dict) and error["message"] == "producer failed"
    if ending == "pause":
        assert (
            rig.engine.resume("run", devices=DeviceSnapshot(())).lifecycle == "running"
        )
        rig.engine.stop("run")
        assert rig.engine.execute("run").lifecycle == "stopped"
    else:
        assert start_replacement(rig).lifecycle == "running"
        assert rig.engine.execute("new").committed_seq == 1


@pytest.mark.parametrize("control", ["pause", "stop"])
def test_request_before_execute_never_calls_init(rig: Rig, control: str) -> None:
    rig.start(points)
    if control == "pause":
        rig.engine.pause("run")
    else:
        rig.engine.stop("run")
    status = rig.engine.execute("run")
    assert status.lifecycle == ("paused" if control == "pause" else "stopped")
    assert rig.initialized == []
    assert status.call_seq is None
    if control == "pause":
        assert rig.engine.stop("run").lifecycle == "stopped"
    assert "iteration_started" not in rig.kinds()


def test_bad_revision_or_values_do_not_fail_or_publish(rig: Rig) -> None:
    rig.start(points)
    with pytest.raises(RevisionConflict) as caught:
        rig.engine.update_tunables(
            "run",
            (TunableChange("value", 2),),
            expected_revision=1,
            actor=Actor("user", "test"),
        )
    assert (caught.value.expected, caught.value.actual) == (1, 0)
    with pytest.raises(ValidationError):
        rig.engine.update_tunables(
            "run",
            (TunableChange("value", -1),),
            expected_revision=0,
            actor=Actor("user", "test"),
        )
    assert rig.engine.tunables("run").revision == 0
    assert rig.kinds() == ("run_started",)
    assert rig.engine.execute("run").lifecycle == "done"


def test_update_journal_failure_keeps_revision_and_fails_idle_run(rig: Rig) -> None:
    rig.start(points)
    journal = rig.paths.metadata_root / "journal.jsonl"
    journal.unlink()
    journal.mkdir()
    with pytest.raises(IsADirectoryError):
        rig.engine.update_tunables(
            "run",
            (TunableChange("value", 2),),
            expected_revision=0,
            actor=Actor("user", "test"),
        )
    status = rig.engine.status()
    assert status is not None and status.lifecycle == "failed"
    assert status.revision == 0
    assert "IsADirectoryError" in str(status.reason)
    assert rig.engine.tunables("run").values == {"value": 1}
    assert rig.initialized == []
    with pytest.raises(InvalidRunState):
        rig.engine.execute("run")


def test_wrong_run_and_invalid_lifecycle_are_explicit(rig: Rig) -> None:
    with pytest.raises(RunMismatch) as caught:
        rig.engine.execute("missing")
    assert (caught.value.expected, caught.value.actual) == ("missing", None)
    rig.start(points)
    with pytest.raises(RunMismatch) as caught:
        rig.engine.stop("other")
    assert caught.value.actual == "run"
    with pytest.raises(InvalidRunState):
        rig.start(points)
    with pytest.raises(InvalidRunState):
        rig.engine.resume("run", devices=DeviceSnapshot(()))
    rig.engine.execute("run")
    with pytest.raises(InvalidRunState):
        rig.engine.execute("run")
    with pytest.raises(InvalidRunState):
        rig.engine.pause("run")
    with pytest.raises(ValueError, match="since"):
        rig.engine.records("run", since=-1)

    def replacement(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        return points(env, plan, tun, state)

    declared = workflow(
        "replacement",
        plan=Plan,
        tunables=Tunables,
        state=State,
        record=int,
        init=rig.init,
    )(replacement)
    rig.engine.start(
        declared,
        plan=Plan(count=1),
        tunables=Tunables(),
        paths=RunPaths(rig.root / "new-meta", rig.root / "new-data"),
        identity=RunIdentity("new", DeviceSnapshot(()), "offline"),
    )
    with pytest.raises(RunMismatch):
        rig.engine.records("run")
    assert rig.engine.execute("new").committed_seq == 1


def test_start_revalidates_constructed_model_and_requires_before_creating_paths(
    rig: Rig,
) -> None:
    def requires_soc(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        return points(env, plan, tun, state)

    declared = workflow(
        "requires-soc",
        plan=Plan,
        tunables=Tunables,
        state=State,
        record=int,
        init=rig.init,
        requires=("soc",),
    )(requires_soc)
    with pytest.raises(MissingCapability, match="soc"):
        rig.engine.start(
            declared,
            plan=Plan(),
            tunables=Tunables(),
            paths=rig.paths,
            identity=RunIdentity("run", DeviceSnapshot(()), "offline"),
        )
    assert not rig.paths.metadata_root.exists()
    assert rig.engine.status() is None

    def fresh(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        return points(env, plan, tun, state)

    with pytest.raises(ValidationError):
        rig.start(fresh, plan=Plan.model_construct(count=-1, values=[]))
    assert not rig.paths.data_root.exists()


def test_init_failure_is_failed_without_step_or_commit(rig: Rig) -> None:
    def broken_init(_env: InitEnv[int], _plan: Plan) -> State:
        raise OSError("init failed")

    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        return points(env, plan, tun, state)

    declared = workflow(
        "bad-init",
        plan=Plan,
        tunables=Tunables,
        state=State,
        record=int,
        init=broken_init,
    )(step)
    rig.engine.start(
        declared,
        plan=Plan(),
        tunables=Tunables(),
        paths=rig.paths,
        identity=RunIdentity("run", DeviceSnapshot(()), "offline"),
    )
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert "init failed" in str(status.reason)
    assert status.call_seq is None
    assert rig.engine.records("run") == ()
    assert rig.kinds() == ("run_started", "run_ended")


@pytest.mark.parametrize("copied", ["state", "record"])
def test_deepcopy_failure_during_next_never_commits(rig: Rig, copied: str) -> None:
    class BrokenState(State):
        def __deepcopy__(self, _memo: dict[int, object]) -> State:
            raise ValueError("state copy failed")

    class BrokenRecord(int):
        def __deepcopy__(self, _memo: dict[int, object]) -> int:
            raise ValueError("record copy failed")

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        yield from env.wait_until(START)
        if copied == "state":
            return Next(1, BrokenState(index=1))
        return Next(BrokenRecord(1), state)

    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert f"{copied} copy failed" in str(status.reason)
    assert status.committed_seq == 0
    assert rig.engine.records("run") == ()
    assert "step_failed" in rig.kinds()


def test_commit_journal_failure_does_not_publish_record_or_next_state(rig: Rig) -> None:
    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        yield from env.wait_until(START)
        journal = rig.paths.metadata_root / "journal.jsonl"
        journal.unlink()
        journal.mkdir()
        state.index = 1
        return Next(1, state)

    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert "IsADirectoryError" in str(status.reason)
    assert status.committed_seq == 0
    assert rig.engine.records("run") == ()


def test_unknown_yield_fails_at_effect_boundary_with_authoring_hint(rig: Rig) -> None:
    class UnknownEffect(Effect):
        def __init__(self) -> None:
            pass

    def step(
        _env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        yield UnknownEffect()
        return Next(1, state)

    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert "yield from env.<effect>" in str(status.reason)
    assert status.committed_seq == 0
