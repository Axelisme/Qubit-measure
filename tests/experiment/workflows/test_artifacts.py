"""Observable run-artifact behavior through the package's typed I/O seam."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock

import pytest
from pydantic import TypeAdapter
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.acquisition import capture_software_provenance
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import (
    Aborted,
    Done,
    Engine,
    EnginePorts,
    Next,
    Run,
    Step,
    WorkflowEnv,
)
from zcu_tools.experiment.workflows.artifacts import RunArtifacts
from zcu_tools.experiment.workflows.encoding import encode_record
from zcu_tools.experiment.workflows.journal import (
    ExperimentFailed,
    IterationStarted,
    Paused,
    PauseRequested,
    Resumed,
    RunEnded,
    RunMetadata,
    StepCommitted,
    StepDiscarded,
    StepFailed,
    StepFinished,
    StopRequested,
    TunablesChanged,
    describe_error,
)
from zcu_tools.experiment.workflows.models import (
    Actor,
    RunIdentity,
    RunPaths,
)

from ._engine_fakes import (
    START,
    Plan,
    Rig,
    State,
    Tunables,
)
from ._engine_fakes import (
    Cfg as EngineCfg,
)
from ._engine_fakes import (
    ManualClock as EngineClock,
)


class ManualClock:
    """Deterministic UTC clock; artifact tests never wait."""

    def __init__(self) -> None:
        self.value = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self.value

    def wait_until(self, target: datetime, cancel_signal: StopSignal) -> None:
        raise AssertionError("Artifact storage must not wait or use hardware")


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def metadata(tmp_path: Path, clock: ManualClock) -> RunMetadata:
    return RunMetadata(
        identity=RunIdentity(
            "id-not-the-directory-slug",
            SoftwareProvenance(
                software_versions={},
                git_commit=None,
                git_dirty=None,
                qick_version=None,
                soc_fingerprint=None,
                hostname="test-host",
            ),
        ),
        workflow="storage-contract",
        plan={"targets": [1, 2]},
        tunables={"reps": 2},
        roots=RunPaths(tmp_path / "metadata-slug", tmp_path / "data-slug"),
        started_at=clock.now(),
    )


@pytest.fixture
def artifacts(metadata: RunMetadata, clock: ManualClock) -> RunArtifacts:
    return RunArtifacts(metadata, clock, RLock())


def test_start_documents_are_detached_and_use_identity_not_slug(
    artifacts: RunArtifacts, metadata: RunMetadata
) -> None:
    if isinstance(metadata.plan, dict):
        metadata.plan["targets"] = [9]
    manifest_path = artifacts.paths.metadata_root / "manifest.json"
    initial = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert initial["format"] == "zcu-workflow-run"
    assert initial["format_version"] == 1
    assert initial["lifecycle"] == "running"
    assert initial["ended_at"] is None
    assert initial["journal"] == str(artifacts.journal_path)
    assert initial["initial_tunables"] == {"reps": 2}
    assert initial["plan"] == {"targets": [1, 2]}
    lines = artifacts.journal_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["kind"] == "run_started"
    assert event["revision"] == 0
    assert event["seq"] == 1
    assert event["run_id"] == metadata.identity.run_id
    assert event["identity"] == initial["identity"]
    assert (
        event["roots"]
        == initial["roots"]
        == {
            "metadata_root": str(metadata.roots.metadata_root),
            "data_root": str(metadata.roots.data_root),
        }
    )
    assert datetime.fromisoformat(event["time"]) == metadata.started_at
    artifacts.set_lifecycle("paused")
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["plan"] == {
        "targets": [1, 2]
    }


@pytest.mark.parametrize("native_capture", [False, True])
def test_start_preserves_native_provenance_in_manifest_and_journal(
    tmp_path: Path, metadata: RunMetadata, clock: ManualClock, *, native_capture: bool
) -> None:
    provenance = (
        capture_software_provenance(source_directory=tmp_path)
        if native_capture
        else SoftwareProvenance(
            software_versions={"test-package": "1.2.3"},
            git_commit=None,
            git_dirty=None,
            qick_version=None,
            soc_fingerprint=None,
            hostname=None,
        )
    )
    startup = replace(
        metadata, identity=replace(metadata.identity, provenance=provenance)
    )
    artifacts = RunArtifacts(startup, clock, RLock())
    manifest = json.loads(
        (artifacts.paths.metadata_root / "manifest.json").read_text(encoding="utf-8")
    )
    event = json.loads(
        artifacts.journal_path.read_text(encoding="utf-8").splitlines()[0]
    )
    expected = TypeAdapter(SoftwareProvenance).dump_python(provenance, mode="json")
    assert manifest["identity"]["provenance"] == expected
    assert event["identity"]["provenance"] == expected
    assert event["identity"] == manifest["identity"]


def test_relative_unicode_roots_remain_absolute_after_cwd_changes(
    tmp_path: Path,
    metadata: RunMetadata,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    roots = RunPaths(Path("metadata 資料"), Path("data 資料"))
    artifacts = RunArtifacts(replace(metadata, roots=roots), clock, RLock())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    iteration = artifacts.new_iteration(1)

    artifacts.append(StepFinished(1, 0, iteration.iteration_dir, "done", None))
    artifacts.set_lifecycle("done")
    assert iteration.root == tmp_path / "data 資料" / "iter/000001"
    assert artifacts.paths.metadata_root == tmp_path / "metadata 資料"
    assert artifacts.paths.data_root == tmp_path / "data 資料"
    assert not tuple(elsewhere.iterdir())
    manifest = json.loads(
        (artifacts.paths.metadata_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["lifecycle"] == "done"
    assert manifest["roots"]["data_root"] == str(tmp_path / "data 資料")


@pytest.mark.parametrize("call_seq", [0, -1, True])
def test_invalid_call_number_creates_no_iteration(
    artifacts: RunArtifacts, call_seq: int
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        artifacts.new_iteration(call_seq)
    assert not tuple(artifacts.paths.data_root.iterdir())


def test_invalid_start_time_creates_no_roots(
    metadata: RunMetadata, clock: ManualClock
) -> None:
    invalid = replace(metadata, started_at=datetime(2026, 10, 6))
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        RunArtifacts(invalid, clock, RLock())
    assert not metadata.roots.metadata_root.exists()
    assert not metadata.roots.data_root.exists()


def test_mandatory_startup_nonfinite_does_not_use_record_fallback(
    metadata: RunMetadata, clock: ManualClock
) -> None:
    invalid = replace(metadata, tunables={"reps": float("nan")})
    with pytest.raises(ValueError, match="Out of range float"):
        RunArtifacts(invalid, clock, RLock())
    assert not metadata.roots.metadata_root.exists()
    assert not metadata.roots.data_root.exists()


def test_journal_round_trips_all_outcomes_in_append_order(
    artifacts: RunArtifacts, clock: ManualClock
) -> None:
    iteration = artifacts.new_iteration(1)
    error = describe_error(ValueError("original cause"))
    events = (
        IterationStarted(1, 0, 0, iteration.iteration_dir),
        ExperimentFailed(1, 1, "run_probe", "retry exhausted", error),
        StepCommitted(
            1,
            1,
            0,
            encode_record({"t1": 8}),
            iteration.iteration_dir,
        ),
        StepFinished(1, 0, iteration.iteration_dir, "done", None),
        StepDiscarded(1, 0, iteration.iteration_dir, "pause"),
        StepFailed(1, 0, iteration.iteration_dir, error),
        TunablesChanged(Actor("agent", "協作者"), 0, 1, {"reps": 3}),
        PauseRequested(clock.now(), 1),
        Paused(1, 1),
        Resumed(1, 1),
        StopRequested(clock.now(), 1),
        RunEnded("failed", "original cause", 1, 1),
    )
    before = (artifacts.paths.metadata_root / "manifest.json").read_bytes()
    for event in events:
        artifacts.append(event)
    assert (artifacts.paths.metadata_root / "manifest.json").read_bytes() == before
    raw = artifacts.journal_path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert "協作者" in raw
    lines = [json.loads(line) for line in raw.splitlines()]
    assert [line["seq"] for line in lines] == list(range(1, len(events) + 2))
    assert [line["kind"] for line in lines[1:]] == [event.kind for event in events]
    committed = lines[3]
    assert committed["encoded_record"] == {
        "mode": "json",
        "value": {"t1": 8},
        "error": None,
    }
    assert lines[2]["error"]["message"] == "original cause"
    assert "ValueError: original cause" in lines[6]["error"]["traceback"]
    assert "result" not in lines[2]
    assert all(line["format_version"] == 1 for line in lines)


def test_lifecycle_manifest_replaces_only_on_change(
    artifacts: RunArtifacts, clock: ManualClock, metadata: RunMetadata
) -> None:
    path = artifacts.paths.metadata_root / "manifest.json"
    clock.value += timedelta(seconds=5)
    artifacts.set_lifecycle("paused")
    paused = json.loads(path.read_text(encoding="utf-8"))
    assert paused["lifecycle"] == "paused"
    assert paused["ended_at"] is None
    assert datetime.fromisoformat(paused["updated_at"]) == clock.now()
    clock.value += timedelta(seconds=5)
    artifacts.set_lifecycle("failed", "disk failure")
    ended = json.loads(path.read_text(encoding="utf-8"))
    assert ended["lifecycle"] == "failed"
    assert ended["reason"] == "disk failure"
    assert datetime.fromisoformat(ended["started_at"]) == metadata.started_at
    assert datetime.fromisoformat(ended["ended_at"]) == clock.now()
    assert not tuple(path.parent.glob(".manifest-*.tmp.json"))
    with pytest.raises(ValueError, match="lifecycle change"):
        artifacts.set_lifecycle("failed", "must not replace terminal reason")
    assert json.loads(path.read_text(encoding="utf-8"))["reason"] == "disk failure"


@pytest.mark.parametrize("call_seq", [1, 2, 1000000])
def test_iteration_numbers_do_not_wrap(artifacts: RunArtifacts, call_seq: int) -> None:
    iteration = artifacts.new_iteration(call_seq)
    assert iteration.iteration_dir == f"iter/{call_seq:06d}"
    assert iteration.root == artifacts.paths.data_root / iteration.iteration_dir
    assert iteration.root.is_dir()
    with pytest.raises(FileExistsError):
        artifacts.new_iteration(call_seq)


def test_manifest_failure_retains_last_document_and_temporary(
    artifacts: RunArtifacts, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = artifacts.paths.metadata_root / "manifest.json"
    before = path.read_bytes()
    cause = OSError("manifest replace failed")

    def broken_replace(source: Path, destination: Path) -> Path:
        raise cause

    clock.value += timedelta(seconds=10)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", broken_replace)
        with pytest.raises(OSError, match="replace failed") as caught:
            artifacts.set_lifecycle("paused")
        assert caught.value is cause
    assert path.read_bytes() == before
    assert len(tuple(path.parent.glob(".manifest-*.tmp.json"))) == 1
    artifacts.append(RunEnded("failed", str(cause), None, 0))
    artifacts.set_lifecycle("failed", str(cause))
    assert json.loads(path.read_text(encoding="utf-8"))["lifecycle"] == "failed"


def test_journal_io_failure_is_not_followed_by_invented_success(
    artifacts: RunArtifacts,
) -> None:
    artifacts.journal_path.unlink()
    artifacts.journal_path.mkdir()
    with pytest.raises(IsADirectoryError):
        artifacts.append(Paused(None, 0))
    with pytest.raises(RuntimeError, match="unusable after an I/O failure"):
        artifacts.append(RunEnded("failed", "journal broken", None, 0))
    assert not tuple(artifacts.journal_path.iterdir())


def test_nonfinite_mandatory_payload_fails_without_consuming_sequence(
    artifacts: RunArtifacts,
) -> None:
    change = TunablesChanged(Actor("user", "user"), 0, 1, {"reps": float("inf")})
    with pytest.raises(ValueError, match="Out of range float"):
        artifacts.append(change)
    artifacts.append(Paused(None, 0))
    lines = [
        json.loads(line)
        for line in artifacts.journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [line["seq"] for line in lines] == [1, 2]
    assert lines[-1]["kind"] == "paused"


@pytest.mark.parametrize("root_name", ["metadata_root", "data_root"])
def test_start_refuses_existing_root_without_creating_other(
    metadata: RunMetadata, clock: ManualClock, root_name: str
) -> None:
    root = getattr(metadata.roots, root_name)
    root.mkdir()
    (root / "keep.txt").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Run root already exists"):
        RunArtifacts(metadata, clock, RLock())
    other = (
        metadata.roots.data_root
        if root_name == "metadata_root"
        else metadata.roots.metadata_root
    )
    assert not other.exists()
    assert (root / "keep.txt").read_text(encoding="utf-8") == "existing"


def test_second_root_creation_failure_keeps_first_root(
    metadata: RunMetadata, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    cause = OSError("second disk refused creation")
    original = Path.mkdir

    def broken_mkdir(
        path: Path, mode: int = 0o777, *, parents: bool = False, exist_ok: bool = False
    ) -> None:
        if path == metadata.roots.data_root:
            raise cause
        original(path, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", broken_mkdir)
    with pytest.raises(OSError, match="second disk") as caught:
        RunArtifacts(metadata, clock, RLock())
    assert caught.value is cause
    assert metadata.roots.metadata_root.is_dir()
    assert not (metadata.roots.metadata_root / "manifest.json").exists()
    assert not metadata.roots.data_root.exists()


@pytest.mark.parametrize(
    "reference",
    [
        "../outside",
        "/outside",
        "iter/../outside",
        "iter\\outside",
        "C:/outside",
        "C:outside",
    ],
)
def test_journal_rejects_external_path_references(
    artifacts: RunArtifacts, reference: str
) -> None:
    before = artifacts.journal_path.read_bytes()
    with pytest.raises(ValueError, match="relative paths"):
        artifacts.append(IterationStarted(1, 0, 0, reference))
    assert artifacts.journal_path.read_bytes() == before


@pytest.fixture
def record_rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


def iteration_document(rig: Rig, call_seq: int) -> dict[str, object]:
    """Read a completed invocation through its on-disk JSON contract."""
    path = rig.paths.data_root / f"iter/{call_seq:06d}/record.json"
    assert path.is_file(), f"Completed invocation {call_seq} must have record.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_iteration_record_contains_inputs_and_journal_result_after_next_and_done(
    record_rig: Rig,
) -> None:
    rig = record_rig

    def experiment(run: Run[EngineCfg]) -> int:
        return run.cfg.value * 10

    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        assert not (env.iteration_dir / "record.json").exists()
        if state.index == plan.count:
            return Done()
        yield from env.run(experiment, EngineCfg(tun.value))
        state.index += 1
        return Next(tun.value, state)

    rig.start(step, plan=Plan(count=1, values=[10, 20]))
    assert not tuple(rig.paths.data_root.rglob("record.json"))
    final = rig.engine.execute("run")
    assert (final.lifecycle, final.committed_seq) == ("done", 1)
    committed = iteration_document(rig, 1)
    finished = iteration_document(rig, 2)
    for document in (committed, finished):
        assert set(document) == {
            "format",
            "format_version",
            "identity",
            "workflow",
            "plan",
            "tunables",
            "started",
            "result",
        }
        assert document["format"] == "zcu-workflow-iteration"
        assert document["format_version"] == 1
        assert document["identity"] == rig.events()[0]["identity"]
        assert document["workflow"] == "example"
        assert document["plan"] == {"count": 1, "values": [10, 20]}
        assert document["tunables"] == {"value": 1}
    assert committed["started"] == {
        "call_seq": 1,
        "commit_seq": 0,
        "revision": 0,
        "kind": "iteration_started",
    }
    assert committed["result"] == {
        "kind": "step_committed",
        "call_seq": 1,
        "commit_seq": 1,
        "revision": 0,
        "encoded_record": {"mode": "json", "value": 1, "error": None},
        "iteration_dir": "iter/000001",
    }
    assert finished["started"] == {
        "call_seq": 2,
        "commit_seq": 1,
        "revision": 0,
        "kind": "iteration_started",
    }
    assert finished["result"] == {
        "kind": "step_finished",
        "call_seq": 2,
        "revision": 0,
        "iteration_dir": "iter/000002",
        "outcome": "done",
        "reason": None,
    }
    assert rig.engine.records("run")[0].encoded_record.value == 1
    assert not tuple(rig.paths.data_root.rglob("*.tmp.json"))


def test_iteration_record_pause_resume_retains_captured_tunables(
    record_rig: Rig,
) -> None:
    rig = record_rig

    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        if state.index == plan.count:
            return Done()
        yield from env.wait_until(START + timedelta(seconds=1))
        state.index += 1
        return Next(tun.value, state)

    def change_and_pause(_signal: StopSignal) -> None:
        rig.engine.update_tunables(
            "run",
            Tunables.model_construct(value=8),
            expected_revision=0,
            actor=Actor("agent", "test"),
        )
        rig.engine.pause("run")

    rig.clock.on_wait = change_and_pause
    rig.start(step, plan=Plan(count=1))
    assert rig.engine.execute("run").lifecycle == "paused"
    discarded = iteration_document(rig, 1)
    assert discarded["tunables"] == {"value": 1}
    assert discarded["started"] == {
        "call_seq": 1,
        "commit_seq": 0,
        "revision": 0,
        "kind": "iteration_started",
    }
    assert discarded["result"] == {
        "kind": "step_discarded",
        "call_seq": 1,
        "revision": 0,
        "iteration_dir": "iter/000001",
        "reason": "pause",
    }
    assert rig.engine.records("run") == ()
    rig.clock.on_wait = None
    rig.engine.resume("run")
    assert rig.engine.execute("run").lifecycle == "done"
    resumed = iteration_document(rig, 2)
    assert resumed["tunables"] == {"value": 8}
    assert resumed["started"] == {
        "call_seq": 2,
        "commit_seq": 0,
        "revision": 1,
        "kind": "iteration_started",
    }
    assert rig.engine.records("run")[0].encoded_record.value == 8
    assert iteration_document(rig, 1) == discarded


@pytest.mark.parametrize("outcome", ["failed", "aborted", "stop", "interrupt"])
def test_iteration_record_preserves_failure_abort_and_stop_classification(
    record_rig: Rig,
    outcome: str,
) -> None:
    rig = record_rig

    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        if outcome == "failed":
            raise ValueError("original workflow cause")
        if outcome == "aborted":
            return Aborted("quality rejected")
        if outcome == "interrupt":
            raise KeyboardInterrupt()
        rig.engine.stop("run")
        yield from env.wait_until(START)
        return Done()

    rig.start(step)
    final = rig.engine.execute("run")
    assert final.lifecycle == (
        outcome if outcome in ("failed", "aborted") else "stopped"
    )
    document = iteration_document(rig, 1)
    result = document["result"]
    assert isinstance(result, dict)
    assert result == {
        key: value
        for key, value in rig.events()[-2].items()
        if key not in ("format_version", "seq", "run_id", "time")
    }
    if outcome == "failed":
        assert result["kind"] == "step_failed"
        assert result["error"]["message"] == "original workflow cause"
    elif outcome == "aborted":
        assert result["kind"] == "step_finished"
        assert result["reason"] == "quality rejected"
    else:
        assert result["kind"] == "step_discarded"
        assert result["reason"] == "stop"
    assert rig.engine.records("run") == ()


class JournalObservingClock(EngineClock):
    """Observe a real journal append through the injected clock port."""

    before_now: Callable[[], None] | None = None

    def now(self) -> datetime:
        observer = self.before_now
        self.before_now = None
        if observer is not None:
            observer()
        return super().now()


def test_iteration_record_is_absent_until_result_is_written_to_journal(
    record_rig: Rig,
) -> None:
    rig = record_rig
    clock = JournalObservingClock()
    rig.engine = Engine(
        EnginePorts(rig.plots, rig.progress, clock, context=7, devices=rig.devices)
    )
    observed: list[str] = []

    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        path = env.iteration_dir / "record.json"
        assert not path.exists()
        if state.index == 1:
            iteration_document(rig, 1)
            return Done()

        def before_result_append() -> None:
            assert rig.kinds()[-1] == "iteration_started"
            assert not path.exists()
            observed.append("no record before journal result")

        state.index += 1
        yield from env.wait_until(START)
        clock.before_now = before_result_append
        return Next(tun.value, state)

    rig.start(step)
    assert rig.engine.execute("run").lifecycle == "done"
    assert observed == ["no record before journal result"]
    iteration_document(rig, 2)


def test_iteration_record_replace_failure_logs_and_keeps_commits_and_next_state(
    record_rig: Rig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig = record_rig
    reached: list[int] = []

    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        reached.append(state.index)
        if state.index == 0:
            # A real filesystem conflict fails replace, not journal or run files.
            (env.iteration_dir / "record.json").mkdir()
        elif state.index == 1:
            assert rig.kinds()[-2:] == ("step_committed", "iteration_started")
            assert len(rig.engine.records("run")) == 1
        else:
            return Done()
        state.index += 1
        yield from env.wait_until(START)
        return Next(tun.value, state)

    rig.start(step)
    final = rig.engine.execute("run")
    assert (final.lifecycle, final.committed_seq) == ("done", 2)
    assert reached == [0, 1, 2]
    # Later invocations still publish their own summaries and state is committed.
    iteration_document(rig, 2)
    iteration_document(rig, 3)
    assert [record.commit_seq for record in rig.engine.records("run")] == [1, 2]
    assert "step_failed" not in rig.kinds()
    errors = [
        (record.getMessage(), record.exc_info[1])
        for record in caplog.records
        if record.exc_info is not None
    ]
    assert any(
        "record" in message.lower() and isinstance(error, OSError)
        for message, error in errors
    )
    assert (rig.paths.data_root / "iter/000001/record.json").is_dir()


def test_iteration_record_publication_replaces_instead_of_editing_destination(
    record_rig: Rig,
) -> None:
    rig = record_rig
    external = rig.root / "external.json"
    external.write_text("unrelated file", encoding="utf-8")

    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        yield from env.wait_until(START)
        # A real destination symlink distinguishes replace from in-place writes.
        (env.iteration_dir / "record.json").symlink_to(external)
        return Done()

    rig.start(step)
    assert rig.engine.execute("run").lifecycle == "done"
    path = rig.paths.data_root / "iter/000001/record.json"
    assert not path.is_symlink()
    assert iteration_document(rig, 1)["format"] == "zcu-workflow-iteration"
    assert external.read_text(encoding="utf-8") == "unrelated file"
