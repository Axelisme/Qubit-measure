"""Observable run-artifact behavior through the package's typed I/O seam."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock

import pytest
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows.artifacts import RunArtifacts
from zcu_tools.experiment.workflows.encoding import encode_record
from zcu_tools.experiment.workflows.journal import (
    ChangedValue,
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
    Completed,
    RunIdentity,
    RunPaths,
)
from zcu_tools.experiment.workflows.ports import DeviceSnapshot


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
            "id-not-the-directory-slug", DeviceSnapshot(()), "test-host"
        ),
        workflow="storage-contract",
        plan={"targets": [1, 2]},
        tunables={"reps": 2},
        requires=(),
        roots=RunPaths(tmp_path / "metadata-slug", tmp_path / "data-slug"),
        started_at=clock.now(),
    )


@pytest.fixture
def artifacts(metadata: RunMetadata, clock: ManualClock) -> RunArtifacts:
    return RunArtifacts(metadata, clock, RLock())


@dataclass
class Cfg:
    reps: int = 2


def save_text(completed: Completed[Cfg, str], path: Path) -> None:
    path.write_text(f"{completed.cfg.reps}:{completed.result}", encoding="utf-8")


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

    def closed_saver(value: Completed[Cfg, str], path: Path) -> None:
        with path.open("x", encoding="utf-8", newline="\n") as writer:
            writer.write(value.result)
        assert writer.closed

    iteration.save(1, "run_t1", Completed(Cfg(), "完整結果"), closed_saver)
    artifacts.append(
        StepFinished(1, 0, iteration.iteration_dir, iteration.run_files, "done", None)
    )
    artifacts.set_lifecycle("done")
    final = tmp_path / "data 資料" / "iter/000001/runs/01-run_t1.h5"
    assert final.read_text(encoding="utf-8") == "完整結果"
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
    iteration.save(2, "run_t1", Completed(Cfg(), "signal"), save_text)
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
            iteration.run_files,
        ),
        StepFinished(1, 0, iteration.iteration_dir, iteration.run_files, "done", None),
        StepDiscarded(1, 0, iteration.iteration_dir, iteration.run_files, "pause"),
        StepFailed(1, 0, iteration.iteration_dir, iteration.run_files, error),
        TunablesChanged(Actor("agent", "協作者"), 0, 1, (ChangedValue("reps", 2, 3),)),
        PauseRequested(clock.now(), 1),
        Paused(1, 1),
        Resumed(1, 1, DeviceSnapshot(())),
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
    assert committed["run_files"] == ["runs/02-run_t1.h5"]
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
    assert (
        iteration.files_dir
        == artifacts.paths.data_root / iteration.iteration_dir / "files"
    )
    assert iteration.files_dir.is_dir()
    assert (iteration.files_dir.parent / "runs").is_dir()
    with pytest.raises(FileExistsError):
        artifacts.new_iteration(call_seq)


def test_completed_publication_uses_exact_temporary_and_retains_final(
    artifacts: RunArtifacts,
) -> None:
    iteration = artifacts.new_iteration(1)
    completed = Completed(Cfg(5), "full result")
    seen: list[Path] = []
    final = iteration.files_dir.parent / "runs" / "03-run_t1.h5"

    def saver(value: Completed[Cfg, str], path: Path) -> None:
        assert value is completed
        assert not final.exists()
        assert path.parent == final.parent
        assert path.name.startswith(".03-run_t1-")
        assert path.name.endswith(".tmp.h5")
        seen.append(path)
        save_text(value, path)

    iteration.save(3, "run_t1", completed, saver)
    assert final.read_text(encoding="utf-8") == "5:full result"
    assert iteration.run_files == ("runs/03-run_t1.h5",)
    assert len(seen) == 1
    assert not seen[0].exists()
    assert not tuple(iteration.files_dir.iterdir())


def test_existing_final_is_never_overwritten(artifacts: RunArtifacts) -> None:
    iteration = artifacts.new_iteration(1)
    final = iteration.files_dir.parent / "runs" / "01-run_t1.h5"
    final.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        iteration.save(1, "run_t1", Completed(Cfg(), "replacement"), save_text)
    assert final.read_text(encoding="utf-8") == "original"
    assert iteration.run_files == ()
    assert not tuple(final.parent.glob("*.tmp.h5"))


def test_final_created_during_saver_is_not_overwritten(artifacts: RunArtifacts) -> None:
    iteration = artifacts.new_iteration(1)
    final = iteration.files_dir.parent / "runs" / "01-run_t1.h5"

    def racing_saver(value: Completed[Cfg, str], path: Path) -> None:
        save_text(value, path)
        final.write_text("external final", encoding="utf-8")

    with pytest.raises(FileExistsError):
        iteration.save(1, "run_t1", Completed(Cfg(), "replacement"), racing_saver)
    assert final.read_text(encoding="utf-8") == "external final"
    assert iteration.run_files == ()
    assert len(tuple(final.parent.glob(".01-run_t1-*.tmp.h5"))) == 1


def test_saver_failure_retains_partial_temporary_and_original_cause(
    artifacts: RunArtifacts,
) -> None:
    iteration = artifacts.new_iteration(1)
    cause = OSError("writer failed after partial data")
    seen: list[Path] = []

    def broken_saver(value: Completed[Cfg, str], path: Path) -> None:
        path.write_text(value.result[:3], encoding="utf-8")
        seen.append(path)
        raise cause

    with pytest.raises(OSError, match="writer failed") as caught:
        iteration.save(1, "run_t1", Completed(Cfg(), "partial"), broken_saver)
    assert caught.value is cause
    assert seen[0].read_text(encoding="utf-8") == "par"
    assert not seen[0].with_name("01-run_t1.h5").exists()
    assert iteration.run_files == ()


def test_link_failure_does_not_retry_or_publish(
    artifacts: RunArtifacts, monkeypatch: pytest.MonkeyPatch
) -> None:
    iteration = artifacts.new_iteration(1)
    cause = OSError("filesystem does not support hard links")
    attempts: list[tuple[Path, Path]] = []

    def broken_link(source: Path, destination: Path) -> None:
        attempts.append((source, destination))
        raise cause

    monkeypatch.setattr(os, "link", broken_link)
    with pytest.raises(OSError, match="does not support") as caught:
        iteration.save(1, "run_t1", Completed(Cfg(), "full"), save_text)
    assert caught.value is cause
    assert len(attempts) == 1
    source, final = attempts[0]
    assert source.read_text(encoding="utf-8") == "2:full"
    assert not final.exists()
    assert iteration.run_files == ()


def test_unlink_failure_retains_published_final_in_failure_journal(
    artifacts: RunArtifacts, monkeypatch: pytest.MonkeyPatch
) -> None:
    iteration = artifacts.new_iteration(1)
    cause = OSError("temporary cleanup denied")
    originals = Path.unlink
    seen: list[Path] = []

    def broken_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path.name.endswith(".tmp.h5"):
            seen.append(path)
            raise cause
        originals(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", broken_unlink)
    with pytest.raises(OSError, match="cleanup denied") as caught:
        iteration.save(1, "run_t1", Completed(Cfg(), "full"), save_text)
    assert caught.value is cause
    final = iteration.files_dir.parent / "runs" / "01-run_t1.h5"
    assert final.read_text(encoding="utf-8") == "2:full"
    assert iteration.run_files == ("runs/01-run_t1.h5",)
    assert os.path.samefile(seen[0], final)
    artifacts.append(
        StepFailed(
            1, 0, iteration.iteration_dir, iteration.run_files, describe_error(cause)
        )
    )
    failed = json.loads(
        artifacts.journal_path.read_text(encoding="utf-8").splitlines()[-1]
    )
    assert failed["kind"] == "step_failed"
    assert failed["run_files"] == ["runs/01-run_t1.h5"]
    assert failed["error"]["message"] == "temporary cleanup denied"


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
    change = TunablesChanged(
        Actor("user", "user"), 0, 1, (ChangedValue("reps", 2, float("inf")),)
    )
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
    "reference", ["../outside", "/outside", "iter/../outside", "iter\\outside"]
)
def test_journal_rejects_external_path_references(
    artifacts: RunArtifacts, reference: str
) -> None:
    before = artifacts.journal_path.read_bytes()
    with pytest.raises(ValueError, match="relative paths"):
        artifacts.append(IterationStarted(1, 0, 0, reference))
    assert artifacts.journal_path.read_bytes() == before
