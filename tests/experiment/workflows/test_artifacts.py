"""Typed artifact I/O and async commit publication through public seams."""

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock

import pytest
from pydantic import TypeAdapter
from zcu_tools.experiment.workflows import Actor, Live, RunPaths, WorkflowEnv
from zcu_tools.experiment.workflows.artifacts import RunArtifacts
from zcu_tools.experiment.workflows.journal import (
    Paused,
    PauseRequested,
    Resumed,
    RunEnded,
    RunMetadata,
    StopRequested,
    TunablesChanged,
)

from tests._filesystem import DIRECTORY_OPEN_ERROR, symlink_or_skip

from ._engine_fakes import START, ManualClock, Plan, Point, Rig, Tunables, identity


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def metadata(tmp_path: Path) -> RunMetadata:
    return RunMetadata(
        identity("id-not-the-directory-slug"),
        "storage-contract",
        {"targets": [1, 2]},
        {"reps": 2},
        RunPaths(tmp_path / "metadata-slug", tmp_path / "data-slug"),
        START,
    )


@pytest.fixture
def artifacts(metadata: RunMetadata, clock: ManualClock) -> RunArtifacts:
    return RunArtifacts(metadata, clock, RLock())


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    rig = Rig(tmp_path)
    yield rig
    rig.close()


def test_start_documents_are_detached_and_use_identity_not_slug(
    artifacts: RunArtifacts, metadata: RunMetadata
) -> None:
    if isinstance(metadata.plan, dict):
        metadata.plan["targets"] = [9]
    path = artifacts.paths.metadata_root / "manifest.json"
    initial = json.loads(path.read_text())
    assert initial["lifecycle"] == "running"
    assert initial["ended_at"] is None
    assert initial["journal"] == str(artifacts.journal_path)
    assert initial["initial_tunables"] == {"reps": 2}
    assert initial["plan"] == {"targets": [1, 2]}
    event = json.loads(artifacts.journal_path.read_text())
    assert event["kind"] == "run_started" and event["seq"] == 1
    assert event["run_id"] == metadata.identity.run_id
    assert event["identity"] == initial["identity"]
    assert event["roots"] == initial["roots"]
    assert datetime.fromisoformat(event["time"]) == metadata.started_at
    artifacts.set_lifecycle("paused")
    assert json.loads(path.read_text())["plan"] == {"targets": [1, 2]}


def test_start_preserves_native_provenance_in_manifest_and_journal(
    artifacts: RunArtifacts, metadata: RunMetadata
) -> None:
    manifest = json.loads((artifacts.paths.metadata_root / "manifest.json").read_text())
    event = json.loads(artifacts.journal_path.read_text())
    expected = TypeAdapter(type(metadata.identity.provenance)).dump_python(
        metadata.identity.provenance, mode="json"
    )
    assert manifest["identity"]["provenance"] == expected
    assert event["identity"]["provenance"] == expected


def test_relative_unicode_roots_remain_absolute_after_cwd_change(
    tmp_path: Path,
    metadata: RunMetadata,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    artifacts = RunArtifacts(
        replace(metadata, roots=RunPaths(Path("metadata 資料"), Path("data 資料"))),
        clock,
        RLock(),
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    directory = artifacts.new_commit_dir()
    artifacts.set_lifecycle("done")
    assert directory == tmp_path / "data 資料/commits/000001"
    assert directory.is_dir()
    assert artifacts.paths.metadata_root == tmp_path / "metadata 資料"
    assert not tuple(elsewhere.iterdir())


def test_invalid_start_time_creates_no_roots(
    metadata: RunMetadata, clock: ManualClock
) -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        RunArtifacts(
            metadata.model_copy(update={'started_at': datetime(2026, 10, 6)}), clock, RLock()
        )
    assert not metadata.roots.metadata_root.exists()
    assert not metadata.roots.data_root.exists()


def test_mandatory_startup_nonfinite_does_not_use_record_fallback(
    metadata: RunMetadata, clock: ManualClock
) -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        RunArtifacts(replace(metadata, tunables={"reps": float("nan")}), clock, RLock())
    assert not metadata.roots.metadata_root.exists()
    assert not metadata.roots.data_root.exists()


def test_journal_round_trips_control_events_in_append_order(
    artifacts: RunArtifacts, clock: ManualClock
) -> None:
    events = (
        TunablesChanged(Actor("agent", "協作者"), 0, 1, {"reps": 3}),
        PauseRequested(clock.now(), 1),
        Paused(1, 0),
        Resumed(0, 1),
        StopRequested(clock.now(), 1),
        RunEnded("cancelled", None, 1, 0),
    )
    path = artifacts.paths.metadata_root / "manifest.json"
    before = path.read_bytes()
    for event in events:
        artifacts.append(event)
    assert path.read_bytes() == before
    raw = artifacts.journal_path.read_text()
    assert raw.endswith("\n") and "協作者" in raw
    lines = [json.loads(line) for line in raw.splitlines()]
    assert [line["seq"] for line in lines] == list(range(1, len(events) + 2))
    assert [line["kind"] for line in lines[1:]] == [event.kind for event in events]
    assert lines[1]["tunables"] == {"reps": 3}


def test_lifecycle_manifest_replaces_only_on_change(
    artifacts: RunArtifacts, clock: ManualClock, metadata: RunMetadata
) -> None:
    path = artifacts.paths.metadata_root / "manifest.json"
    clock.value += timedelta(seconds=5)
    artifacts.set_lifecycle("paused")
    paused = json.loads(path.read_text())
    assert paused["lifecycle"] == "paused" and paused["ended_at"] is None
    assert datetime.fromisoformat(paused["updated_at"]) == clock.now()
    clock.value += timedelta(seconds=5)
    artifacts.set_lifecycle("failed", "disk failure")
    ended = json.loads(path.read_text())
    assert ended["reason"] == "disk failure"
    assert datetime.fromisoformat(ended["started_at"]) == metadata.started_at
    assert datetime.fromisoformat(ended["ended_at"]) == clock.now()
    with pytest.raises(ValueError, match="lifecycle change"):
        artifacts.set_lifecycle("failed", "replacement reason")
    assert json.loads(path.read_text())["reason"] == "disk failure"


def test_manifest_replace_failure_retains_last_document(
    artifacts: RunArtifacts, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = artifacts.paths.metadata_root / "manifest.json"
    before = path.read_bytes()
    cause = OSError("manifest replace failed")

    def broken_replace(source: Path, destination: Path) -> Path:
        raise cause

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", broken_replace)
        with pytest.raises(OSError, match="replace failed") as caught:
            artifacts.set_lifecycle("paused")
        assert caught.value is cause
    assert path.read_bytes() == before
    assert len(tuple(path.parent.glob(".manifest-*.tmp.json"))) == 1
    artifacts.set_lifecycle("failed", str(cause))
    assert json.loads(path.read_text())["lifecycle"] == "failed"


def test_journal_io_failure_is_not_followed_by_invented_success(
    artifacts: RunArtifacts,
) -> None:
    artifacts.journal_path.unlink()
    artifacts.journal_path.mkdir()
    with pytest.raises(DIRECTORY_OPEN_ERROR):
        artifacts.append(Paused(None, 0))
    with pytest.raises(RuntimeError, match="unusable after an I/O failure"):
        artifacts.append(RunEnded("failed", "journal broken", None, 0))
    assert not tuple(artifacts.journal_path.iterdir())


def test_nonfinite_mandatory_payload_does_not_consume_sequence(
    artifacts: RunArtifacts,
) -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        artifacts.append(
            TunablesChanged(Actor("user", "user"), 0, 1, {"reps": float("inf")})
        )
    artifacts.append(Paused(None, 0))
    lines = [
        json.loads(line) for line in artifacts.journal_path.read_text().splitlines()
    ]
    assert [line["seq"] for line in lines] == [1, 2]


@pytest.mark.parametrize("root_name", ["metadata_root", "data_root"])
def test_start_refuses_existing_root_without_creating_other(
    metadata: RunMetadata, clock: ManualClock, root_name: str
) -> None:
    root = getattr(metadata.roots, root_name)
    root.mkdir()
    (root / "keep.txt").write_text("existing")
    with pytest.raises(FileExistsError, match="Run root already exists"):
        RunArtifacts(metadata, clock, RLock())
    other = (
        metadata.roots.data_root
        if root_name == "metadata_root"
        else metadata.roots.metadata_root
    )
    assert not other.exists()
    assert (root / "keep.txt").read_text() == "existing"


def test_second_root_creation_failure_keeps_first_root(
    metadata: RunMetadata, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path.mkdir

    def broken_mkdir(
        path: Path, mode: int = 0o777, *, parents: bool = False, exist_ok: bool = False
    ) -> None:
        if path == metadata.roots.data_root:
            raise OSError("second disk refused creation")
        original(path, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", broken_mkdir)
    with pytest.raises(OSError, match="second disk"):
        RunArtifacts(metadata, clock, RLock())
    assert metadata.roots.metadata_root.is_dir()
    assert not (metadata.roots.metadata_root / "manifest.json").exists()


def test_commit_document_matches_journal_and_retains_read_revisions(rig: Rig) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        first = tun.current
        rig.engine.update_tunables(
            "run", Tunables(value=8), expected_revision=0, actor=Actor("agent", "test")
        )
        latest = tun.current
        await env.commit(Point(value=first.value + latest.value))

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    path = rig.paths.data_root / "commits/000001/record.json"
    document = json.loads(path.read_text())
    committed = next(
        event for event in rig.events() if event["kind"] == "record_committed"
    )
    assert document["record"] == committed["encoded_record"]
    assert document["commit_seq"] == committed["commit_seq"] == 1
    assert document["directory"] == committed["directory"] == "commits/000001"
    assert document["committed_at"] == committed["committed_at"]
    assert document["started"]["revisions"] == [0, 1]
    assert document["record_type"] == committed["record_type"]


def test_record_output_failure_keeps_commits_and_author_continues(
    rig: Rig, caplog: pytest.LogCaptureFixture
) -> None:
    reached: list[int] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        for index in range(plan.count):
            directory = env.new_commit_dir()
            if index == 0:
                (directory / "record.json").mkdir()
            reached.append(index)
            await env.commit(Point(value=index), directory)

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert final.committed_seq == 2 and reached == [0, 1]
    assert (rig.paths.data_root / "commits/000001/record.json").is_dir()
    assert (rig.paths.data_root / "commits/000002/record.json").is_file()
    assert [r.commit_seq for r in rig.engine.records("run")] == [1, 2]
    assert any(
        "record" in record.getMessage().lower() and record.exc_info is not None
        for record in caplog.records
    )


def test_commit_publication_replaces_instead_of_editing_destination(rig: Rig) -> None:
    external = rig.root / "external.json"
    external.write_text("unrelated file")

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        directory = env.new_commit_dir()
        symlink_or_skip(directory / "record.json", external)
        await env.commit(Point(value=1), directory)

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    path = rig.paths.data_root / "commits/000001/record.json"
    assert not path.is_symlink()
    assert json.loads(path.read_text())["record"]["value"] == {"value": 1}
    assert external.read_text() == "unrelated file"
