"""Commit allocation, publication and typed/raw saved-record contracts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment import workflows as w
from zcu_tools.experiment.workflows import saved_run
from zcu_tools.plotting.plots import NonPresentingHost

from ._engine_ports import ManualClock, RecordingBar, RecordingDevices


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    values: tuple[int, ...] = (7, 8)


class Knobs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Record(BaseModel):
    index: int


class OtherRecord(BaseModel):
    index: int


type Rig = tuple[w.Engine[int], w.RunPaths]


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    plots = w.WorkflowPlots(NonPresentingHost())
    engine = w.Engine(
        w.EnginePorts(
            plots, RecordingBar, ManualClock(), context=7, devices=RecordingDevices()
        )
    )
    yield engine, w.RunPaths(tmp_path / "metadata", tmp_path / "data")
    plots.finish()
    plots.release()


def execute(rig: Rig, script: w.WorkflowScript[Plan, Knobs, None, int]) -> None:
    engine, paths = rig
    engine.start(
        script,
        plan=Plan(),
        tunables=Knobs(),
        paths=paths,
        identity=w.RunIdentity(
            "run",
            SoftwareProvenance(
                software_versions={},
                git_commit=None,
                git_dirty=None,
                qick_version=None,
                soc_fingerprint=None,
                hostname="offline",
            ),
        ),
    )
    status = engine.execute("run")
    assert status.lifecycle == "done", status.reason


def test_allocation_order_is_independent_of_commit_order_and_duplicate_is_rejected(
    rig: Rig,
) -> None:
    _engine, paths = rig
    allocated: list[Path] = []

    @w.workflow("allocation", "Allocate raw-data paths independently of commits")
    async def script(env: w.WorkflowEnv[int], plan: Plan, tun: w.Live[Knobs]) -> None:
        first, second = env.new_commit_dir(), env.new_commit_dir()
        allocated.extend((first, second))
        (first / "raw.bin").write_bytes(b"first")
        await env.commit(Record(index=2), second)
        await env.commit(Record(index=1), first)
        with pytest.raises(ValueError, match="already committed"):
            await env.commit(Record(index=99), first)
        unused = env.new_commit_dir()
        (unused / "notes.txt").write_text("uncommitted", encoding="utf-8")
        await env.commit(Record(index=4))

    execute(rig, script)
    records = saved_run.read_records(paths.metadata_root, Record)
    assert [
        (record.commit_seq, record.directory.name, record.record.index)
        for record in records
    ] == [(1, "000002", 2), (2, "000001", 1), (3, "000004", 4)]
    assert allocated == [
        paths.data_root / "commits/000001",
        paths.data_root / "commits/000002",
    ]
    assert (allocated[0] / "raw.bin").read_bytes() == b"first"
    assert (paths.data_root / "commits/000003/notes.txt").read_text() == "uncommitted"
    assert [record.directory.name for record in records] == [
        "000002",
        "000001",
        "000004",
    ]


def test_commit_dirs_is_lazy_and_zip_allocates_only_the_authored_values(
    rig: Rig,
) -> None:
    _engine, paths = rig
    counts: list[int] = []

    @w.workflow("zip", "Allocate exactly once for each authored loop value")
    async def script(env: w.WorkflowEnv[int], plan: Plan, tun: w.Live[Knobs]) -> None:
        directories = env.commit_dirs()
        counts.append(len(tuple((paths.data_root / "commits").glob("*"))))
        for value, directory in zip(plan.values, directories, strict=False):
            await env.commit(Record(index=value), directory)
        counts.append(len(tuple((paths.data_root / "commits").glob("*"))))

    execute(rig, script)
    assert counts == [0, 2]
    assert [
        record.record.index
        for record in saved_run.read_records(paths.metadata_root, TypeAdapter(Record))
    ] == [7, 8]


def test_record_schema_and_envelope_are_persisted_once_per_record_type(
    rig: Rig,
) -> None:
    _engine, paths = rig

    @w.workflow("schema", "Publish model schema with each distinct record type")
    async def script(env: w.WorkflowEnv[int], plan: Plan, tun: w.Live[Knobs]) -> None:
        directory = env.new_commit_dir()
        await env.commit(Record(index=1), directory)
        await env.commit(Record(index=2))

    execute(rig, script)
    manifest = saved_run.read_manifest(paths.metadata_root, "schema")
    name = f"{Record.__module__}.{Record.__qualname__}"
    assert manifest.record_schemas == {name: Record.model_json_schema()}
    records = saved_run.read_records(paths.metadata_root, Record)
    assert [record.record_type for record in records] == [name, name]
    assert [record.commit_seq for record in records] == [1, 2]
    assert all(record.committed_at.tzinfo is not None for record in records)
    first = json.loads(records[0].record_path.read_text())
    assert first["directory"] == "commits/000001"
    assert first["started"]["call_seq"] == 1
    assert first["started"]["commit_seq"] == 0


def test_typed_reader_rejects_wrong_type_and_bad_fields_but_raw_preserves_json(
    rig: Rig,
) -> None:
    _engine, paths = rig

    @w.workflow("reader", "Read only declared committed record envelopes")
    async def script(env: w.WorkflowEnv[int], plan: Plan, tun: w.Live[Knobs]) -> None:
        await env.commit(Record(index=1), env.new_commit_dir())

    execute(rig, script)
    source = paths.data_root / "commits/000001/record.json"
    with pytest.raises(ValueError, match="record.json"):
        saved_run.read_records(paths.metadata_root, OtherRecord)
    payload = json.loads(source.read_text())
    payload["record"]["value"]["index"] = "not an integer"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="record.json"):
        saved_run.read_records(paths.metadata_root, Record)
    assert saved_run.read_records_raw(paths.metadata_root)[0].record.value == {
        "index": "not an integer"
    }


def test_foreign_directories_do_not_publish_records(rig: Rig, tmp_path: Path) -> None:
    engine, paths = rig
    foreign = tmp_path / "foreign"
    foreign.mkdir()

    @w.workflow("foreign", "Reject directories not allocated by this run")
    async def script(env: w.WorkflowEnv[int], plan: Plan, tun: w.Live[Knobs]) -> None:
        with pytest.raises(ValueError, match="allocated"):
            await env.commit(Record(index=99), foreign)
        await env.commit(Record(index=1), env.new_commit_dir())

    execute(rig, script)
    assert len(engine.records("run")) == 1
    assert [
        record.record.index
        for record in saved_run.read_records(paths.metadata_root, Record)
    ] == [1]
