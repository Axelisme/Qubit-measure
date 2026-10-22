"""Saved readers validate committed models without interpreting leftover files."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Self

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    ModelWrapValidatorHandler,
    TypeAdapter,
    model_validator,
)
from zcu_tools.experiment.workflows import Live, WorkflowEnv
from zcu_tools.experiment.workflows.saved_run import (
    native_path,
    read_manifest,
    read_records,
    read_records_raw,
)

from ._engine_fakes import Plan, Point, Rig, Tunables


class OtherPoint(BaseModel):
    other: str


class AfterPoint(BaseModel):
    model_config = ConfigDict(title="NotTheQualifiedType")
    value: int

    @model_validator(mode="after")
    def positive(self) -> Self:
        if self.value < 0:
            raise ValueError("point must be positive")
        return self


class WrappedPoint(BaseModel):
    value: int

    @model_validator(mode="wrap")
    @classmethod
    def positive(cls, value: object, handler: ModelWrapValidatorHandler[Self]) -> Self:
        point = handler(value)
        if point.value < 0:
            raise ValueError("point must be positive")
        return point


@pytest.mark.parametrize("model", [AfterPoint, WrappedPoint], ids=["after", "wrap"])
@pytest.mark.parametrize("use_adapter", [False, True], ids=["model", "adapter"])
def test_typed_reader_runs_model_validators_and_preserves_source_and_identity(
    tmp_path: Path, model: type[BaseModel], use_adapter: bool
) -> None:
    rig = Rig(tmp_path)
    point = model.model_validate({"value": 2})

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        env.commit(point)

    try:
        rig.start(script)
        assert rig.engine.execute("run").lifecycle == "done"
        target = TypeAdapter(model) if use_adapter else model
        records = read_records(rig.paths.metadata_root, target)
        assert [row.record for row in records] == [point]
        assert records[0].record_type == f"{model.__module__}.{model.__qualname__}"
        source = records[0].record_path
        document = json.loads(source.read_text())
        document["record"]["value"] = {"value": -1}
        source.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="point must be positive") as caught:
            read_records(rig.paths.metadata_root, target)
        assert str(source) in str(caught.value)
        assert read_records_raw(rig.paths.metadata_root)[0].record.value == {
            "value": -1
        }
    finally:
        rig.close()


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    rig = Rig(tmp_path)

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        with env.child("point") as uncommitted:
            (uncommitted.dir / "native.h5").write_text("leftover is not a record")
        with env.child("point") as later, env.child("point") as earlier:
            earlier.commit(Point(value=2))
            later.commit(Point(value=1))

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    yield rig
    rig.close()


@pytest.mark.parametrize(
    "adapter", [Point, TypeAdapter(Point)], ids=["model", "adapter"]
)
def test_typed_reader_orders_commits_and_ignores_uncommitted_native_files(
    rig: Rig, adapter: type[Point] | TypeAdapter[Point]
) -> None:
    records = read_records(rig.paths.metadata_root, adapter)
    assert tuple(row.record for row in records) == (Point(value=2), Point(value=1))
    assert [row.directory.name for row in records] == ["point-000003", "point-000002"]
    assert [row.commit_seq for row in records] == [1, 2]
    assert all(row.record_path == row.directory / "record.json" for row in records)


def test_missing_committed_file_does_not_invent_data_from_native_files(
    rig: Rig,
) -> None:
    missing = rig.paths.data_root / "point-000003/record.json"
    missing.unlink()
    records = read_records(rig.paths.metadata_root, Point)
    assert tuple(row.record for row in records) == (Point(value=1),)
    assert tuple(row.directory.name for row in records) == ("point-000002",)


def test_reader_type_mismatch_names_source_and_raw_reader_retains_json(
    rig: Rig,
) -> None:
    source = rig.paths.data_root / "point-000003/record.json"
    with pytest.raises(ValueError, match="Expected record type") as caught:
        read_records(rig.paths.metadata_root, OtherPoint)
    assert str(source) in str(caught.value)
    assert [row.record.value for row in read_records_raw(rig.paths.metadata_root)] == [
        {"value": 2},
        {"value": 1},
    ]


def test_reader_validation_failure_names_source_without_validating_raw_json(
    rig: Rig,
) -> None:
    source = rig.paths.data_root / "point-000003/record.json"
    document = json.loads(source.read_text())
    document["record"]["value"] = {"value": "invalid integer"}
    source.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="value") as caught:
        read_records(rig.paths.metadata_root, Point)
    assert str(source) in str(caught.value)
    assert read_records_raw(rig.paths.metadata_root)[0].record.value == {
        "value": "invalid integer"
    }


def test_manifest_schema_is_information_not_validation(rig: Rig) -> None:
    path = rig.paths.metadata_root / "manifest.json"
    document = json.loads(path.read_text())
    for name in document["record_schemas"]:
        document["record_schemas"][name] = {"not": {}}
    path.write_text(json.dumps(document))
    assert tuple(
        row.record for row in read_records(rig.paths.metadata_root, Point)
    ) == (Point(value=2), Point(value=1))


def test_malformed_record_and_wrong_workflow_name_source(rig: Rig) -> None:
    with pytest.raises(ValueError, match="Expected workflow") as caught:
        read_manifest(rig.paths.metadata_root, "other")
    assert "manifest.json" in str(caught.value)
    source = rig.paths.data_root / "point-000003/record.json"
    source.write_text("not json")
    with pytest.raises(ValueError, match="JSON") as caught:
        read_records_raw(rig.paths.metadata_root)
    assert str(source) in str(caught.value)


@pytest.mark.parametrize(
    "filename", ["../outside", "/outside", "C:/outside", "C:outside", "nested\\outside"]
)
def test_native_reference_cannot_escape_commit_directory(
    rig: Rig, filename: str
) -> None:
    with pytest.raises(ValueError, match="Invalid workflow-relative filename"):
        native_path(rig.paths.data_root / "point-000003", filename)
