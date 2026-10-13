"""Caller-owned values and opaque metadata through the standalone parameter seam."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from ruamel.yaml import YAML
from zcu_tools.format_version import FormatError, VersionError, YamlMap
from zcu_tools.resources.document_store import ConflictError, DocumentChange

if TYPE_CHECKING:
    from zcu_tools.resources.parameter_fields import FieldView
    from zcu_tools.resources.parameters import ParameterView


class Qubit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    t1: float = Field(gt=0)
    frequency: float


class Values(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qubit: Qubit
    gain: float = 1.0


def open_values[T: BaseModel](path: Path, model: type[T]) -> ParameterView[T]:
    # Resolve inside the test so an absent public capability is an assertion red,
    # not a collection error that hides the missing standalone contract.
    assert find_spec("zcu_tools.resources.parameters") is not None, (
        "Standalone caller-model parameter access must be available"
    )
    opener = getattr(
        import_module("zcu_tools.resources.parameters"), "open_parameters", None
    )
    assert callable(opener), "Standalone parameters need a single public opener"
    return opener(path, model)


def fields(value: object) -> FieldView:
    from zcu_tools.resources.parameter_fields import FieldView

    assert isinstance(value, FieldView)
    return value


@pytest.fixture
def path(tmp_path: Path) -> Path:
    path = tmp_path / "caller.yaml"
    path.write_text(
        "format: zcu.parameter-container\nformat_version: '2.0'\n"
        "components:\n  qubit:\n    t1: 12.5\n    frequency: 4567.0\n"
        "metadata:\n  qubit.t1:\n    source: outside/run/unknown\n"
        "    stderr: -3.5\n    cloned_from: {caller: original}\n",
        encoding="utf-8",
    )
    return path


def test_caller_schema_values_metadata_round_trip_and_live_reads(path: Path) -> None:
    params = open_values(path, Values)
    qubit = fields(params.qubit)
    scalar = qubit.t1
    assert isinstance(scalar, float)
    assert scalar * 2 == 25
    original_meta = params.meta("qubit.t1")
    assert original_meta == {
        "source": "outside/run/unknown",
        "stderr": -3.5,
        "cloned_from": {"caller": "original"},
    }
    with params.edit() as draft:
        fields(draft.qubit).t1 = 20
        draft.set_meta("qubit.frequency", {"comment": ["caller", {"nested": None}]})
        assert draft.snapshot().qubit.t1 == 20
        assert qubit.t1 == 12.5
    assert qubit.t1 == 20
    assert params.meta("qubit.t1") == original_meta
    loaded = open_values(path, Values)
    assert loaded.snapshot().qubit.frequency == 4567
    assert loaded.meta("qubit.frequency") == {"comment": ["caller", {"nested": None}]}
    loaded.snapshot().qubit.t1 = 999
    assert loaded.snapshot().qubit.t1 == 20
    fields(params.qubit).t1 = 30
    assert open_values(path, Values).snapshot().qubit.t1 == 30
    assert params.meta("qubit.t1") == original_meta


@pytest.mark.parametrize("key", ["stderr", "source", "comment", "caller_note"])
def test_opaque_metadata_replacement_clear_and_detached_read(
    path: Path, key: str
) -> None:
    params = open_values(path, Values)
    metadata: YamlMap = {key: {"values": [-9, None, True], "external": "missing/run"}}
    before = params.snapshot()
    with params.edit() as draft:
        draft.set_meta("qubit.t1", metadata)
        assert draft.meta("qubit.t1") == metadata
    assert params.snapshot() == before
    metadata[key] = "mutated caller mapping"
    readback = params.meta("qubit.t1")
    assert readback == {key: {"values": [-9, None, True], "external": "missing/run"}}
    assert readback is not None
    readback[key] = "mutated readback"
    assert open_values(path, Values).meta("qubit.t1") == {
        key: {"values": [-9, None, True], "external": "missing/run"}
    }
    with params.edit() as draft:
        draft.set_meta("qubit.t1", {})
        assert draft.meta("qubit.t1") == {}
    assert params.meta("qubit.t1") == {}
    assert params.snapshot() == before


@pytest.mark.parametrize(
    "metadata", [{"bad": object()}, {1: "non-string key"}, {"bad": (1, 2)}]
)
def test_invalid_metadata_rejects_whole_edit(path: Path, metadata: object) -> None:
    params = open_values(path, Values)
    before, disk = params.snapshot(), path.read_bytes()
    with pytest.raises(ValidationError):
        with params.edit() as draft:
            draft.set("qubit.t1", 40)
            draft.set_meta("qubit.t1", metadata)
    assert params.snapshot() == before
    assert path.read_bytes() == disk


def test_value_schema_and_body_rejections_do_not_publish(path: Path) -> None:
    params = open_values(path, Values)
    before, disk = params.snapshot(), path.read_bytes()
    with pytest.raises(ValidationError):
        with params.edit() as draft:
            draft.set_meta("qubit.t1", {"stderr": "anything"})
            draft.set("qubit.t1", -1)
    with pytest.raises(RuntimeError, match="caller abort"):
        with params.edit() as draft:
            draft.set("qubit.t1", 99)
            raise RuntimeError("caller abort")
    assert params.snapshot() == before
    assert path.read_bytes() == disk
    with params.edit() as draft:
        with pytest.raises(ValidationError):
            draft.set("qubit.t1", -1)
        assert draft.snapshot() == before
        draft.set("qubit.t1", 25)
    assert params.snapshot().qubit.t1 == 25


def test_snapshot_refresh_and_successful_empty_edit_adopt_disk(path: Path) -> None:
    params = open_values(path, Values)
    other = open_values(path, Values)
    with other.edit() as draft:
        draft.set("qubit.t1", 41)
        draft.set_meta("qubit.t1", {"caller": "external"})
    assert params.snapshot().qubit.t1 == 12.5
    assert params.refresh() is True
    assert params.snapshot().qubit.t1 == 41
    assert params.meta("qubit.t1") == {"caller": "external"}
    assert params.refresh() is False
    with other.edit() as draft:
        draft.set("qubit.t1", 42)
    with params.edit():
        pass
    assert params.snapshot().qubit.t1 == 42


def test_conflict_rejects_values_and_metadata_together(path: Path) -> None:
    params = open_values(path, Values)
    other = open_values(path, Values)
    before = params.snapshot()
    with pytest.raises(ConflictError):
        with params.edit() as draft:
            draft.set("qubit.t1", 50)
            draft.set_meta("qubit.t1", {"stderr": 100})
            with other.edit() as concurrent:
                concurrent.set("qubit.t1", 60)
    assert params.snapshot() == before
    disk = open_values(path, Values)
    assert disk.snapshot().qubit.t1 == 60
    assert disk.meta("qubit.t1") == params.meta("qubit.t1")


def test_metadata_uses_existing_mapping_leaf_merge_and_conflict(path: Path) -> None:
    params = open_values(path, Values)
    other = open_values(path, Values)
    with params.edit() as draft:
        draft.set_meta(
            "qubit.t1",
            {
                "source": "outside/run/unknown",
                "stderr": 4,
                "cloned_from": {"caller": "original"},
            },
        )
        with other.edit() as concurrent:
            concurrent.set_meta(
                "qubit.t1",
                {
                    "source": "another/outside/run",
                    "stderr": -3.5,
                    "cloned_from": {"caller": "original"},
                },
            )
    assert params.meta("qubit.t1") == {
        "source": "another/outside/run",
        "stderr": 4,
        "cloned_from": {"caller": "original"},
    }
    with pytest.raises(ConflictError):
        with params.edit() as draft:
            draft.set_meta("qubit.t1", {"stderr": 10})
            with other.edit() as concurrent:
                concurrent.set_meta("qubit.t1", {"stderr": 20})
    assert open_values(path, Values).meta("qubit.t1") == {"stderr": 20}


def test_replace_failure_preserves_snapshot_and_file(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = open_values(path, Values)
    before, disk = params.snapshot(), path.read_bytes()
    original = Path.replace

    def fail_replace(source: Path, target: Path) -> Path:
        if target == path:
            raise OSError("synthetic replace failure")
        return original(source, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        with params.edit() as draft:
            draft.set("qubit.t1", 55)
            draft.set_meta("qubit.t1", {"caller": "new"})
    assert params.snapshot() == before
    assert path.read_bytes() == disk


def test_observer_failure_does_not_undo_published_commit(
    path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    params = open_values(path, Values)
    notifications: list[DocumentChange] = []

    def fail_after_publication(change: DocumentChange) -> None:
        notifications.append(change)
        assert params.snapshot().qubit.t1 == 77
        raise RuntimeError("synthetic observer failure")

    unsubscribe: Callable[[], None] = params.subscribe(fail_after_publication)
    with params.edit() as draft:
        draft.set("qubit.t1", 77)
        draft.set_meta("qubit.t1", {"source": "external/source"})
    assert len(notifications) == 1
    assert params.snapshot().qubit.t1 == 77
    assert open_values(path, Values).meta("qubit.t1") == {"source": "external/source"}
    assert "synthetic observer failure" in caplog.text
    unsubscribe()


@pytest.mark.parametrize("policy", ["forbid", "ignore", "allow"])
@pytest.mark.parametrize("minor", ["2.0", "2.7"])
def test_caller_extra_policy_is_independent_of_container_minor(
    path: Path, policy: Literal["forbid", "ignore", "allow"], minor: str
) -> None:
    class CallerValues(BaseModel):
        model_config = ConfigDict(extra=policy)
        qubit: Qubit

    document = YAML(typ="safe").load(path.read_text())
    document["format_version"] = minor
    document["components"]["extension"] = "caller extra"
    YAML().dump(document, path)
    if policy == "forbid":
        with pytest.raises(ValidationError):
            open_values(path, CallerValues)
        return
    params = open_values(path, CallerValues)
    assert params.snapshot().model_extra == (
        {"extension": "caller extra"} if policy == "allow" else None
    )
    with params.edit() as draft:
        draft.set("qubit.t1", 91)
    persisted = YAML(typ="safe").load(path.read_text())
    assert ("extension" in persisted["components"]) is (policy == "allow")


def test_forward_minor_preserves_container_fields_and_units(path: Path) -> None:
    path.write_text(
        path.read_text().replace("'2.0'", "'2.8'") + "future: {note: preserved}\n"
    )
    params = open_values(path, Values)
    original = path.read_bytes()
    with params.edit():
        pass
    assert path.read_bytes() == original
    with params.edit() as draft:
        draft.set("qubit.frequency", 6789)
        draft.set_meta("qubit.frequency", {"stderr": -4, "unit": "caller unit"})
    document = YAML(typ="safe").load(path.read_text())
    assert document["future"] == {"note": "preserved"}
    assert document["components"]["qubit"]["frequency"] == 6789
    assert document["metadata"]["qubit.frequency"] == {
        "stderr": -4,
        "unit": "caller unit",
    }


@pytest.mark.parametrize(
    "header, error",
    [("format: different", FormatError), ("format_version: '3.0'", VersionError)],
)
def test_header_rejection_on_refresh_keeps_cached_snapshot(
    path: Path, header: str, error: type[ValueError]
) -> None:
    params = open_values(path, Values)
    before = params.snapshot()
    lines = path.read_text().splitlines()
    lines[0 if header.startswith("format:") else 1] = header
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(error):
        params.refresh()
    assert params.snapshot() == before


@pytest.mark.parametrize(
    "version, old_table", [("1.0", False), ("1.9", True), ("2.0", True), ("2.8", True)]
)
def test_old_parameter_format_is_rejected_with_path_and_versions(
    path: Path, version: str, old_table: bool
) -> None:
    content = path.read_text().replace("'2.0'", f"'{version}'")
    if old_table:
        content += "provenance: {}\n"
    path.write_text(content)
    with pytest.raises(ValueError) as rejected:
        open_values(path, Values)
    message = str(rejected.value)
    assert str(path) in message
    assert version in message
    assert "2.0" in message


def test_missing_document_is_not_created(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        open_values(path, Values)
