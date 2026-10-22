"""Typed metadata-only reads for saved-run lists and load dialogs."""

import json
from pathlib import Path

import h5py
import pytest
from pydantic import TypeAdapter
from zcu_tools import datafile
from zcu_tools.experiment import native_persistence

from tests._native_support import native_metadata


@pytest.fixture
def metadata_doc() -> datafile.JsonObject:
    """Serialize historical evidence through the public metadata JSON contract."""
    return TypeAdapter(datafile.JsonObject).validate_python(
        native_metadata().model_dump(mode="json")
    )


@pytest.fixture
def metadata_path(tmp_path: Path, metadata_doc: datafile.JsonObject) -> Path:
    """Create a generic container without requiring experiment cfg or arrays."""
    path = tmp_path / "saved-run.h5"
    datafile.save(
        path,
        datafile.Container(
            {},
            {"metadata": metadata_doc},
            {},
            datafile.ContainerAttrs("3.0", "synthetic", "experiment-run", {}),
        ),
    )
    return path


def test_metadata_from_container_returns_typed_evidence_without_io(
    metadata_doc: datafile.JsonObject,
) -> None:
    metadata_doc["future"] = {"unknown": [True, None]}
    container = datafile.Container(
        {},
        {"metadata": metadata_doc},
        {},
        datafile.ContainerAttrs("3.0", "synthetic", "experiment-run", {}),
    )
    actual = native_persistence.metadata_from_container(container)
    assert actual == native_metadata()
    assert container.docs["metadata"]["future"] == {"unknown": [True, None]}


def test_metadata_from_container_rejects_missing_metadata() -> None:
    container = datafile.Container(
        {}, {}, {}, datafile.ContainerAttrs("3.0", "synthetic", None, {})
    )
    with pytest.raises(ValueError, match="/docs/metadata"):
        native_persistence.metadata_from_container(container)


@pytest.mark.parametrize("completion", ["saved", "", 4])
def test_metadata_from_container_rejects_invalid_metadata(
    metadata_doc: datafile.JsonObject, completion: datafile.JsonScalar
) -> None:
    metadata_doc["completion"] = completion
    container = datafile.Container(
        {},
        {"metadata": metadata_doc},
        {},
        datafile.ContainerAttrs("3.0", "synthetic", None, {}),
    )
    with pytest.raises(ValueError, match="/docs/metadata"):
        native_persistence.metadata_from_container(container)


def test_read_run_metadata_returns_typed_evidence_despite_unreadable_vars(
    metadata_path: Path,
) -> None:
    with h5py.File(metadata_path, "r+") as file:
        file.create_group("vars/broken")
        file.create_dataset("docs/cfg", data="not JSON")
    with pytest.raises(datafile.FormatError):
        datafile.load(metadata_path)
    actual = native_persistence.read_run_metadata(metadata_path)
    assert actual == native_metadata()
    assert actual.snapshot.entry_id.hex == "12345678123456781234567812345678"
    assert actual.provenance.hostname == "synthetic-host"


def test_read_run_metadata_missing_doc_names_source(metadata_path: Path) -> None:
    with h5py.File(metadata_path, "r+") as file:
        del file["docs/metadata"]
    with pytest.raises(KeyError) as caught:
        native_persistence.read_run_metadata(metadata_path)
    assert str(metadata_path) in str(caught.value)
    assert "metadata" in str(caught.value)


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": "invalid"},
        {"completion": "saved"},
        {"started_at": "2026-10-06T01:00:00+08:00"},
        {"snapshot": {"entry_id": "not-a-uuid"}},
        {"provenance": {"git_dirty": "unknown"}},
    ],
)
def test_read_run_metadata_rejects_invalid_typed_doc(
    metadata_path: Path,
    metadata_doc: datafile.JsonObject,
    change: datafile.JsonObject,
) -> None:
    metadata_doc.update(change)
    with h5py.File(metadata_path, "r+") as file:
        del file["docs/metadata"]
        file.create_dataset("docs/metadata", data=json.dumps(metadata_doc))
    with pytest.raises(ValueError) as caught:
        native_persistence.read_run_metadata(metadata_path)
    assert str(metadata_path) in str(caught.value)
    assert "metadata" in str(caught.value)
