"""Selective document reads from native containers."""

import json
from pathlib import Path

import h5py
import pytest
from zcu_tools import datafile


@pytest.fixture
def document_file(tmp_path: Path) -> Path:
    """Write docs plus an unreadable variable without loading it."""
    path = tmp_path / "documents.h5"
    with h5py.File(path, "w") as file:
        file.attrs.update(
            format="zcu.container", format_version="3.9", producer="synthetic"
        )
        file.create_group("vars").create_group("broken")
        file.create_group("groups")
        docs = file.create_group("docs")
        docs.create_dataset("cfg", data='{"values":{"label":"保留"}}')
        docs.create_dataset("metadata", data='{"run_id":"historical"}')
        docs.create_dataset("unrelated", data="not JSON")
    return path


def test_read_doc_returns_one_detached_object_without_loading_vars(
    document_file: Path,
) -> None:
    with pytest.raises(datafile.FormatError):
        datafile.load(document_file)
    actual = datafile.read_doc(document_file, "cfg")
    document_file.unlink()
    assert actual["values"] == {"label": "保留"}


def test_read_docs_detaches_valid_documents_alongside_individual_failures(
    document_file: Path,
) -> None:
    target = document_file.with_name("external.h5")
    with h5py.File(target, "w") as file:
        file.create_dataset("doc", data='{"outside":true}')
    with h5py.File(document_file, "r+") as file:
        file["docs/external"] = h5py.ExternalLink(str(target), "/doc")
        file["docs/unrequested"] = h5py.ExternalLink(str(target), "/does-not-exist")

    requested = ("cfg", "absent", "unrelated", "external", "metadata")
    results = datafile.read_docs(document_file, requested)

    for name, category in (
        ("absent", KeyError),
        ("unrelated", datafile.FormatError),
        ("external", datafile.FormatError),
    ):
        with pytest.raises(category) as single:
            datafile.read_doc(document_file, name)
        assert isinstance(results[name], category)
        assert str(results[name]) == str(single.value)
    document_file.unlink()
    assert tuple(results) == requested
    assert results["cfg"] == {"values": {"label": "保留"}}
    assert results["metadata"] == {"run_id": "historical"}


def test_read_docs_bad_header_raises_same_path_qualified_error_as_read_doc(
    document_file: Path,
) -> None:
    with h5py.File(document_file, "r+") as file:
        file.attrs["format_version"] = "4.0"
    with pytest.raises(datafile.FormatError) as single:
        datafile.read_doc(document_file, "cfg")
    with pytest.raises(datafile.FormatError) as batch:
        datafile.read_docs(document_file, ("cfg", "metadata"))
    assert str(batch.value) == str(single.value)


def test_read_doc_ignores_invalid_unselected_document(document_file: Path) -> None:
    with h5py.File(document_file, "r+") as file:
        cfg_path = "docs/cfg"
        del file[cfg_path]
        file.create_dataset(cfg_path, data="not JSON")
    expected: datafile.JsonObject = {"run_id": "historical"}
    assert datafile.read_doc(document_file, "metadata") == expected


def test_missing_doc_raises_key_error_naming_source(document_file: Path) -> None:
    with pytest.raises(KeyError) as caught:
        datafile.read_doc(document_file, "absent")
    assert str(document_file) in str(caught.value)
    assert "absent" in str(caught.value)


@pytest.mark.parametrize("name", ["", ".", "..", "cfg/values", "a\0b"])
def test_read_doc_rejects_nonsegment_names(document_file: Path, name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        datafile.read_doc(document_file, name)


@pytest.mark.parametrize(
    "text", ["[]", "null", '{"value":NaN}', '{"value":Infinity}', "not JSON"]
)
def test_read_doc_rejects_nonobject_or_nonfinite_json(
    document_file: Path, text: str
) -> None:
    with h5py.File(document_file, "r+") as file:
        cfg_path = "docs/cfg"
        del file[cfg_path]
        file.create_dataset(cfg_path, data=text)
    with pytest.raises(datafile.FormatError) as caught:
        datafile.read_doc(document_file, "cfg")
    assert str(document_file) in str(caught.value)
    assert "cfg" in str(caught.value)


@pytest.mark.parametrize("version", ["1.0", "4.0", "invalid"])
def test_read_doc_rejects_unsupported_container_version(
    tmp_path: Path, version: str
) -> None:
    path = tmp_path / "unsupported.h5"
    with h5py.File(path, "w") as file:
        file.attrs.update(
            format="zcu.container", format_version=version, producer="test"
        )
        file.create_dataset("docs/note", data=json.dumps({"label": "example"}))
    with pytest.raises(datafile.FormatError) as caught:
        datafile.read_doc(path, "note")
    assert str(path) in str(caught.value)


def test_read_doc_rejects_external_document_link(tmp_path: Path) -> None:
    target = tmp_path / "target.h5"
    with h5py.File(target, "w") as file:
        file.create_dataset("note", data='{"label":"external"}')
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as file:
        file.attrs.update(format="zcu.container", format_version="3.0", producer="test")
        file.create_group("docs")["note"] = h5py.ExternalLink(str(target), "/note")
    with pytest.raises(datafile.FormatError, match="external") as caught:
        datafile.read_doc(source, "note")
    assert str(source) in str(caught.value)
