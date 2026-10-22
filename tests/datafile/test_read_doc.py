"""Single-document reads across container and legacy native wire formats."""

import json
from pathlib import Path

import h5py
import pytest
from zcu_tools import datafile


@pytest.fixture(params=["container", "legacy"])
def document_file(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Write docs plus an unreadable variable in each supported wire format."""
    path = tmp_path / "documents.h5"
    with h5py.File(path, "w") as file:
        if request.param == "container":
            file.attrs.update(
                format="zcu.container", format_version="3.9", producer="synthetic"
            )
            file.create_group("vars").create_group("broken")
            file.create_group("groups")
            docs = file.create_group("docs")
            docs.create_dataset("cfg", data='{"values":{"label":"保留"}}')
            docs.create_dataset("metadata", data='{"run_id":"historical"}')
            docs.create_dataset("unrelated", data="not JSON")
        else:
            file.attrs.update(
                format="zcu.experiment-data",
                format_version="2.9",
                run_id="historical",
                experiment="synthetic",
                started_at="",
                finished_at="",
                completion="unknown",
            )
            file.create_group("data").create_group("broken").attrs["signal"] = "absent"
            cfg = file.create_dataset("cfg", data='{"label":"保留"}')
            cfg.attrs.update(cfg_type="SyntheticCfg", cfg_schema_version="1.0")
            file.create_dataset("context", data='{"entry_name":"historical entry"}')
            file.create_dataset("device", data='{"flux":{"value":0.25}}')
            provenance = file.create_group("provenance")
            provenance.attrs.update(
                software_versions='{"synthetic":"1.0"}', git_dirty="unknown"
            )
    return path


def test_read_doc_returns_one_detached_object_without_loading_vars(
    document_file: Path,
) -> None:
    with pytest.raises(datafile.FormatError):
        datafile.load(document_file)
    actual = datafile.read_doc(document_file, "cfg")
    document_file.unlink()
    assert actual["values"] == {"label": "保留"}


def test_read_doc_ignores_invalid_unselected_document(document_file: Path) -> None:
    with h5py.File(document_file, "r+") as file:
        cfg_path = "docs/cfg" if "docs" in file else "cfg"
        del file[cfg_path]
        file.create_dataset(cfg_path, data="not JSON")
        legacy = "docs" not in file
    expected: datafile.JsonObject = {"run_id": "historical"}
    if legacy:
        expected.update(
            experiment="synthetic",
            started_at=None,
            finished_at=None,
            completion="unknown",
            snapshot={"entry_name": "historical entry"},
            device_snapshot={"flux": {"value": 0.25}},
            provenance={
                "software_versions": {"synthetic": "1.0"},
                "git_dirty": None,
                "git_commit": None,
                "qick_version": None,
                "soc_fingerprint": None,
                "hostname": None,
                "simulation_id": None,
            },
        )
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
        cfg_path = "docs/cfg" if "docs" in file else "cfg"
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
