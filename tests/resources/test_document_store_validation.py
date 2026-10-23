"""Document-wide validation and content-keyed reads through public store edits."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import TypeAdapter
from zcu_tools.format_version import FormatVersion, YamlMap
from zcu_tools.resources.document_store import DocumentStore

from tests.resources._document_store_fakes import SyntheticDocument


def make_store(path: Path) -> DocumentStore[SyntheticDocument]:
    return DocumentStore(path, SyntheticDocument, format="synthetic")


def test_edit_validates_complete_projection_without_revalidating_each_leaf(
    document_path: Path,
) -> None:
    with patch.object(
        TypeAdapter,
        "validate_python",
        autospec=True,
        side_effect=TypeAdapter.validate_python,
    ) as validate:
        store = make_store(document_path)
        with store.edit() as draft:
            draft.values["left"] = 10.0

    assert store.snapshot().values == {"left": 10.0, "right": 2.0}
    scalar_validations = [
        call.args[1]
        for call in validate.call_args_list
        if isinstance(call.args[1], (str, int, float, bool))
    ]
    assert scalar_validations == []


def test_unchanged_reads_reuse_validation_and_isolate_mutable_snapshots(
    document_path: Path,
) -> None:
    validations: list[float] = []
    cleanups: list[FormatVersion] = []

    def validate(document: SyntheticDocument) -> None:
        validations.append(document.values["left"])

    def cleanup(document: YamlMap, version: FormatVersion) -> tuple[str, ...]:
        cleanups.append(version)
        return ()

    store = DocumentStore(
        document_path,
        SyntheticDocument,
        format="synthetic",
        validate=validate,
        read_cleanup=cleanup,
    )
    assert not store.refresh()
    document_path.touch()
    assert not store.refresh()
    store.snapshot().values["left"] = 99.0
    with pytest.raises(RuntimeError, match="abort"), store.edit() as draft:
        draft.values["left"] = 88.0
        raise RuntimeError("abort")
    with store.edit() as draft:
        assert draft.values == {"left": 1.0, "right": 2.0}
    assert store.snapshot().values == {"left": 1.0, "right": 2.0}
    assert validations == [1.0]
    assert cleanups == [FormatVersion(1, 0)]


def test_external_bytes_revalidate_even_with_identical_size_and_mtime(
    document_path: Path,
) -> None:
    validations: list[float] = []

    def validate(document: SyntheticDocument) -> None:
        validations.append(document.values["left"])

    store = DocumentStore(
        document_path, SyntheticDocument, format="synthetic", validate=validate
    )
    before = document_path.stat()
    original = document_path.read_bytes()
    document_path.write_bytes(original.replace(b"left: 1.0", b"left: 3.0"))
    os.utime(document_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert document_path.stat().st_size == before.st_size
    assert store.refresh()
    assert not store.refresh()
    assert store.snapshot().values == {"left": 3.0, "right": 2.0}
    assert validations == [1.0, 3.0]


def test_nonempty_edit_validates_candidate_without_revalidating_unchanged_reads(
    document_path: Path,
) -> None:
    validations: list[float] = []

    def validate(document: SyntheticDocument) -> None:
        validations.append(document.values["left"])

    store = DocumentStore(
        document_path, SyntheticDocument, format="synthetic", validate=validate
    )
    with store.edit() as draft:
        draft.values["left"] = 10.0
    assert store.snapshot().values == {"left": 10.0, "right": 2.0}
    assert validations == [1.0, 10.0]


def test_cleanup_replacement_is_cached_only_after_successful_validation(
    document_path: Path,
) -> None:
    document_path.write_text(
        "format: synthetic\nformat_version: '1.0'\n"
        "values: {left: 1.0, right: 2.0, bad: -1.0}\n",
        encoding="utf-8",
    )
    cleanups: list[FormatVersion] = []
    validations: list[float] = []

    def cleanup(document: YamlMap, version: FormatVersion) -> tuple[str, ...]:
        cleanups.append(version)
        values = document["values"]
        if isinstance(values, dict) and "bad" in values:
            del values["bad"]
            return ("bad",)
        return ()

    def validate(document: SyntheticDocument) -> None:
        validations.append(document.values["left"])

    store = DocumentStore(
        document_path,
        SyntheticDocument,
        format="synthetic",
        validate=validate,
        read_cleanup=cleanup,
    )
    assert not store.refresh()
    with store.edit():
        pass
    assert store.snapshot().values == {"left": 1.0, "right": 2.0}
    assert validations == [1.0]
    assert cleanups == [FormatVersion(1, 0)]


def test_failed_external_validation_is_retried_without_polluting_cached_values(
    document_path: Path,
) -> None:
    validations: list[float] = []

    def validate(document: SyntheticDocument) -> None:
        left = document.values["left"]
        validations.append(left)
        if left == 3.0:
            raise ValueError("rejected external value")

    store = DocumentStore(
        document_path, SyntheticDocument, format="synthetic", validate=validate
    )
    original = document_path.read_bytes()
    document_path.write_bytes(original.replace(b"left: 1.0", b"left: 3.0"))
    for _ in range(2):
        with pytest.raises(ValueError, match="rejected external value"):
            store.refresh()
    assert store.snapshot().values == {"left": 1.0, "right": 2.0}
    document_path.write_bytes(original)
    assert not store.refresh()
    assert validations == [1.0, 3.0, 3.0]
