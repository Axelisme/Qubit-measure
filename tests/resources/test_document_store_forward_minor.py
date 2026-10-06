from contextlib import ExitStack
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict
from ruamel.yaml import YAML
from zcu_tools.resources.document_store import ConflictError, DocumentStore

from tests.resources._document_store_fakes import KnownValues, StrictDocument


class StrictSequenceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: str
    format_version: str
    values: list[KnownValues]


def make_strict_sequence_store(path: Path) -> DocumentStore[StrictSequenceDocument]:
    return DocumentStore(path, StrictSequenceDocument, format="synthetic")


def test_newer_minor_keeps_unknown_nested_fields_outside_the_typed_snapshot(
    document_path: Path,
) -> None:
    document_path.write_text(
        "format: synthetic\nformat_version: '1.2'\nvalues:\n"
        "  left: 1.0\n  right: 2.0\n  future: 7.000 # future nested\n"
        "future_top: [next, version] # future top\n",
        encoding="utf-8",
    )
    store = DocumentStore(document_path, StrictDocument, format="synthetic")
    assert store.snapshot().model_dump() == {
        "format": "synthetic",
        "format_version": "1.2",
        "values": {"left": 1.0, "right": 2.0},
    }
    with store.edit() as draft:
        draft.values.left = 10.0

    assert store.snapshot().values.left == 10.0
    assert store.snapshot().format_version == "1.2"
    text = document_path.read_text(encoding="utf-8")
    assert "  future: 7.000 # future nested" in text
    assert "future_top: [next, version] # future top" in text
    assert (
        DocumentStore(document_path, StrictDocument, format="synthetic")
        .snapshot()
        .values.left
        == 10.0
    )


def test_known_sequence_edit_preserves_future_fields_comments_and_untouched_nodes(
    document_path: Path,
) -> None:
    document_path.write_text(
        "format: synthetic\nformat_version: '1.2'\nvalues: # sequence note\n"
        "  - left: 1.000 # edited\n    right: 2.000 # unchanged\n"
        "    future: 7.000 # future nested\n"
        "  - left: 3.000 # untouched element\n    right: 4.000\n",
        encoding="utf-8",
    )
    store = make_strict_sequence_store(document_path)
    with store.edit() as draft:
        draft.values[0].left = 10.0

    assert store.snapshot().format_version == "1.2"
    assert store.snapshot().values[0].left == 10.0
    text = document_path.read_text(encoding="utf-8")
    assert "# sequence note" in text
    assert "# edited" in text
    lines = [" ".join(line.split()) for line in text.splitlines()]
    assert "right: 2.000 # unchanged" in lines
    assert "future: 7.000 # future nested" in lines
    assert any("left: 3.000 # untouched element" in line for line in lines)
    reopened = make_strict_sequence_store(document_path)
    assert reopened.snapshot() == store.snapshot()
    persisted = YAML(typ="safe").load(text)
    assert persisted["values"][0]["future"] == 7.0
    assert persisted["format_version"] == "1.2"


@pytest.mark.parametrize("operation", ["delete-middle", "insert-middle", "delete-tail"])
def test_resized_sequence_preserves_prefix_without_moving_future_fields(
    document_path: Path,
    operation: str,
) -> None:
    document_path.write_text(
        "format: synthetic\nformat_version: '1.2'\nvalues:\n"
        "  - left: 1.000 # prefix\n    right: 2.000\n    future: prefix\n"
        "  - left: 3.000\n    right: 4.000\n    future: middle\n"
        "  - left: 5.000\n    right: 6.000\n    future: tail\n",
        encoding="utf-8",
    )
    store = make_strict_sequence_store(document_path)
    prefix = {"left": 1.0, "right": 2.0, "future": "prefix"}
    with store.edit() as draft:
        if operation == "delete-middle":
            del draft.values[1]
            expected = [prefix, {"left": 5.0, "right": 6.0}]
        elif operation == "insert-middle":
            draft.values.insert(1, KnownValues(left=7.0, right=8.0))
            expected = [
                prefix,
                {"left": 7.0, "right": 8.0},
                {"left": 3.0, "right": 4.0},
                {"left": 5.0, "right": 6.0},
            ]
        else:
            del draft.values[-1]
            expected = [
                prefix,
                {"left": 3.0, "right": 4.0, "future": "middle"},
            ]
    text = document_path.read_text(encoding="utf-8")
    persisted = YAML(typ="safe").load(text)
    assert persisted["values"] == expected
    lines = [" ".join(line.split()) for line in text.splitlines()]
    assert any("left: 1.000 # prefix" in line for line in lines)
    assert persisted["format_version"] == "1.2"
    assert make_strict_sequence_store(document_path).snapshot() == store.snapshot()


def test_appending_to_typed_sequence_keeps_existing_future_nodes(
    document_path: Path,
) -> None:
    document_path.write_text(
        "format: synthetic\nformat_version: '1.2'\nvalues:\n"
        "  - left: 1.000 # original\n    right: 2.000\n"
        "    future: 7.000 # future nested\n",
        encoding="utf-8",
    )
    store = make_strict_sequence_store(document_path)
    with store.edit() as draft:
        draft.values.append(KnownValues(left=3.0, right=4.0))

    text = document_path.read_text(encoding="utf-8")
    assert "# original" in text
    assert "# future nested" in text
    persisted = YAML(typ="safe").load(text)
    assert persisted["values"] == [
        {"left": 1.0, "right": 2.0, "future": 7.0},
        {"left": 3.0, "right": 4.0},
    ]
    assert store.snapshot().format_version == "1.2"


def test_sequence_edits_conflict_as_a_whole_without_losing_future_nodes(
    document_path: Path,
) -> None:
    document_path.write_text(
        "format: synthetic\nformat_version: '1.2'\nvalues:\n"
        "  - left: 1.000\n    right: 2.000\n    future: 7.000 # future\n",
        encoding="utf-8",
    )
    first = make_strict_sequence_store(document_path)
    second = make_strict_sequence_store(document_path)
    with ExitStack() as stack:
        draft = stack.enter_context(first.edit())
        draft.values[0].left = 10.0
        with second.edit() as other:
            other.values[0].right = 20.0
        committed = document_path.read_bytes()
        with pytest.raises(ConflictError) as caught:
            stack.close()

    assert caught.value.path == ("values",)
    assert document_path.read_bytes() == committed
    assert "# future" in document_path.read_text(encoding="utf-8")
    assert first.snapshot().values[0].left == 1.0
    assert second.snapshot().values[0].right == 20.0
