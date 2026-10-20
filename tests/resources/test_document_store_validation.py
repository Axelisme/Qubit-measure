"""Document-wide serialization and read validation through public store edits."""

from pathlib import Path
from unittest.mock import patch

from pydantic import TypeAdapter
from ruamel.yaml import YAML
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


def test_edit_reuses_unchanged_yaml_bytes_but_revalidates_each_read(
    document_path: Path,
) -> None:
    validations: list[float] = []

    def validate(document: SyntheticDocument) -> None:
        validations.append(document.values["left"])

    store = DocumentStore(
        document_path, SyntheticDocument, format="synthetic", validate=validate
    )
    with (
        patch.object(YAML, "load", autospec=True, side_effect=YAML.load) as load,
        store.edit() as draft,
    ):
        draft.values["left"] = 10.0
    assert store.snapshot().values == {"left": 10.0, "right": 2.0}
    assert validations == [1.0, 1.0, 1.0, 10.0]
    assert load.call_count == 0

    document_path.write_text(
        "format: synthetic\nformat_version: '1.0'\nvalues: {left: 30.0, right: 2.0}\n",
        encoding="utf-8",
    )
    assert store.refresh()
    assert store.snapshot().values == {"left": 30.0, "right": 2.0}
