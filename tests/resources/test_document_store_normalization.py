from contextlib import ExitStack
from enum import StrEnum
from pathlib import Path

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    model_serializer,
    model_validator,
)
from ruamel.yaml import YAML
from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.resources.document_store import (
    ConflictError,
    DocumentChange,
    DocumentStore,
)


class ModernValues(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modern: int = Field(ge=0)
    right: int

    @model_validator(mode="before")
    @classmethod
    def rename_legacy(cls, value: YamlMap) -> YamlMap:
        converted = dict(value)
        if "legacy" in converted:
            converted["modern"] = converted.pop("legacy")
        return converted


class ModernDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: str
    format_version: str
    values: ModernValues
    description: str


@pytest.mark.parametrize("shape", ["scalar", "collision", "relocated"])
def test_future_extras_reject_reshaped_serialization_without_publication(
    tmp_path: Path, shape: str
) -> None:
    class ShapedValues(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: int

        @model_serializer
        def serialize_values(self) -> YamlValue:
            if shape == "scalar":
                return str(self.value)
            if shape == "collision":
                return {"value": self.value, "future": "serializer-owned"}
            return {"value": self.value}

    class ShapedDocument(BaseModel):
        model_config = ConfigDict(
            extra="forbid", serialize_by_alias=shape == "relocated"
        )
        format: str
        format_version: str
        values: ShapedValues = Field(serialization_alias="relocated")

    path = tmp_path / "shaped.yaml"
    path.write_text(
        "format: synthetic\nformat_version: '1.0'\nvalues:\n  value: 7\n",
        encoding="utf-8",
    )
    store = DocumentStore(path, ShapedDocument, format="synthetic")
    changes: list[DocumentChange] = []
    store.subscribe(changes.append)
    path.write_text(
        "format: synthetic\nformat_version: '1.2'\nvalues:\n"
        "  value: 8\n  future: next\n",
        encoding="utf-8",
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="temporary extras"):
        store.refresh()
    with pytest.raises(ValueError, match="temporary extras"), store.edit():
        pass
    with pytest.raises(ValueError, match="temporary extras"):
        DocumentStore(path, ShapedDocument, format="synthetic")
    assert path.read_bytes() == before
    assert store.snapshot().values.value == 7
    assert changes == []


@pytest.mark.parametrize("version", ["1.0", "1.2"])
def test_legacy_conversion_is_read_only_until_a_nonempty_commit(
    document_path: Path, version: str
) -> None:
    future = (
        "  future: 'next' # nested future\nfuture_top: [next] # top future\n"
        if version == "1.2"
        else ""
    )
    document_path.write_text(
        f"format: synthetic\nformat_version: '{version}'\nvalues:\n"
        "  legacy: 7 # obsolete\n  right: 2 # untouched\n"
        + future
        + "description: 'before' # description\n",
        encoding="utf-8",
    )
    before = document_path.read_bytes()
    store = DocumentStore(document_path, ModernDocument, format="synthetic")
    changes: list[DocumentChange] = []
    store.subscribe(changes.append)
    assert store.snapshot().values.modern == 7
    assert store.snapshot().values.model_extra in (None, {})
    assert "future" not in store.snapshot().values.model_fields_set
    assert not store.refresh()
    with store.edit():
        pass
    assert document_path.read_bytes() == before
    assert changes == []
    document_path.write_text(
        document_path.read_text().replace("legacy: 7", "legacy: 8"), encoding="utf-8"
    )
    refreshed_bytes = document_path.read_bytes()
    assert store.refresh()
    assert document_path.read_bytes() == refreshed_bytes
    assert store.snapshot().values.modern == 8
    changes.clear()
    with store.edit() as draft:
        draft.description = "after"
    saved = YAML(typ="safe").load(document_path)
    assert saved["values"]["modern"] == 8
    assert "legacy" not in saved["values"]
    assert saved["format_version"] == version
    text = document_path.read_text()
    assert "right: 2 # untouched" in text
    if version == "1.2":
        assert "future: 'next' # nested future" in text
        assert "future_top: [next] # top future" in text
    assert len(changes) == 1
    assert changes[0].reason == "commit"
    assert changes[0].paths == (("description",),)
    assert (
        DocumentStore(document_path, ModernDocument, format="synthetic")
        .snapshot()
        .values.modern
        == 8
    )


@pytest.mark.parametrize("representation", ["legacy", "modern"])
def test_concurrent_logical_leaf_conflicts_across_legacy_and_modern_keys(
    document_path: Path, representation: str
) -> None:
    document_path.write_text(
        "format: synthetic\nformat_version: '1.0'\nvalues:\n"
        "  legacy: 7\n  right: 2\ndescription: before\n",
        encoding="utf-8",
    )
    store = DocumentStore(document_path, ModernDocument, format="synthetic")
    changes: list[DocumentChange] = []
    store.subscribe(changes.append)
    with ExitStack() as stack:
        draft = stack.enter_context(store.edit())
        draft.values.modern = 10
        draft.description = "must not publish"
        document_path.write_text(
            "format: synthetic\nformat_version: '1.0'\nvalues:\n"
            f"  {representation}: 20\n  right: 2\ndescription: before\n",
            encoding="utf-8",
        )
        winner = document_path.read_bytes()
        with pytest.raises(ConflictError) as caught:
            stack.close()
    assert caught.value.path == ("values", "modern")
    assert caught.value.original == 7 and caught.value.current == 20
    assert document_path.read_bytes() == winner
    assert store.snapshot().values.modern == 7
    assert store.snapshot().description == "before"
    assert changes == []


@pytest.mark.parametrize("representation", ["legacy", "modern"])
def test_different_leaf_commit_uses_latest_conversion_not_base_values(
    document_path: Path, representation: str
) -> None:
    document_path.write_text(
        "format: synthetic\nformat_version: '1.2'\nvalues:\n"
        "  legacy: 7\n  right: 2\n  future: initial\ndescription: before\n",
        encoding="utf-8",
    )
    store = DocumentStore(document_path, ModernDocument, format="synthetic")
    with store.edit() as draft:
        draft.values.right = 9
        document_path.write_text(
            "format: synthetic\nformat_version: '1.2'\nvalues:\n"
            f"  {representation}: 20\n  right: 2\n  future: 'latest' # future\n"
            "description: before\n",
            encoding="utf-8",
        )
    saved = YAML(typ="safe").load(document_path)
    assert saved["values"] == {"modern": 20, "right": 9, "future": "latest"}
    assert store.snapshot().values.modern == 20
    assert store.snapshot().values.right == 9
    assert "future: 'latest' # future" in document_path.read_text()


@pytest.mark.parametrize("extra_policy", ["forbid", "ignore"])
def test_forward_models_inside_dict_list_tuple_hide_extras_but_preserve_yaml(
    document_path: Path, extra_policy: str
) -> None:
    class Nested(BaseModel):
        model_config = ConfigDict(
            extra="forbid" if extra_policy == "forbid" else "ignore"
        )
        value: int

    class Containers(BaseModel):
        model_config = ConfigDict(extra="forbid")
        format: str
        format_version: str
        mapping: dict[str, Nested]
        sequence: list[Nested]
        pair: tuple[Nested, Nested]
        description: str

    document_path.write_text(
        "format: synthetic\nformat_version: '1.2'\n"
        "mapping:\n  first: {value: 1, future: mapping}\n"
        "sequence:\n  - {value: 2, future: sequence}\n"
        "pair:\n  - {value: 3, future: left}\n  - {value: 4, future: right}\n"
        "description: before\n",
        encoding="utf-8",
    )
    before = document_path.read_bytes()
    store = DocumentStore(document_path, Containers, format="synthetic")
    snapshot = store.snapshot()
    nodes = [snapshot.mapping["first"], snapshot.sequence[0], *snapshot.pair]
    for node in nodes:
        assert node.model_dump() == {"value": node.value}
        assert "future" not in node.model_fields_set
        assert node.model_extra in (None, {})
    assert isinstance(snapshot.pair, tuple)
    assert document_path.read_bytes() == before
    with store.edit() as draft:
        draft.description = "after"
    saved = YAML(typ="safe").load(document_path)
    assert saved["mapping"]["first"] == {"value": 1, "future": "mapping"}
    assert saved["sequence"][0] == {"value": 2, "future": "sequence"}
    assert saved["pair"] == [
        {"value": 3, "future": "left"},
        {"value": 4, "future": "right"},
    ]
    reopened = DocumentStore(document_path, Containers, format="synthetic").snapshot()
    assert reopened.model_dump() == {**snapshot.model_dump(), "description": "after"}


def test_current_ignore_policy_and_retained_legacy_keys_follow_serialization(
    document_path: Path,
) -> None:
    class Ignoring(BaseModel):
        model_config = ConfigDict(extra="ignore")
        format: str
        format_version: str
        modern: int
        description: str

        @model_validator(mode="before")
        @classmethod
        def copy_legacy(cls, value: YamlMap) -> YamlMap:
            converted = dict(value)
            if "legacy" in converted:
                converted["modern"] = converted["legacy"]
            return converted

    for version in ("1.0", "1.2"):
        document_path.write_text(
            f"format: synthetic\nformat_version: '{version}'\n"
            "legacy: 7\nunknown: future\ndescription: before\n",
            encoding="utf-8",
        )
        store = DocumentStore(document_path, Ignoring, format="synthetic")
        assert store.snapshot().modern == 7
        with store.edit() as draft:
            draft.description = "after"
        saved = YAML(typ="safe").load(document_path)
        assert saved["modern"] == 7
        if version == "1.0":
            assert "legacy" not in saved and "unknown" not in saved
        else:
            assert saved["legacy"] == 7 and saved["unknown"] == "future"


def test_coercion_equivalent_raw_nodes_survive_an_unrelated_commit(
    document_path: Path,
) -> None:
    class Label(StrEnum):
        READY = "ready"

    class Coercible(BaseModel):
        format: str
        format_version: str
        number: float
        title: str
        label: Label
        converted: int
        description: str

    document_path.write_text(
        "format: synthetic\nformat_version: '1.0'\n"
        "number: 7 # same numeric value\ntitle: 'ready' # same string\n"
        'label: "ready" # enum string\nconverted: "123" # actual conversion\n'
        "description: before\n",
        encoding="utf-8",
    )
    store = DocumentStore(document_path, Coercible, format="synthetic")
    before = document_path.read_bytes()
    assert store.snapshot().converted == 123
    assert document_path.read_bytes() == before
    with store.edit() as draft:
        draft.description = "after"
    saved = document_path.read_text()
    assert "number: 7 # same numeric value" in saved
    assert "title: 'ready' # same string" in saved
    assert 'label: "ready" # enum string' in saved
    assert YAML(typ="safe").load(saved)["converted"] == 123


def test_commit_consumes_serialized_values_not_original_model_attributes(
    document_path: Path,
) -> None:
    class Serialized(BaseModel):
        format: str
        format_version: str
        value: int
        description: str

        @field_serializer("value")
        def padded_value(self, value: int) -> str:
            return f"{value:03d}"

    document_path.write_text(
        "format: synthetic\nformat_version: '1.0'\nvalue: 7\ndescription: before\n",
        encoding="utf-8",
    )
    store = DocumentStore(document_path, Serialized, format="synthetic")
    before = document_path.read_bytes()
    assert store.snapshot().value == 7
    assert store.snapshot().model_dump()["value"] == "007"
    assert document_path.read_bytes() == before
    with store.edit() as draft:
        draft.description = "after"
    assert YAML(typ="safe").load(document_path)["value"] == "007"
    assert store.snapshot().value == 7
    assert (
        DocumentStore(document_path, Serialized, format="synthetic").snapshot().value
        == 7
    )


@pytest.mark.parametrize(
    "failure_stage", ["conversion", "merged-model", "custom", "replace"]
)
def test_normalizing_commit_failure_preserves_disk_snapshot_and_notifications(
    document_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    class BoundedDocument(ModernDocument):
        @model_validator(mode="after")
        def bounded(self) -> "BoundedDocument":
            if self.values.modern + self.values.right > 10:
                raise ValueError("sum exceeds ten")
            return self

    def validate(document: ModernDocument) -> None:
        if document.values.modern + document.values.right > 10:
            raise ValueError("sum exceeds ten")

    def fail_replace(_source: Path, _target: Path | str) -> Path:
        raise OSError("replace unavailable")

    document_path.write_text(
        "format: synthetic\nformat_version: '1.0'\nvalues:\n"
        "  legacy: 7\n  right: 2\ndescription: before\n",
        encoding="utf-8",
    )
    store = DocumentStore(
        document_path,
        BoundedDocument if failure_stage == "merged-model" else ModernDocument,
        format="synthetic",
        validate=validate if failure_stage == "custom" else None,
    )
    before = store.snapshot().model_dump()
    changes: list[DocumentChange] = []
    store.subscribe(changes.append)
    with ExitStack() as stack:
        draft = stack.enter_context(store.edit())
        draft.description = "must not publish"
        if failure_stage in ("merged-model", "custom"):
            draft.values.modern = 8
            document_path.write_text(
                document_path.read_text().replace("right: 2", "right: 3"),
                encoding="utf-8",
            )
        elif failure_stage == "conversion":
            document_path.write_text(
                document_path.read_text().replace("legacy: 7", "legacy: -1"),
                encoding="utf-8",
            )
        else:
            monkeypatch.setattr(Path, "replace", fail_replace)
        winner = document_path.read_bytes()
        expected = OSError if failure_stage == "replace" else ValueError
        with pytest.raises(expected):
            stack.close()
    assert document_path.read_bytes() == winner
    assert store.snapshot().model_dump() == before
    assert changes == []
