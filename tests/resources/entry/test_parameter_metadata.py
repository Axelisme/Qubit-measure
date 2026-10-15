"""Entry setup/point values with opaque caller metadata in one parameter document."""

from contextlib import ExitStack
from pathlib import Path

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.document_store import ConflictError
from zcu_tools.resources.entry import PointView, ResultEntry, SetupView

from .view_support import field_view


@pytest.fixture(params=["setup", "point"])
def container(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[ResultEntry, SetupView | PointView, Path]:
    root = tmp_path / "results"
    entry = ResultEntry.create("entry", result_root=root, database_root=tmp_path / "db")
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=5000.0)
    if request.param == "setup":
        return entry, entry.setup, root / "entry/setup.yaml"
    return entry, entry.new_point("a"), root / "entry/points/a/point.yaml"


def reopen(tmp_path: Path, view: SetupView | PointView) -> SetupView | PointView:
    entry = ResultEntry.open(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "db"
    )
    return entry.use_point("a") if isinstance(view, PointView) else entry.setup


def test_values_and_metadata_commit_together_and_reopen(
    container: tuple[ResultEntry, SetupView | PointView, Path], tmp_path: Path
) -> None:
    _entry, view, source = container
    metadata: YamlMap = {
        "stderr": -0.2,
        "source": "outside/run/unknown",
        "comment": ["caller", {"nested": None}],
    }
    with view.edit() as draft:
        draft.set("Q1.rate", 5300.0)
        draft.set_meta("Q1.rate", metadata)
        assert draft.meta("Q1.rate") == metadata
        assert view.meta("Q1.rate") is None
    assert view.Q1.rate == 5300.0
    assert view.meta("Q1.rate") == metadata
    stored = YAML(typ="safe").load(source.read_text())
    assert stored["format_version"] == "2.0"
    assert stored["components"]["Q1"]["rate"] == pytest.approx(5300.0)
    assert stored["metadata"]["Q1.rate"] == metadata
    restored = reopen(tmp_path, view)
    assert restored.Q1.rate == 5300.0
    assert restored.meta("Q1.rate") == metadata


@pytest.mark.parametrize(
    ("write", "path"),
    [
        ("attribute", "Q1.rate"),
        ("draft_attribute", "Q1.rate"),
        ("set", "Q1.rate"),
        ("container", "Q1.wiring.delay"),
        ("nested_ext", "Q1.ext.batch.rate"),
        ("description", "general.description"),
        ("general_ext", "general.ext.temperature"),
    ],
)
def test_value_writes_keep_existing_metadata(
    container: tuple[ResultEntry, SetupView | PointView, Path], write: str, path: str
) -> None:
    _entry, view, source = container
    metadata: YamlMap = {"stderr": 0.4, "note": "caller policy"}
    with view.edit() as draft:
        draft.set_meta(path, metadata)
    if write == "attribute":
        view.Q1.rate = 5300.0
    elif write == "container":
        with view.edit() as draft:
            draft.set("Q1.wiring", {"ch": 3, "delay": 1.2})
    elif write == "nested_ext":
        field_view(view.Q1.ext).batch = {"rate": 12.0}
    elif write == "description":
        view.description = "accepted point"
    elif write == "general_ext":
        view.general.ext.temperature = 11.0
    else:
        with view.edit() as draft:
            if write == "draft_attribute":
                draft.Q1.rate = 5300.0
            else:
                draft.set("Q1.rate", 5300.0)
    assert view.meta(path) == metadata
    stored = YAML(typ="safe").load(source.read_text())
    assert stored["metadata"] == {path: metadata}


@pytest.mark.parametrize("key", ["stderr", "source", "comment", "caller_note"])
def test_metadata_replacement_and_clear_keep_values_and_detach_reads(
    container: tuple[ResultEntry, SetupView | PointView, Path], key: str
) -> None:
    _entry, view, _source = container
    metadata: YamlMap = {key: {"values": [-9, None, True], "external": "missing/run"}}
    with view.edit() as draft:
        draft.set_meta("Q1.rate", {"stale": 1, key: "old"})
    with view.edit() as draft:
        draft.set_meta("Q1.rate", metadata)
    assert view.Q1.rate == 5000.0
    metadata[key] = "mutated caller mapping"
    readback = view.meta("Q1.rate")
    assert readback == {key: {"values": [-9, None, True], "external": "missing/run"}}
    assert readback is not None
    readback[key] = "mutated readback"
    assert view.meta("Q1.rate") == {
        key: {"values": [-9, None, True], "external": "missing/run"}
    }
    with view.edit() as draft:
        draft.set_meta("Q1.rate", {})
    assert view.meta("Q1.rate") == {}
    assert view.Q1.rate == 5000.0


@pytest.mark.parametrize(
    "metadata", [{"bad": object()}, {1: "non-string key"}, {"bad": (1, 2)}]
)
def test_invalid_metadata_representation_rejects_whole_edit(
    container: tuple[ResultEntry, SetupView | PointView, Path], metadata: object
) -> None:
    _entry, view, source = container
    before = source.read_bytes()

    def invalid_edit() -> None:
        with view.edit() as draft:
            draft.set("Q1.rate", 5300.0)
            draft.set_meta("Q1.rate", metadata)

    with pytest.raises(ValidationError):
        invalid_edit()
    assert view.Q1.rate == 5000.0
    assert view.meta("Q1.rate") is None
    assert source.read_bytes() == before


@pytest.mark.parametrize("failure", ["body", "schema"])
def test_failed_value_transaction_discards_its_metadata(
    container: tuple[ResultEntry, SetupView | PointView, Path], failure: str
) -> None:
    _entry, view, source = container
    before = source.read_bytes()
    error = RuntimeError if failure == "body" else ValidationError

    def fail_transaction() -> None:
        with view.edit() as draft:
            draft.set("Q1.rate", 5300.0)
            draft.set_meta("Q1.rate", {"stderr": 0.2})
            if failure == "body":
                raise RuntimeError("abort")
            draft.Q1.duration = "not-a-number"

    with pytest.raises(error):
        fail_transaction()
    assert view.Q1.rate == 5000.0
    assert view.meta("Q1.rate") is None
    assert source.read_bytes() == before


def test_conflict_keeps_the_winning_value_and_metadata_together(
    container: tuple[ResultEntry, SetupView | PointView, Path], tmp_path: Path
) -> None:
    _entry, view, source = container
    other = reopen(tmp_path, view)
    with ExitStack() as transaction:
        draft = transaction.enter_context(view.edit())
        draft.set("Q1.rate", 5300.0)
        draft.set_meta("Q1.rate", {"stderr": 0.2})
        with other.edit() as competing:
            competing.set("Q1.rate", 5400.0)
            competing.set_meta("Q1.rate", {"stderr": 0.3})
        winning_bytes = source.read_bytes()
        with pytest.raises(ConflictError):
            transaction.close()
    assert source.read_bytes() == winning_bytes
    view.refresh()
    assert view.Q1.rate == 5400.0
    assert view.meta("Q1.rate") == {"stderr": 0.3}


def test_seed_copies_setup_metadata_into_an_independent_point(tmp_path: Path) -> None:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "db"
    )
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=5000.0)
    with entry.setup.edit() as draft:
        draft.set_meta("Q1.rate", {"stderr": 0.1, "cloned_from": "caller text"})
    point = entry.new_point("seed")
    assert point.meta("Q1.rate") == {"stderr": 0.1, "cloned_from": "caller text"}
    with entry.setup.edit() as draft:
        draft.set_meta("Q1.rate", {"stderr": 0.9})
    assert point.meta("Q1.rate") == {"stderr": 0.1, "cloned_from": "caller text"}
    with point.edit() as draft:
        draft.set_meta("Q1.rate", {"comment": "point only"})
    assert entry.setup.meta("Q1.rate") == {"stderr": 0.9}


@pytest.mark.parametrize("clone_from", ["label", "view"])
def test_same_entry_clone_copies_values_and_metadata_verbatim(
    tmp_path: Path, clone_from: str
) -> None:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "db"
    )
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=5000.0)
    original = entry.new_point("original")
    metadata: dict[str, YamlMap] = {
        "Q1.rate": {"source": "outside/run", "stderr": 0.2},
        "general.ext.flux": {"cloned_from": {"caller": "kept"}},
        "Q1.missing": {"comment": "metadata without a value"},
    }
    with original.edit() as draft:
        draft.set("Q1.rate", 5300.0)
        draft.set("general.ext.flux", 1.5)
        for path, mapping in metadata.items():
            draft.set_meta(path, mapping)
    clone = entry.new_point(
        "clone", clone_from="original" if clone_from == "label" else original
    )
    assert clone.Q1.rate == 5300.0
    for path, mapping in metadata.items():
        assert clone.meta(path) == mapping
    stored = YAML(typ="safe").load(
        (tmp_path / "results/entry/points/clone/point.yaml").read_text()
    )
    assert stored["metadata"] == metadata
    with clone.edit() as draft:
        draft.set_meta("Q1.rate", {"comment": "clone only"})
    assert original.meta("Q1.rate") == metadata["Q1.rate"]


@pytest.mark.parametrize("document", ["setup", "point"])
@pytest.mark.parametrize(
    ("version", "old_table"),
    [("1.0", False), ("1.9", True), ("2.0", True), ("2.8", True)],
)
def test_old_parameter_documents_are_rejected_with_path_and_versions(
    tmp_path: Path, document: str, version: str, old_table: bool
) -> None:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "db"
    )
    entry.new_point("a")
    source = tmp_path / (
        "results/entry/setup.yaml"
        if document == "setup"
        else "results/entry/points/a/point.yaml"
    )
    stored = YAML(typ="safe").load(source.read_text())
    stored["format_version"] = version
    if old_table:
        stored["provenance"] = {}
    with source.open("w", encoding="utf-8") as stream:
        YAML().dump(stored, stream)
    before = source.read_bytes()

    def reopen_point() -> None:
        reopened = ResultEntry.open(
            "entry", result_root=tmp_path / "results", database_root=tmp_path / "db"
        )
        reopened.use_point("a")

    with pytest.raises(ValueError, match="2.0") as rejected:
        reopen_point()
    message = str(rejected.value)
    assert str(source) in message
    assert version in message
    assert source.read_bytes() == before
