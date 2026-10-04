"""Value/source transactions through the public entry and view interfaces."""

import json
import os
from collections.abc import Generator
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from ruamel.yaml import YAML
from zcu_tools.format_version import FormatError, VersionError, YamlMap
from zcu_tools.resources.document_store import ConflictError
from zcu_tools.resources.entry import (
    ComponentSchema,
    MissingReferenceError,
    PointView,
    Provenance,
    ResultEntry,
    SetupView,
    UnitSpec,
    UnknownFieldError,
    component_registry,
)


class Timing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: Annotated[float, UnitSpec("µs")]


class NotebookComponent(ComponentSchema):
    timing: Timing


@pytest.fixture
def notebook_kind(registry_state_guard: None) -> Generator[str]:
    kind = "test/provenance-notebook"
    component_registry.register(kind, NotebookComponent)
    try:
        yield kind
    finally:
        component_registry.unregister(kind)


def assert_metadata(actual: Provenance | None, expected: Provenance) -> None:
    assert actual is not None
    fields = asdict(expected)
    if expected.stderr is not None:
        fields["stderr"] = pytest.approx(expected.stderr)
    assert asdict(actual) == fields


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


@pytest.mark.parametrize("owner", ["Q1", "general"])
@pytest.mark.parametrize(
    ("payload", "key_path"),
    [
        ({"": 1.0}, ("",)),
        ({"a.b": 1.0}, ("a.b",)),
        ({"nested": {"a.b": 1.0}}, ("nested", "a.b")),
        ({"series": [{"": 1.0}]}, ("series", 0, "")),
    ],
)
def test_extension_key_syntax_rejection_preserves_draft_and_source(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    owner: str,
    payload: YamlMap,
    key_path: tuple[str | int, ...],
) -> None:
    _entry, view, source = container
    field = f"{owner}.ext.accepted.rate"
    expected = Provenance("evt-1", "fit", "run-1", "2026-10-04T00:00:00Z", 0.2)
    with view.edit() as draft:
        draft.set(f"{owner}.ext.accepted", {"rate": 12.0})
        draft.set(field, 12.0, provenance=expected)
    before = source.read_bytes()
    with view.edit() as draft:
        with pytest.raises(ValidationError) as error:
            draft.set(f"{owner}.ext", payload)
        assert any(
            detail["loc"][-len(key_path) :] == key_path
            for detail in error.value.errors()
        )
        extensions = draft.general.ext if owner == "general" else draft.Q1.ext
        assert extensions.accepted == {"rate": 12.0}
    extensions = view.general.ext if owner == "general" else view.Q1.ext
    assert extensions.accepted == {"rate": 12.0}
    assert_metadata(view.meta(field), expected)
    assert source.read_bytes() == before


@pytest.mark.parametrize("key", ["", "a.b"])
@pytest.mark.parametrize("owner", ["Q1", "general"])
def test_extension_key_syntax_rejection_preserves_item_write(
    container: tuple[ResultEntry, SetupView | PointView, Path], key: str, owner: str
) -> None:
    _entry, view, source = container
    before = source.read_bytes()
    previous = view.meta("Q1.rate")
    extensions = view.general.ext if owner == "general" else view.Q1.ext
    with pytest.raises(ValidationError):
        extensions[key] = 12.0
    with view.edit() as draft:
        extensions = draft.general.ext if owner == "general" else draft.Q1.ext
        with pytest.raises(ValidationError):
            extensions.batch = [{key: 12.0}]
        with pytest.raises(AttributeError):
            _ = extensions.batch
    with pytest.raises(KeyError):
        _ = extensions[key]
    assert view.meta("Q1.rate") == previous
    assert source.read_bytes() == before


@pytest.mark.parametrize("key", ["", "a.b"])
def test_extension_key_syntax_rejects_add_without_publishing(
    container: tuple[ResultEntry, SetupView | PointView, Path], key: str
) -> None:
    _entry, view, source = container
    before = source.read_bytes()
    with pytest.raises(ValidationError):
        view.add_component("Q2", kind="fake/drive/a", ext={"batch": [{key: 12.0}]})
    with pytest.raises(AttributeError):
        _ = view.Q2
    assert view.Q1.rate == 5000.0
    assert source.read_bytes() == before


@pytest.mark.parametrize("owner", ["Q1", "general"])
@pytest.mark.parametrize("key", ["", "a.b"])
def test_extension_key_syntax_rejects_open_and_refresh_without_publishing(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    tmp_path: Path,
    owner: str,
    key: str,
) -> None:
    _entry, view, source = container
    previous = view.meta("Q1.rate")
    yaml = YAML(typ="safe")
    document = yaml.load(source.read_text())
    target = (
        document["general"] if owner == "general" else document["components"][owner]
    )
    target["ext"] = {"batch": [{key: 12.0}]}
    with source.open("w") as stream:
        yaml.dump(document, stream)
    before = source.read_bytes()
    with pytest.raises(ValidationError) as error:
        view.refresh()
    assert any(
        detail["loc"][-3:] == ("batch", 0, key) for detail in error.value.errors()
    )
    assert view.Q1.rate == 5000.0
    assert view.meta("Q1.rate") == previous
    if isinstance(view, PointView):
        reopened = ResultEntry.open(
            "entry", result_root=tmp_path / "results", database_root=tmp_path / "db"
        )
        with pytest.raises(ValidationError):
            reopened.use_point("a")
    else:
        with pytest.raises(ValidationError):
            ResultEntry.open(
                "entry", result_root=tmp_path / "results", database_root=tmp_path / "db"
            )
    assert source.read_bytes() == before


@pytest.mark.parametrize("owner", ["Q1", "general"])
def test_legal_extension_keys_preserve_seed_clone_reload_and_sources(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    tmp_path: Path,
    owner: str,
) -> None:
    entry, _view, _source = container
    payload: YamlMap = {"_arbitrary-key": {"flags": [True, None], "radius": 12.0}}
    entry.setup.Q1.ext["fit notes"] = payload
    original = entry.new_point("original")
    assert original.Q1.ext["fit notes"] == payload
    extensions = original.general.ext if owner == "general" else original.Q1.ext
    extensions["fit notes"] = payload
    field = f"{owner}.ext.fit notes._arbitrary-key.radius"
    expected = Provenance("evt-1", "fit", "run-1", "2026-10-04T00:00:00Z", 0.18)
    with original.edit() as draft:
        draft.set(field, 12.0, provenance=expected)
    payload["_arbitrary-key"] = {"radius": 99.0}
    assert extensions["fit notes"] != payload
    clone = entry.new_point("clone", clone_from=original)
    metadata = clone.meta(field)
    assert metadata is not None
    assert metadata.source == expected.source
    assert metadata.stderr == expected.stderr
    assert metadata.cloned_from == {"entry_id": entry.entry_id, "point": "original"}
    reopened = ResultEntry.open(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "db"
    )
    restored = reopened.use_point("clone")
    assert restored.meta(field) == metadata
    with restored.edit() as draft:
        draft.set(field, 12.0, provenance=expected)
    assert_metadata(restored.meta(field), expected)
    assert_metadata(original.meta(field), expected)


@pytest.mark.parametrize("write", ["attribute", "draft_attribute", "set"])
def test_manual_write_records_source_and_value_together(
    container: tuple[ResultEntry, SetupView | PointView, Path], write: str
) -> None:
    _entry, view, source = container
    assert view.meta("Q1.duration") is None
    before = datetime.now(timezone.utc)
    if write == "attribute":
        view.Q1.duration = 12.0
    else:
        with view.edit() as draft:
            if write == "draft_attribute":
                draft.Q1.duration = 12.0
            else:
                draft.set("Q1.duration", 12.0)
            assert view.meta("Q1.duration") is None
    after = datetime.now(timezone.utc)
    metadata = view.meta("Q1.duration")
    assert metadata is not None
    assert asdict(metadata) == {
        "source": "manual",
        "kind": None,
        "run_id": None,
        "at": metadata.at,
        "stderr": None,
        "cloned_from": None,
    }
    assert before <= datetime.fromisoformat(metadata.at) <= after
    stored = YAML(typ="safe").load(source.read_text())
    assert stored["components"]["Q1"]["duration"] == pytest.approx(12.0)
    assert stored["provenance"]["Q1.duration"] == {
        key: value for key, value in asdict(metadata).items() if key != "cloned_from"
    }
    view.refresh()
    assert view.meta("Q1.duration") == metadata


@pytest.fixture
def ledger(container: tuple[ResultEntry, SetupView | PointView, Path]) -> Path:
    entry, view, source = container
    root = source.parents[2] if isinstance(view, PointView) else source.parent
    path = root / "records/ledger.jsonl"
    event = {
        "format": "zcu.ledger",
        "format_version": "1.0",
        "id": "evt-1",
        "entry_id": entry.entry_id,
        "type": "accepted",
        "at": "2026-10-04T00:00:00Z",
    }
    path.write_text(json.dumps(event) + "\n")
    return path


@pytest.mark.parametrize(
    ("path", "value", "stderr", "stored_value", "stored_stderr"),
    [
        ("Q1.duration", 20.0, 0.18, 20.0, 0.18),
        ("Q1.rate", 5100.0, 0.2, 5100.0, 0.2),
        ("Q1.wiring.delay", 1.2, 0.05, 1.2, 0.05),
        ("Q1.ext.noise", 11.0, 2.0, 11.0, 2.0),
    ],
)
def test_explicit_source_and_stderr_round_trip_in_the_values_document(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    path: str,
    value: float,
    stderr: float,
    stored_value: float,
    stored_stderr: float,
) -> None:
    _entry, view, source = container
    before = ledger.read_bytes()
    metadata = Provenance(
        "evt-1", "duration/decay", "run-1", "2026-10-04T00:00:00Z", stderr
    )
    with view.edit() as draft:
        draft.set(path, value, provenance=metadata)
    assert_metadata(view.meta(path), metadata)
    stored = YAML(typ="safe").load(source.read_text())
    assert stored["provenance"][path]["stderr"] == pytest.approx(stored_stderr)
    node = stored["components"]
    for segment in path.split("."):
        node = node[segment]
    assert node == pytest.approx(stored_value)
    view.refresh()
    assert_metadata(view.meta(path), metadata)
    assert ledger.read_bytes() == before


@pytest.mark.parametrize("version", ["1.1", "1.90"])
def test_newer_minor_ledger_event_can_be_referenced_without_rewriting_it(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    version: str,
) -> None:
    _entry, view, _source = container
    event = json.loads(ledger.read_text())
    event["format_version"] = version
    event["future_analysis"] = {"confidence": None, "steps": [{"new": True}]}
    ledger.write_text(json.dumps(event) + "\n")
    before = ledger.read_bytes()
    metadata = Provenance("evt-1", "fit", "run-1", "2026-10-04T00:00:00Z", 0.2)
    with view.edit() as draft:
        draft.set("Q1.rate", 5300.0, provenance=metadata)
    assert view.Q1.rate == 5300.0
    assert_metadata(view.meta("Q1.rate"), metadata)
    view.refresh()
    assert_metadata(view.meta("Q1.rate"), metadata)
    assert ledger.read_bytes() == before


@pytest.mark.parametrize(
    ("write", "path"),
    [
        ("add", "Q2.rate"),
        ("wiring", "Q1.wiring.ch"),
        ("nested_ext", "Q1.ext.batch.rate"),
        ("description", "general.description"),
        ("general_ext", "general.ext.temperature"),
        ("draft_general", "general.description"),
        ("set_general", "general.ext.temperature"),
    ],
)
def test_manual_sources_cover_created_values_and_written_container_leaves(
    container: tuple[ResultEntry, SetupView | PointView, Path], write: str, path: str
) -> None:
    _entry, view, source = container
    before = datetime.now(timezone.utc)
    if write == "add":
        view.add_component("Q2", kind="fake/drive/a", rate=5200.0)
    elif write == "wiring":
        with view.edit() as draft:
            draft.set("Q1.wiring", {"ch": 3, "delay": 1.2})
    elif write == "nested_ext":
        view.Q1.ext.batch = {"rate": 12.0}
    elif write == "description":
        view.description = "accepted point"
    elif write == "general_ext":
        view.general.ext.temperature = 11.0
    else:
        with view.edit() as draft:
            if write == "draft_general":
                draft.general.description = "accepted point"
            else:
                draft.set("general.ext.temperature", 11.0)
    metadata = view.meta(path)
    assert metadata is not None
    assert_metadata(metadata, Provenance("manual", None, None, metadata.at, None))
    assert before <= datetime.fromisoformat(metadata.at) <= datetime.now(timezone.utc)
    stored = YAML(typ="safe").load(source.read_text())
    assert stored["provenance"][path]["source"] == "manual"
    assert stored["provenance"][path]["at"] == metadata.at


@pytest.mark.parametrize("failure", ["missing", "absent_event", "foreign", "malformed"])
def test_invalid_local_source_keeps_the_draft_value_and_source_unchanged(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    failure: str,
) -> None:
    _entry, view, source = container
    before = source.read_bytes()
    previous = view.meta("Q1.rate")
    if failure == "missing":
        ledger.unlink()
    elif failure == "absent_event":
        ledger.write_text(ledger.read_text().replace("evt-1", "evt-other"))
    elif failure == "foreign":
        event = json.loads(ledger.read_text())
        event["entry_id"] = "another-entry"
        ledger.write_text(json.dumps(event) + "\n")
    else:
        ledger.write_text("{invalid-json\n")
    with view.edit() as draft:
        with pytest.raises((ValueError, FileNotFoundError)):
            draft.set(
                "Q1.rate",
                5300.0,
                provenance=Provenance(
                    "evt-1", "fit", "run-1", "2026-10-04T00:00:00Z", 0.2
                ),
            )
        assert draft.Q1.rate == 5000.0
    assert view.Q1.rate == 5000.0
    assert view.meta("Q1.rate") == previous
    assert source.read_bytes() == before


@pytest.mark.parametrize("failure", ["body", "schema", "reference"])
def test_failed_value_transaction_discards_its_formal_source(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    failure: str,
) -> None:
    _entry, view, source = container
    before = source.read_bytes()
    previous = view.meta("Q1.rate")
    error = (
        RuntimeError
        if failure == "body"
        else ValidationError
        if failure == "schema"
        else MissingReferenceError
    )

    def fail_transaction() -> None:
        with view.edit() as draft:
            draft.set(
                "Q1.rate",
                5300.0,
                provenance=Provenance(
                    "evt-1", "fit", "run-1", "2026-10-04T00:00:00Z", 0.2
                ),
            )
            if failure == "body":
                raise RuntimeError("abort")
            if failure == "schema":
                draft.Q1.duration = "not-a-number"
            else:
                draft.Q1.sense = "missing"

    with pytest.raises(error):
        fail_transaction()
    assert view.Q1.rate == 5000.0
    assert view.meta("Q1.rate") == previous
    assert source.read_bytes() == before


def test_notebook_nested_stderr_uses_the_registered_leaf_unit(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    notebook_kind: str,
) -> None:
    _entry, view, source = container
    view.add_component("N1", kind=notebook_kind, timing={"width": 2.0})
    metadata = Provenance("evt-1", "timing/fit", None, "2026-10-04T00:00:00Z", 0.18)
    with view.edit() as draft:
        draft.set("N1.timing.width", 3.0, provenance=metadata)
    assert view.N1.timing == {"width": 3.0}
    assert_metadata(view.meta("N1.timing.width"), metadata)
    stored = YAML(typ="safe").load(source.read_text())
    assert stored["components"]["N1"]["timing"]["width"] == pytest.approx(3.0)
    assert stored["provenance"]["N1.timing.width"]["stderr"] == pytest.approx(0.18)
    view.refresh()
    assert_metadata(view.meta("N1.timing.width"), metadata)


@pytest.mark.parametrize(
    ("field", "raw_value", "error_type"),
    [
        ("format", "zcu.point", FormatError),
        ("format", None, FormatError),
        ("format_version", None, FormatError),
        ("format_version", "invalid", FormatError),
        ("format_version", "2.0", VersionError),
    ],
)
def test_invalid_ledger_header_keeps_the_value_source_and_files_unchanged(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    field: str,
    raw_value: str | None,
    error_type: type[FormatError],
) -> None:
    _entry, view, source = container
    event = json.loads(ledger.read_text())
    if raw_value is None:
        event.pop(field)
    else:
        event[field] = raw_value
    ledger.write_text(json.dumps(event) + "\n")
    ledger_before = ledger.read_bytes()
    before = source.read_bytes()
    previous = view.meta("Q1.rate")
    with view.edit() as draft:
        with pytest.raises(error_type) as error:
            draft.set(
                "Q1.rate",
                5300.0,
                provenance=Provenance(
                    "evt-1", "fit", "run-1", "2026-10-04T00:00:00Z", 0.2
                ),
            )
        assert error.value.source == ledger
        assert error.value.field == field
        assert error.value.actual == raw_value
        assert draft.Q1.rate == 5000.0
    assert view.Q1.rate == 5000.0
    assert view.meta("Q1.rate") == previous
    assert source.read_bytes() == before
    assert ledger.read_bytes() == ledger_before


def test_meta_is_cached_and_missing_values_or_sources_return_none(
    container: tuple[ResultEntry, SetupView | PointView, Path],
) -> None:
    _entry, view, source = container
    view.Q1.duration = 12.0
    expected = view.meta("Q1.duration")
    assert expected is not None
    stored = YAML(typ="safe").load(source.read_text())
    stored["provenance"]["Q1.coherence"] = stored["provenance"]["Q1.duration"]
    stored["provenance"].pop("Q1.rate")
    with source.open("w") as stream:
        YAML(typ="rt").dump(stored, stream)
    view.refresh()
    source.unlink()
    assert view.meta("Q1.duration") == expected
    assert view.meta("Q1.rate") is None
    assert view.meta("Q1.coherence") is None
    assert view.meta("unknown.rate") is None


@pytest.mark.parametrize("field", ["t1err", "t1_err"])
def test_flat_error_fields_are_rejected_without_changing_value_or_source(
    container: tuple[ResultEntry, SetupView | PointView, Path], field: str
) -> None:
    _entry, view, source = container
    before = source.read_bytes()
    with pytest.raises(UnknownFieldError), view.edit() as draft:
        draft.set(f"Q1.{field}", 0.18)
    assert view.Q1.rate == 5000.0
    assert source.read_bytes() == before


def test_conflict_keeps_the_winning_value_and_source_together(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    tmp_path: Path,
) -> None:
    _entry, view, source = container
    other_entry = ResultEntry.open(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "db"
    )
    other = (
        other_entry.use_point("a") if isinstance(view, PointView) else other_entry.setup
    )
    with ExitStack() as transaction:
        draft = transaction.enter_context(view.edit())
        draft.set(
            "Q1.rate",
            5300.0,
            provenance=Provenance("evt-1", "fit", "run-1", "2026-10-04T00:00:00Z", 0.2),
        )
        draft.Q1.duration = 12.0
        other.Q1.rate = 5400.0
        winning_bytes = source.read_bytes()
        with pytest.raises(ConflictError):
            transaction.close()
    assert source.read_bytes() == winning_bytes
    view.refresh()
    assert view.Q1.rate == 5400.0
    metadata = view.meta("Q1.rate")
    assert metadata is not None
    assert metadata.source == "manual"
    assert metadata.stderr is None
    assert view.meta("Q1.duration") is None


@pytest.mark.parametrize("container", ["point"], indirect=True)
def test_clone_sources_are_independent_and_reacceptance_clears_the_origin(
    container: tuple[ResultEntry, SetupView | PointView, Path], ledger: Path
) -> None:
    entry, _view, _source = container
    original = entry.new_point("original")
    expected = Provenance("evt-1", "fit", "run-1", "2026-10-04T00:00:00Z", 0.18)
    with original.edit() as draft:
        draft.set("Q1.duration", 12.0, provenance=expected)
    clone = entry.new_point("clone", clone_from=original)
    metadata = clone.meta("Q1.duration")
    assert metadata is not None
    origin = metadata.cloned_from
    assert origin is not None
    assert origin == {"entry_id": entry.entry_id, "point": "original"}
    origin["point"] = "mutated-return"
    assert clone.meta("Q1.duration") != metadata
    with clone.edit() as draft:
        draft.set("Q1.duration", 12.0, provenance=expected)
    assert_metadata(clone.meta("Q1.duration"), expected)
    assert_metadata(original.meta("Q1.duration"), expected)
    clone.Q1.duration = 12.0
    accepted = clone.meta("Q1.duration")
    assert accepted is not None
    assert_metadata(accepted, Provenance("manual", None, None, accepted.at, None))
    assert_metadata(original.meta("Q1.duration"), expected)


@pytest.mark.parametrize("container", ["setup"], indirect=True)
def test_setup_seed_preserves_formal_source_without_following_later_edits(
    container: tuple[ResultEntry, SetupView | PointView, Path], ledger: Path
) -> None:
    entry, setup, _source = container
    expected = Provenance("evt-1", "fit", "run-1", "2026-10-04T00:00:00Z", 0.18)
    with setup.edit() as draft:
        draft.set("Q1.duration", 12.0, provenance=expected)
    point = entry.new_point("seed")
    assert_metadata(point.meta("Q1.duration"), expected)
    setup.Q1.duration = 15.0
    assert point.Q1.duration == 12.0
    assert_metadata(point.meta("Q1.duration"), expected)
    point.Q1.duration = 13.0
    assert setup.Q1.duration == pytest.approx(15.0)
    assert point.meta("Q1.duration") != setup.meta("Q1.duration")


def test_failed_atomic_replace_preserves_value_source_and_file(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _entry, view, source = container
    before = source.read_bytes()
    previous = view.meta("Q1.rate")

    def reject_replace(_source: str | Path, _destination: str | Path) -> None:
        raise OSError("replace rejected")

    monkeypatch.setattr(os, "replace", reject_replace)
    with pytest.raises(OSError, match="replace rejected"), view.edit() as draft:
        draft.set(
            "Q1.rate",
            5300.0,
            provenance=Provenance("evt-1", "fit", "run-1", "2026-10-04T00:00:00Z", 0.2),
        )
    assert view.Q1.rate == 5000.0
    assert view.meta("Q1.rate") == previous
    assert source.read_bytes() == before


def test_newer_minor_retains_unknown_source_fields_when_accepting_a_value(
    container: tuple[ResultEntry, SetupView | PointView, Path],
) -> None:
    _entry, view, source = container
    stored = YAML(typ="safe").load(source.read_text())
    stored["format_version"] = "1.1"
    stored["provenance"]["Q1.rate"]["future_quality"] = {"method": "future"}
    with source.open("w") as stream:
        YAML(typ="rt").dump(stored, stream)
    view.refresh()
    view.Q1.rate = 5300.0
    accepted = YAML(typ="safe").load(source.read_text())
    assert accepted["format_version"] == "1.1"
    assert accepted["provenance"]["Q1.rate"]["future_quality"] == {"method": "future"}
    metadata = view.meta("Q1.rate")
    assert metadata is not None
    assert metadata.source == "manual"
