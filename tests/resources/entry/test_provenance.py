"""Value/source transactions through the public entry and view interfaces."""

import json
import os
from collections.abc import Generator
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from ruamel.yaml import YAML
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.document_store import ConflictError
from zcu_tools.resources.entry import (
    AcceptedPayload,
    AcceptedWrite,
    ComponentSchema,
    LedgerEvent,
    PointView,
    Provenance,
    ResultEntry,
    SetupView,
    component_registry,
)
from zcu_tools.resources.entry.views import FieldView

_EVENT_ID = "b13fea4c-1678-4b5c-b7a2-edb975c20611"


class Timing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: float


class NotebookComponent(ComponentSchema):
    model_config = ConfigDict(extra="forbid")
    timing: Timing


@pytest.fixture
def notebook_kind(registry_state_guard: None) -> Generator[str]:
    kind = "test/provenance-notebook"
    component_registry.register(kind, NotebookComponent)
    try:
        yield kind
    finally:
        component_registry.unregister(kind)


def field_view(value: object) -> FieldView:
    """Require an editable container returned by a public view."""
    assert isinstance(value, FieldView)
    return value


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
def test_legal_extension_keys_preserve_seed_clone_reload_and_sources(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    tmp_path: Path,
    owner: str,
) -> None:
    entry, _view, _source = container
    payload: YamlMap = {"_arbitrary-key": {"flags": [True, None], "radius": 12.0}}
    field_view(entry.setup.Q1.ext)["fit notes"] = payload
    original = entry.new_point("original")
    notes = field_view(field_view(original.Q1.ext)["fit notes"])
    data = field_view(notes["_arbitrary-key"])
    assert data.flags == [True, None]
    assert data.radius == 12.0
    extensions = field_view(
        original.general.ext if owner == "general" else original.Q1.ext
    )
    extensions["fit notes"] = payload
    field = f"{owner}.ext.fit notes._arbitrary-key.radius"
    expected = Provenance(_EVENT_ID, "fit", "run-1", "2026-10-04T00:00:00Z", 0.18)
    with original.edit() as draft:
        draft.set(field, 12.0, provenance=expected)
    payload["_arbitrary-key"] = {"radius": 99.0}
    accepted_notes = field_view(extensions["fit notes"])
    accepted_data = field_view(accepted_notes["_arbitrary-key"])
    assert accepted_data.radius == 12.0
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
    event = LedgerEvent(
        id=_EVENT_ID,
        kind="accepted",
        at="2026-10-04T00:00:00Z",
        entry_id=entry.entry_id,
        origin="notebook",
        payload=AcceptedPayload(
            writes=(AcceptedWrite(path="Q1.rate", value=5300.0),),
            analysis_summary={},
        ),
    )
    entry.ledger.append(event)
    return root / "records/ledger.jsonl"


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
        _EVENT_ID, "duration/decay", "run-1", "2026-10-04T00:00:00Z", stderr
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
    metadata = Provenance(_EVENT_ID, "fit", "run-1", "2026-10-04T00:00:00Z", 0.2)
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
        field_view(view.Q1.ext).batch = {"rate": 12.0}
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
        event = json.loads(ledger.read_text())
        event["id"] = str(uuid4())
        ledger.write_text(json.dumps(event) + "\n")
    elif failure == "foreign":
        event = json.loads(ledger.read_text())
        event["entry_id"] = str(uuid4())
        ledger.write_text(json.dumps(event) + "\n")
    else:
        ledger.write_text("{invalid-json\n")
    with view.edit() as draft:
        with pytest.raises((ValueError, FileNotFoundError)):
            draft.set(
                "Q1.rate",
                5300.0,
                provenance=Provenance(
                    _EVENT_ID, "fit", "run-1", "2026-10-04T00:00:00Z", 0.2
                ),
            )
        assert draft.Q1.rate == 5000.0
    assert view.Q1.rate == 5000.0
    assert view.meta("Q1.rate") == previous
    assert source.read_bytes() == before


@pytest.mark.parametrize("failure", ["body", "schema"])
def test_failed_value_transaction_discards_its_formal_source(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    failure: str,
) -> None:
    _entry, view, source = container
    before = source.read_bytes()
    previous = view.meta("Q1.rate")
    error = RuntimeError if failure == "body" else ValidationError

    def fail_transaction() -> None:
        with view.edit() as draft:
            draft.set(
                "Q1.rate",
                5300.0,
                provenance=Provenance(
                    _EVENT_ID, "fit", "run-1", "2026-10-04T00:00:00Z", 0.2
                ),
            )
            if failure == "body":
                raise RuntimeError("abort")
            draft.Q1.duration = "not-a-number"

    with pytest.raises(error):
        fail_transaction()
    assert view.Q1.rate == 5000.0
    assert view.meta("Q1.rate") == previous
    assert source.read_bytes() == before


def test_notebook_nested_value_and_stderr_round_trip_without_scaling(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    notebook_kind: str,
) -> None:
    _entry, view, source = container
    view.add_component("N1", kind=notebook_kind, timing={"width": 2.0})
    metadata = Provenance(_EVENT_ID, "timing/fit", None, "2026-10-04T00:00:00Z", 0.18)
    with view.edit() as draft:
        draft.set("N1.timing.width", 3.0, provenance=metadata)
    timing = view.N1.timing
    assert isinstance(timing, FieldView)
    assert timing.width == 3.0
    assert_metadata(view.meta("N1.timing.width"), metadata)
    stored = YAML(typ="safe").load(source.read_text())
    assert stored["components"]["N1"]["timing"]["width"] == pytest.approx(3.0)
    assert stored["provenance"]["N1.timing.width"]["stderr"] == pytest.approx(0.18)
    view.refresh()
    assert_metadata(view.meta("N1.timing.width"), metadata)


@pytest.mark.parametrize(
    ("field", "raw_value"),
    [
        ("format", "zcu.point"),
        ("format", None),
        ("format_version", None),
        ("format_version", "invalid"),
        ("format_version", "2.0"),
    ],
)
def test_invalid_ledger_header_keeps_the_value_source_and_files_unchanged(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    field: str,
    raw_value: str | None,
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
        with pytest.raises(ValueError, match="ledger") as error:
            draft.set(
                "Q1.rate",
                5300.0,
                provenance=Provenance(
                    _EVENT_ID, "fit", "run-1", "2026-10-04T00:00:00Z", 0.2
                ),
            )
        assert str(ledger) in str(error.value)
        assert field in str(error.value)
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
    with pytest.raises(ValidationError), view.edit() as draft:
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
            provenance=Provenance(
                _EVENT_ID, "fit", "run-1", "2026-10-04T00:00:00Z", 0.2
            ),
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
    expected = Provenance(_EVENT_ID, "fit", "run-1", "2026-10-04T00:00:00Z", 0.18)
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
    expected = Provenance(_EVENT_ID, "fit", "run-1", "2026-10-04T00:00:00Z", 0.18)
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
            provenance=Provenance(
                _EVENT_ID, "fit", "run-1", "2026-10-04T00:00:00Z", 0.2
            ),
        )
    assert view.Q1.rate == 5000.0
    assert view.meta("Q1.rate") == previous
    assert source.read_bytes() == before


@pytest.mark.parametrize("container", ["point"], indirect=True)
def test_accepted_without_run_commits_local_source_and_reopens(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    tmp_path: Path,
) -> None:
    entry, point, _source = container
    assert isinstance(point, PointView)
    accepted = entry.ledger.get(_EVENT_ID).event
    assert isinstance(accepted.payload, AcceptedPayload)
    assert accepted.payload.run_id is None
    assert accepted.payload.source is None
    assert tuple(event.kind for event in entry.ledger.events()) == ("accepted",)
    before = ledger.read_bytes()
    provenance = Provenance(accepted.id, "accepted", None, accepted.at, None)
    with point.edit() as draft:
        draft.set("Q1.rate", 5300.0, provenance=provenance)
    assert point.Q1.rate == 5300.0
    assert_metadata(point.meta("Q1.rate"), provenance)
    reopened = ResultEntry.open(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "db"
    )
    restored = reopened.use_point("a")
    assert restored.Q1.rate == 5300.0
    assert_metadata(restored.meta("Q1.rate"), provenance)
    assert reopened.ledger.get(accepted.id).event == accepted
    assert ledger.read_bytes() == before


@pytest.mark.parametrize("container", ["point"], indirect=True)
@pytest.mark.parametrize("failure", ["replace", "conflict"])
def test_failed_point_commit_leaves_accepted_event_but_no_rejected_source(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
    tmp_path: Path,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, point, source = container
    accepted = entry.ledger.get(_EVENT_ID).event
    before = source.read_bytes()
    original_meta = point.meta("Q1.rate")
    ledger_before = ledger.read_bytes()
    expected_error = OSError if failure == "replace" else ConflictError
    with ExitStack() as transaction:
        draft = transaction.enter_context(point.edit())
        draft.set(
            "Q1.rate",
            5300.0,
            provenance=Provenance(accepted.id, "accepted", None, accepted.at, None),
        )
        if failure == "replace":

            def reject_replace(_source: str | Path, _destination: str | Path) -> None:
                raise OSError("point replace rejected")

            monkeypatch.setattr(os, "replace", reject_replace)
        else:
            other = ResultEntry.open(
                "entry", result_root=tmp_path / "results", database_root=tmp_path / "db"
            ).use_point("a")
            other.Q1.rate = 5400.0
            before = source.read_bytes()
        with pytest.raises(expected_error):
            transaction.close()
    assert source.read_bytes() == before
    assert point.Q1.rate == 5000.0
    assert point.meta("Q1.rate") == original_meta
    point.refresh()
    assert point.Q1.rate == (5000.0 if failure == "replace" else 5400.0)
    committed_meta = point.meta("Q1.rate")
    assert committed_meta is not None
    assert committed_meta.source == "manual"
    assert entry.ledger.get(accepted.id).event == accepted
    assert ledger.read_bytes() == ledger_before


def test_unknown_local_source_is_translated_with_original_cause(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
) -> None:
    _entry, point, source = container
    unknown = str(uuid4())
    before = source.read_bytes()
    with point.edit() as draft, pytest.raises(ValueError, match=unknown) as caught:
        draft.set(
            "Q1.rate",
            5300.0,
            provenance=Provenance(
                unknown, "accepted", None, "2026-10-06T02:00:00Z", None
            ),
        )
    assert str(ledger) in str(caught.value)
    assert isinstance(caught.value.__cause__, KeyError)
    assert source.read_bytes() == before
    assert point.Q1.rate == 5000.0


def test_source_with_missing_record_reports_file_and_keeps_values_unpublished(
    container: tuple[ResultEntry, SetupView | PointView, Path],
    ledger: Path,
) -> None:
    entry, point, source = container
    event = LedgerEvent(
        id=str(uuid4()),
        kind="accepted",
        at="2026-10-06T02:00:00Z",
        entry_id=entry.entry_id,
        origin="notebook",
        payload=AcceptedPayload(
            writes=(AcceptedWrite(path="Q1.rate", value=5300.0),),
            analysis_summary={},
        ),
    )
    entry.ledger.append(event, record={"fit": 1})
    missing = ledger.parent / f"{event.id}.json"
    missing.unlink()
    before = source.read_bytes()
    ledger_before = ledger.read_bytes()
    with point.edit() as draft, pytest.raises(FileNotFoundError) as caught:
        draft.set(
            "Q1.rate",
            5300.0,
            provenance=Provenance(event.id, "accepted", None, event.at, None),
        )
    assert str(missing) in str(caught.value)
    assert point.Q1.rate == 5000.0
    assert source.read_bytes() == before
    assert ledger.read_bytes() == ledger_before


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
