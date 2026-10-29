"""Point cloning copies parameters, library Specs and aliases without importing events."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from ruamel.yaml import YAML
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.resources.context.library import (
    MODULE_LIBRARY_FORMAT,
    MODULE_LIBRARY_VERSION,
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.document_store import DocumentStore
from zcu_tools.resources.entry import (
    AcceptedPayload,
    AcceptedWrite,
    LedgerEvent,
    PointView,
    ResultEntry,
)
from zcu_tools.resources.entry.schema import (
    PARAMETER_FORMAT,
    PARAMETER_VERSION,
    PointDocument,
)


@pytest.fixture
def foreign_recorded_point(
    entry_roots: tuple[Path, Path],
) -> tuple[ResultEntry, PointView, LedgerEvent]:
    results, database = entry_roots
    foreign = ResultEntry.create("other", result_root=results, database_root=database)
    foreign.setup.add_component("Q1", kind="fake/drive/a", rate=5000.0)
    point = foreign.new_point("source")
    event = LedgerEvent(
        id=str(uuid4()),
        kind="accepted",
        at="2026-10-06T02:00:00Z",
        origin="gui",
        payload=AcceptedPayload(
            writes=(
                AcceptedWrite(path="Q1.rate", value=5300.0),
                AcceptedWrite(path="Q1.duration", value=12.0),
            ),
        ),
    )
    foreign.ledger.append(event, record={"fit": {"samples": [1, True, None, "µ"]}})
    with point.edit() as draft:
        draft.set("Q1.rate", 5300.0)
        draft.set("Q1.duration", 12.0)
        draft.set_meta("Q1.rate", {"source": event.id, "kind": "fit", "stderr": 0.18})
        draft.set_meta("Q1.duration", {"source": "outside/run", "stderr": -1})
        draft.description = "source environment"
    (results / "other/points/source/library.yaml").write_text(
        "format: zcu.module-library\nformat_version: '2.0'\ncomponents: {Q1: {}}\n",
        encoding="utf-8",
    )
    return foreign, point, event


def write_source_library(
    entry: ResultEntry, event: LedgerEvent
) -> ModuleLibraryDocument:
    """Write valid Specs with a legacy library source record."""
    value = ModuleLibraryDocument(
        format="zcu.module-library",
        format_version="2.0",
        components={
            "Q1": ComponentLibrary(
                pulses={
                    "pi": PulseSpec(
                        ch=0,
                        nqz=1,
                        freq=100.0,
                        gain=0.25,
                        waveform="Q1.waveforms.missing",
                    )
                },
                waveforms={"base": ConstWaveformCfg(length=0.05)},
            )
        },
    )
    stored = value.model_dump(mode="json")
    stored["provenance"] = {
        "Q1.pulses.pi.gain": {
            "source": event.id,
            "kind": "rabi",
            "run_id": None,
            "at": event.at,
            "stderr": 0.01,
        }
    }
    (entry.result_path / "points/source/library.yaml").write_text(
        json.dumps(stored), encoding="utf-8"
    )
    return value


@pytest.mark.parametrize("source_mode", ["label", "view"])
@pytest.mark.parametrize("same_entry", [True, False])
def test_clone_copies_values_without_importing_library_events(
    entry: ResultEntry,
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
    source_mode: str,
    same_entry: bool,
) -> None:
    foreign, point, event = foreign_recorded_point
    from zcu_tools.resources.entry.aliases import EntryAliases

    aliases = EntryAliases(aliases={"drive": "Q1", "copy": "$drive"})
    foreign.save_aliases(aliases, point="source")
    source_document = write_source_library(foreign, event)
    source_file = foreign.result_path / "points/source/library.yaml"
    before = source_file.read_bytes()
    destination = foreign if same_entry else entry
    argument = (
        point if source_mode == "view" else ("source" if same_entry else "other/source")
    )
    cloned = destination.new_point("library-copy", mode="clone", clone_from=argument)
    stored = DocumentStore(
        destination.result_path / "points/library-copy/library.yaml",
        ModuleLibraryDocument,
        format=MODULE_LIBRARY_FORMAT,
        supported_version=MODULE_LIBRARY_VERSION,
    ).snapshot()
    library = LibrarySnapshot(stored)
    assert stored.components == source_document.components
    spec = library.module_spec("Q1.pulses.pi")
    assert isinstance(spec, PulseSpec)
    assert spec.waveform == "Q1.waveforms.missing"
    assert cloned.meta("Q1.rate") == point.meta("Q1.rate")
    assert destination.load_aliases(point="library-copy") == aliases
    assert len(destination.ledger.events()) == (1 if same_entry else 0)
    assert source_file.read_bytes() == before


def test_clone_skips_invalid_specs_without_rewriting_source(
    entry: ResultEntry,
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
) -> None:
    foreign, _point, event = foreign_recorded_point
    write_source_library(foreign, event)
    source = foreign.result_path / "points/source/library.yaml"
    document = json.loads(source.read_text())
    document["components"]["Q1"]["pulses"]["pi"]["mask"] = 3
    source.write_text(json.dumps(document))
    original = source.read_bytes()

    entry.new_point("filtered-copy", mode="clone", clone_from="other/source")

    copied = entry.module_library("filtered-copy").snapshot()
    assert copied.module_keys() == ()
    assert copied.waveform_spec("Q1.waveforms.base").length == 0.05
    assert source.read_bytes() == original
    assert list(source.parent.glob("*.bak*")) == []
    assert entry.ledger.events() == ()


def test_library_clone_failure_cleans_destination_and_keeps_source(
    entry: ResultEntry,
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
) -> None:
    foreign, _point, event = foreign_recorded_point
    write_source_library(foreign, event)
    source = foreign.result_path / "points/source/library.yaml"
    document = json.loads(source.read_text())
    document["format_version"] = "1.0"
    source.write_text(json.dumps(document))
    before = source.read_bytes()
    with pytest.raises((ValueError, FileNotFoundError)):
        entry.new_point("failed-library", mode="clone", clone_from="other/source")
    assert not (entry.result_path / "points/failed-library").exists()
    assert entry.list_points() == []
    assert source.read_bytes() == before


@pytest.mark.parametrize("same_entry", [False, True])
def test_clone_preserves_forward_minor_point_metadata(
    entry: ResultEntry,
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
    same_entry: bool,
) -> None:
    foreign, source_point, _event = foreign_recorded_point
    source_file = foreign.result_path / "points/source/parameters.yaml"
    source_store = DocumentStore(
        source_file,
        PointDocument,
        format=PARAMETER_FORMAT,
        supported_version=PARAMETER_VERSION,
    )
    with source_store.edit() as document:
        document.format_version = "2.1"
        document.metadata["Q1.rate"]["future_quality"] = {
            "checked": True,
            "method": "future-validator",
        }
    before = source_file.read_bytes()
    original_metadata = source_store.snapshot().metadata["Q1.rate"]
    destination = foreign if same_entry else entry

    cloned = destination.new_point("future-copy", mode="clone", clone_from=source_point)

    stored = DocumentStore(
        destination.result_path / "points/future-copy/parameters.yaml",
        PointDocument,
        format=PARAMETER_FORMAT,
        supported_version=PARAMETER_VERSION,
    ).snapshot()
    assert stored.format_version == "2.1"
    assert stored.metadata["Q1.rate"] == original_metadata
    assert cloned.meta("Q1.rate") == original_metadata
    assert cloned.Q1.rate == source_point.Q1.rate
    assert source_file.read_bytes() == before


@pytest.mark.parametrize("source_mode", ["label", "view"])
def test_cross_entry_clone_uses_published_values_and_copies_metadata(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
    source_mode: str,
) -> None:
    results, _database = entry_roots
    foreign, stale_point, _event = foreign_recorded_point
    fresh = foreign.use_point("source")
    with fresh.edit() as draft:
        draft.set("Q1.rate", 5400.0)
        draft.set_meta("general.description", {"comment": "fresh only"})
    assert stale_point.Q1.rate == 5300.0
    source_path = results / "other/points/source"
    before = {
        name: (source_path / name).read_bytes()
        for name in ("parameters.yaml", "library.yaml")
    }
    # Clone does not depend on the destination setup.
    (results / "entry/parameters.yaml").write_text("invalid: [", encoding="utf-8")
    source_arg = "other/source" if source_mode == "label" else stale_point
    clone = entry.new_point("target", mode="clone", clone_from=source_arg)
    assert clone.Q1.rate == 5400.0
    assert clone.Q1.duration == 12.0
    assert clone.description == "source environment"
    assert (results / "entry/points/target/library.yaml").read_bytes() == before[
        "library.yaml"
    ]
    for path in ("Q1.rate", "Q1.duration", "general.description"):
        assert clone.meta(path) == fresh.meta(path)
    assert {name: (source_path / name).read_bytes() for name in before} == before
    assert entry.use_point("target").meta("Q1.rate") == fresh.meta("Q1.rate")


@pytest.mark.parametrize("failure", ["point"])
def test_foreign_source_failure_does_not_publish_destination_or_modify_source(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
    failure: str,
) -> None:
    results, _database = entry_roots
    foreign, _point, event = foreign_recorded_point
    write_source_library(foreign, event)
    source = results / "other/points/source/parameters.yaml"
    ledger_path = results / "other/records/ledger.jsonl"
    document = YAML(typ="rt").load(source.read_text())
    document["components"]["Q1"]["rate"] = "invalid"
    with source.open("w", encoding="utf-8") as stream:
        YAML(typ="rt").dump(document, stream)
    before = source.read_bytes()
    ledger_before = ledger_path.read_bytes()
    with pytest.raises((ValueError, FileNotFoundError)):
        entry.new_point("failed", mode="clone", clone_from="other/source")
    assert not (results / "entry/points/failed").exists()
    assert entry.list_points() == []
    assert source.read_bytes() == before
    assert ledger_path.read_bytes() == ledger_before


def test_clone_stale_view_reads_current_contents_at_source_location(
    entry: ResultEntry, entry_roots: tuple[Path, Path], tmp_path: Path
) -> None:
    results, database = entry_roots
    original = ResultEntry.create("other", result_root=results, database_root=database)
    original.setup.add_component("Q1", kind="fake/drive/a", rate=5000.0)
    stale = original.new_point("source")
    original.result_path.rename(tmp_path / "retired")
    replacement = ResultEntry.create(
        "other", result_root=results, database_root=database
    )
    replacement.setup.add_component("Q1", kind="fake/drive/a", rate=6100.0)
    replacement.new_point("source")

    cloned = entry.new_point("current-copy", mode="clone", clone_from=stale)

    assert cloned.Q1.rate == 6100.0
    assert replacement.use_point("source").Q1.rate == 6100.0
    assert entry.ledger.events() == ()


def test_foreign_view_with_other_result_root_is_rejected_without_publishing(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    results, database = entry_roots
    foreign = ResultEntry.create(
        "elsewhere",
        result_root=tmp_path / "other-results",
        database_root=database,
    )
    point = foreign.new_point("source")
    with pytest.raises(ValueError, match="root"):
        entry.new_point("failed", mode="clone", clone_from=point)
    assert not (results / "entry/points/failed").exists()
    assert entry.ledger.events() == ()


@pytest.mark.parametrize("same_entry", [True, False])
@pytest.mark.parametrize("constructor_bound", [True, False])
def test_clone_accepts_point_identity_with_different_database_root(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    tmp_path: Path,
    same_entry: bool,
    constructor_bound: bool,
) -> None:
    results, _database = entry_roots
    source_entry = (
        entry
        if same_entry
        else ResultEntry.create(
            "other", result_root=results, database_root=tmp_path / "source-output"
        )
    )
    source_entry.setup.add_component("Q1", kind="fake/drive/a", rate=5300.0)
    source_entry.new_point("source")
    other_output = tmp_path / "different-output"
    if constructor_bound:
        source = source_entry.result_path / "points/source/parameters.yaml"
        view = PointView(
            DocumentStore(
                source,
                PointDocument,
                format=PARAMETER_FORMAT,
                supported_version=PARAMETER_VERSION,
            ),
            source,
            result_root=results,
            database_root=other_output,
        )
    else:
        (other_output / source_entry.result_path.name).mkdir(parents=True)
        view = ResultEntry.open(
            source_entry.result_path.name,
            result_root=results,
            database_root=other_output,
        ).use_point("source")

    try:
        cloned = entry.new_point("copy", mode="clone", clone_from=view)
    except ValueError as error:
        pytest.fail(f"Database root is not a Point clone identity boundary: {error}")

    assert cloned.Q1.rate == 5300.0
    assert entry.use_point("copy").Q1.rate == 5300.0
    cloned.Q1.rate = 5400.0
    assert source_entry.use_point("source").Q1.rate == 5300.0
    assert entry.ledger.events() == ()


def test_existing_cross_entry_destination_is_not_merged(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
) -> None:
    results, _database = entry_roots
    entry.new_point("target")
    source = results / "entry/points/target/parameters.yaml"
    before = source.read_bytes()
    with pytest.raises(FileExistsError):
        entry.new_point("target", mode="clone", clone_from="other/source")
    assert source.read_bytes() == before
    assert entry.ledger.events() == ()


@pytest.mark.parametrize("same_entry", [True, False])
def test_constructor_bound_view_clones_without_later_identity_registration(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
    same_entry: bool,
) -> None:
    foreign, view, _event = foreign_recorded_point
    results, database = entry_roots
    source = results / "other/points/source/parameters.yaml"
    store = DocumentStore(
        source,
        PointDocument,
        format=PARAMETER_FORMAT,
        supported_version=PARAMETER_VERSION,
    )
    manual = PointView(
        store,
        source,
        result_root=results,
        database_root=database,
    )
    destination = foreign if same_entry else entry
    cloned = destination.new_point("manual-view-copy", mode="clone", clone_from=manual)
    assert cloned.Q1.rate == 5300.0
    assert len(foreign.ledger.events()) == 1
    for path in ("Q1.rate", "Q1.duration"):
        assert cloned.meta(path) == view.meta(path)
