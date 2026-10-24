"""Project result queries preserve historical facts and restrict file access."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from zcu_tools.datafile import Container, ContainerAttrs, JsonObject, save
from zcu_tools.experiment.metadata import RunMetadata, RunSnapshot, SoftwareProvenance
from zcu_tools.gui.app.measure.services.results_index import scan_results
from zcu_tools.resources.entry import (
    AcquiredPayload,
    LedgerEvent,
    ResultEntry,
    SaveLayout,
)

_RUN1 = "20261023T010000Z-aaaaaa"
_RUN2 = "20261023T020000Z-bbbbbb"
_RUN3 = "20261023T030000Z-cccccc"
_AT = "2026-10-23T01:05:00Z"
_CFG: JsonObject = {
    "values": {"readout": {"tones": [{"label": "Q1"}, {"label": "Q2"}]}},
    "cfg_type": "SyntheticCfg",
    "schema_version": "1.0",
}
_ANALYSIS: JsonObject = {
    "format": "zcu.analysis-result",
    "format_version": "1.0",
    "pane": "analysis",
    "parameters": {},
    "summary": {"Q2": {"frequency": 5100.0}},
}


def _layout(root: Path, entry: str, experiment: str, run_id: str) -> SaveLayout:
    return SaveLayout(
        result_path=root / "result" / entry,
        database_path=root / "Database" / entry,
        run_id=run_id,
        experiment=experiment,
        point=None,
        saved_at=datetime(2026, 10, 23, tzinfo=timezone.utc),
    )


def _native(run: Path, entry: ResultEntry, run_id: str, point: str) -> None:
    metadata = RunMetadata(
        run_id=run_id,
        experiment="mux/readout",
        started_at="2026-10-23T01:00:00Z",
        finished_at=None,
        completion="unknown",
        snapshot=RunSnapshot(
            entry_id=UUID(entry.entry_id), point=point, roles={"qubit": "Q1"}
        ),
        provenance=SoftwareProvenance(
            software_versions={},
            git_commit=None,
            git_dirty=None,
            qick_version=None,
            soc_fingerprint=None,
            hostname=None,
        ),
    )
    save(
        run / "data.h5",
        Container(
            vars={},
            docs={"metadata": metadata.model_dump(mode="json"), "cfg": _CFG},
            groups={},
            attrs=ContainerAttrs("3.0", "synthetic-results", None, {}),
        ),
    )


def _acquired(entry: ResultEntry, run_id: str, point: str) -> None:
    entry.ledger.append(
        LedgerEvent(
            id=str(uuid4()),
            entry_id=entry.entry_id,
            kind="acquired",
            at=_AT,
            origin="notebook",
            payload=AcquiredPayload(
                run_id=run_id,
                experiment="mux/readout",
                point=point,
                roles={"qubit": "Q1"},
            ),
        )
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    for name in ("alpha", "beta"):
        entry = ResultEntry.create(
            name, result_root=tmp_path / "result", database_root=tmp_path / "Database"
        )
        for run_id, point in ((_RUN1, "p1"), (_RUN2, "p2")):
            run = _layout(tmp_path, name, "mux/readout", run_id).run_path
            (run / "figures").mkdir(parents=True)
            (run / "figures" / "readout.png").write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")
            (run / "analysis").mkdir()
            (run / "analysis" / "analysis.json").write_text(
                '{"format":"zcu.analysis-result","format_version":"1.0",'
                '"pane":"analysis","parameters":{},'
                '"summary":{"Q2":{"frequency":5100.0}}}',
                encoding="utf-8",
            )
            _native(run, entry, run_id, point)
            if run_id == _RUN1:
                _acquired(entry, run_id, point)
    orphan = _layout(tmp_path, "orphan", "unknown", _RUN3).run_path
    (orphan / "figures").mkdir(parents=True)
    (orphan / "figures" / "only.png").write_bytes(b"orphan PNG")
    # Browsing must not recreate sidecars left by fixture writes.
    for lock in tmp_path.rglob("*.lock"):
        lock.unlink()
    return tmp_path


def test_old_metadata_and_cfg_are_browsed_without_current_experiment_schema(
    tmp_path: Path,
) -> None:
    cfg: JsonObject = {
        "cfg_type": "DeletedCfg",
        "schema_version": "2.0",
        "values": {"old_key": 123},
    }
    samples: tuple[tuple[str, JsonObject], ...] = (
        (
            "old",
            {
                "experiment": "deleted/experiment",
                "snapshot": {"point": "legacy", "roles": {"qubit": "Q9"}},
                "unknown_future_field": True,
            },
        ),
        ("broken", {"experiment": ["not a string"]}),
        ("missing", {}),
    )
    for name, metadata in samples:
        run = tmp_path / "Database" / "entry" / "runs" / name
        run.mkdir(parents=True)
        save(
            run / "data.h5",
            Container(
                vars={},
                docs={"metadata": metadata, "cfg": cfg},
                groups={},
                attrs=ContainerAttrs("3.0", "synthetic", None, {}),
            ),
        )
        (run / "figures").mkdir()
        (run / "figures" / "saved.png").write_bytes(b"saved raster")
    index = scan_results(tmp_path)
    old = index.get_run("entry/old")
    assert (old.experiment, old.point, old.roles) == (
        "deleted/experiment",
        "legacy",
        {"qubit": "Q9"},
    )
    assert old.at is None and old.run_id is None
    assert index.read_cfg(old.key) == cfg
    assert index.read_metadata(old.key)["unknown_future_field"] is True
    assert not old.issues
    broken = index.get_run("entry/broken")
    assert broken.issues and "experiment" in broken.issues[0]
    assert index.read_figure(broken.key, "saved.png") == b"saved raster"
    assert index.get_run("entry/missing").experiment is None
    assert len(index.runs) == 3


def test_index_spans_entries_points_and_preserves_slash_tag(project: Path) -> None:
    index = scan_results(project)
    assert len(index.runs) == 5
    assert [(run.entry, run.point, run.registered) for run in index.runs] == [
        ("alpha", "p1", True),
        ("alpha", "p2", False),
        ("beta", "p1", True),
        ("beta", "p2", False),
        ("orphan", None, False),
    ]
    first = index.runs[0]
    assert first.run_id == _RUN1
    assert first.experiment == "mux/readout"
    assert first.roles == {"qubit": "Q1"}
    assert first.at == _AT
    assert index.runs[1].at == "2026-10-23T01:00:00Z"
    assert index.runs[-1].run_id is None
    assert index.runs[-1].at is None


def test_indexed_documents_and_search_include_tones_missing_from_roles(
    project: Path,
) -> None:
    index = scan_results(project, entries=("alpha",))
    assert len(index.runs) == 2
    run = index.runs[0]
    assert "q2" in run.search_text
    assert index.get_run(run.key) == run
    assert run.run_path == _layout(project, "alpha", "mux/readout", _RUN1).run_path
    assert run.data_path == run.run_path / "data.h5"
    assert run.figure_paths == (run.run_path / "figures" / "readout.png",)
    assert run.analysis_paths == (run.run_path / "analysis" / "analysis.json",)
    assert index.read_cfg(run.key) == _CFG
    assert index.read_analysis(run.key, "analysis.json") == _ANALYSIS
    assert index.read_figure(run.key, "readout.png") == b"\x89PNG\r\n\x1a\nsynthetic"


def test_figure_only_acquisition_uses_ledger_identity(project: Path) -> None:
    data = _layout(project, "alpha", "mux/readout", _RUN1).run_path / "data.h5"
    data.unlink()
    index = scan_results(project, entries=("alpha",))
    assert len(index.runs) == 2
    run = index.runs[0]
    assert (run.run_id, run.experiment, run.point, run.registered) == (
        _RUN1,
        "mux/readout",
        "p1",
        True,
    )
    assert run.data_path is None
    assert index.read_figure(run.key, "readout.png").startswith(b"\x89PNG")
    with pytest.raises(FileNotFoundError, match="data.h5"):
        index.read_cfg(run.key)


@pytest.mark.parametrize("native_state", ["missing", "unreadable"])
def test_acquired_identity_matches_full_run_id_when_adapter_differs(
    project: Path, native_state: str
) -> None:
    original = _layout(project, "alpha", "mux/readout", _RUN1).run_path
    renamed = _layout(project, "alpha", "saving", _RUN1).run_path
    original.rename(renamed)
    data = renamed / "data.h5"
    if native_state == "missing":
        data.unlink()
    else:
        data.write_bytes(b"broken native data")
    # A partial or merely similar suffix must not inherit the acquisition.
    lookalike = renamed.with_name(f"saving_prefix{_RUN1}")
    lookalike.mkdir()
    index = scan_results(project, entries=("alpha",))
    run = index.get_run(f"alpha/{renamed.name}")
    assert (run.registered, run.run_id, run.experiment, run.point) == (
        True,
        _RUN1,
        "mux/readout",
        "p1",
    )
    assert run.roles == {"qubit": "Q1"}
    assert run.at == _AT
    assert len(run.figure_paths) == 1
    assert bool(run.issues) == (native_state == "unreadable")
    other = index.get_run(f"alpha/{lookalike.name}")
    assert (other.registered, other.run_id, other.experiment, other.at) == (
        False,
        None,
        None,
        None,
    )


def test_broken_native_is_retained_with_diagnostic(project: Path) -> None:
    data = _layout(project, "alpha", "mux/readout", _RUN2).run_path / "data.h5"
    data.write_bytes(b"not hdf5")
    index = scan_results(project, entries=("alpha",))
    assert len(index.runs) == 2
    run = index.runs[1]
    assert run.run_id is None
    assert run.data_path == data
    assert any(str(data) in issue for issue in run.issues)
    assert len(run.figure_paths) == 1


def test_browsing_is_read_only_including_ledgers(project: Path) -> None:
    before = {
        p.relative_to(project): p.read_bytes()
        for p in project.rglob("*")
        if p.is_file()
    }
    index = scan_results(project)
    assert len(index.runs) == 5
    for run in index.runs:
        for figure in run.figure_paths:
            index.read_figure(run.key, figure.name)
        if run.data_path is not None:
            index.read_cfg(run.key)
        for analysis in run.analysis_paths:
            index.read_analysis(run.key, analysis.name)
    after = {
        p.relative_to(project): p.read_bytes()
        for p in project.rglob("*")
        if p.is_file()
    }
    assert after == before


def test_selectors_do_not_open_arbitrary_paths(project: Path) -> None:
    index = scan_results(project)
    assert index.runs, "the index must expose a run to select"
    run = index.runs[0]
    for selector in ("../data.h5", str(run.figure_paths[0]), "absent.png"):
        with pytest.raises(KeyError, match="figure"):
            index.read_figure(run.key, selector)
    with pytest.raises(KeyError, match="analysis"):
        index.read_analysis(run.key, "../../setup.yaml")
    with pytest.raises(KeyError, match="run"):
        index.get_run("../escape")


def test_member_symlink_escape_is_rejected_on_scan_and_read(project: Path) -> None:
    index = scan_results(project)
    assert index.runs, "the index must expose a run to select"
    run = index.runs[0]
    outside = project / "outside.png"
    outside.write_bytes(b"outside")
    figure = run.figure_paths[0]
    figure.unlink()
    figure.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        index.read_figure(run.key, figure.name)
    with pytest.raises(ValueError, match="escapes"):
        scan_results(project)


@pytest.mark.parametrize(
    "ancestor", ["Database", "Database/alpha", "Database/alpha/runs"]
)
def test_reads_reject_replaced_ancestor_roots(project: Path, ancestor: str) -> None:
    index = scan_results(project, entries=("alpha",))
    assert index.runs, "the index must expose a run to select"
    run = index.runs[0]
    source = project / ancestor
    moved = project / "detached"
    source.rename(moved)
    source.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        index.read_figure(run.key, "readout.png")
    with pytest.raises(ValueError, match="escapes"):
        index.read_cfg(run.key)
    with pytest.raises(ValueError, match="escapes"):
        index.read_analysis(run.key, "analysis.json")


def test_entry_selection_rejects_traversal(project: Path) -> None:
    with pytest.raises(ValueError, match="entry"):
        scan_results(project, entries=("../elsewhere",))


def test_corrupt_ledger_is_not_silently_unregistered(project: Path) -> None:
    ledger = project / "result" / "alpha" / "records" / "ledger.jsonl"
    ledger.write_text("incomplete", encoding="utf-8")
    with pytest.raises(ValueError, match="ledger.jsonl"):
        scan_results(project)


def test_missing_database_is_an_empty_read_only_index(tmp_path: Path) -> None:
    assert scan_results(tmp_path).runs == ()
    assert list(tmp_path.iterdir()) == []
