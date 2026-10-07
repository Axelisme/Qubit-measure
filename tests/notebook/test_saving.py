"""Real Notebook run/save caller over fake acquisition and production storage."""

from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Self
from unittest.mock import Mock

import pytest
from zcu_tools.datafile import (
    AnalysisDocument,
    SoftwareProvenance,
    load_labber_data,
    load_run_data,
)
from zcu_tools.device import DeviceManager
from zcu_tools.experiment import RunRecord
from zcu_tools.experiment.context import RunContext
from zcu_tools.notebook import NotebookAdapter, NotebookExperiment, PartialSaveError
from zcu_tools.notebook import adapter as adapter_module
from zcu_tools.notebook import saving as saving_module
from zcu_tools.plotting.plots import NonPresentingHost
from zcu_tools.resources.entry import (
    JsonObject,
    LedgerEvent,
    PointView,
    RecordsLedger,
    ResultEntry,
)

from tests._native_support import native_metadata
from tests.experiment._saving_support import SavingCfg, SavingCore, saving_figure


@pytest.fixture(autouse=True)
def captured_software(monkeypatch: pytest.MonkeyPatch) -> None:
    def capture(
        context: RunContext | None, source_directory: Path | None = None
    ) -> SoftwareProvenance:
        return native_metadata().provenance

    monkeypatch.setattr(adapter_module, "capture_software_provenance", capture)


def experiment() -> NotebookExperiment[SavingCore]:
    return NotebookAdapter(
        soc=object(),
        soccfg=object(),
        device_manager=DeviceManager(),
        host=NonPresentingHost(),
    )(SavingCore())


@pytest.fixture
def point(tmp_path: Path) -> PointView:
    entry = ResultEntry.create(
        "entry",
        result_root=tmp_path / "results",
        database_root=tmp_path / "Database",
    )
    return entry.new_point("sweet", description="save-time context")


def analysis_document() -> AnalysisDocument:
    return AnalysisDocument(
        pane="analysis", parameters={"skip": 1}, summary={"frequency": 5000.0}
    )


def test_real_caller_writes_all_formats_and_retains_acquisition(
    point: PointView,
    tmp_path: Path,
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg(gain=2.0))
    acquisition = exp.acquisition_for(source)
    assert not (tmp_path / "Database" / "entry" / "runs").exists()
    saved = exp.save_run(
        point,
        source=source,
        figures={"fit": saving_figure()},
        analysis=(analysis_document(),),
        comment="original comment",
    )
    assert saved.run_id == acquisition.run_id
    assert len(saved.paths) == 5
    native = next(
        item.output.path
        for item in saved.report.outputs
        if item.output.format == "data_h5"
    )
    labber = next(
        item.output.path
        for item in saved.report.outputs
        if item.output.format == "labber"
    )
    loaded = load_run_data(native)
    assert datetime.fromisoformat(loaded.metadata.started_at) == acquisition.started_at
    assert loaded.metadata.finished_at is not None
    assert (
        datetime.fromisoformat(loaded.metadata.finished_at) == acquisition.finished_at
    )
    assert loaded.metadata.completion == acquisition.completion
    assert loaded.metadata.snapshot.point == "sweet"
    assert loaded.metadata.snapshot.description == "save-time context"
    assert loaded.metadata.snapshot.roles == {}
    assert loaded.cfg.values["gain"] == 2.0
    assert loaded.payload.metadata.comment == "original comment"
    assert load_labber_data(str(labber)).z.shape == (3,)
    pngs = [
        item.output.path for item in saved.report.outputs if item.output.format == "png"
    ]
    assert pngs[0].read_bytes() == pngs[1].read_bytes()
    ledger = ResultEntry.open(
        "entry",
        result_root=tmp_path / "results",
        database_root=tmp_path / "Database",
    ).ledger
    assert [event.kind for event in ledger.events()] == ["acquired", "saved"]


def test_resave_rewrites_fixed_outputs_and_uses_new_labber_file(
    point: PointView,
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg(gain=2.0))
    saved = exp.save_run(point, source=source, analysis=(analysis_document(),))
    first = saved.report
    native = next(
        item.output.path for item in first.outputs if item.output.format == "data_h5"
    )
    previous_labber = next(
        item.output.path for item in first.outputs if item.output.format == "labber"
    )
    later_source = exp.run(SavingCfg(gain=9.0))
    assert exp.acquisition_for(later_source).run_id != saved.run_id
    later = saved.save(comment="updated comment")
    assert later.run_id == first.run_id
    assert (
        next(
            item.output.path
            for item in later.outputs
            if item.output.format == "data_h5"
        )
        == native
    )
    assert (
        next(
            item.output.path for item in later.outputs if item.output.format == "labber"
        )
        != previous_labber
    )
    loaded = load_run_data(native)
    assert loaded.cfg.values["gain"] == 2.0
    assert loaded.payload.metadata.comment == "updated comment"


@pytest.mark.parametrize(
    ("offset_hours", "saved_at", "expected_dates"),
    [
        (
            8,
            datetime(2026, 12, 31, 16, 30, tzinfo=UTC),
            (("2027", "01", "Data_0101"), ("2027", "01", "Data_0102")),
        ),
        (
            -7,
            datetime(2027, 1, 1, 1, 30, tzinfo=UTC),
            (("2026", "12", "Data_1231"), ("2027", "01", "Data_0101")),
        ),
        (
            0,
            datetime(2027, 1, 1, 0, 30, tzinfo=UTC),
            (("2027", "01", "Data_0101"), ("2027", "01", "Data_0102")),
        ),
    ],
)
def test_save_and_resave_use_host_local_labber_date(
    point: PointView,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offset_hours: int,
    saved_at: datetime,
    expected_dates: tuple[tuple[str, str, str], tuple[str, str, str]],
) -> None:
    local_zone = timezone(timedelta(hours=offset_hours))
    now = saved_at

    # Isolate clock and host timezone without changing process-global TZ state.
    class HostClock(datetime):
        """Supply fixed instants and emulate this test's local timezone."""

        @classmethod
        def now(cls, tz: tzinfo | None = None) -> Self:
            return cls.fromtimestamp(now.timestamp(), tz)

        def astimezone(self, tz: tzinfo | None = None) -> Self:
            return super().astimezone(local_zone if tz is None else tz)

    monkeypatch.setattr(saving_module, "datetime", HostClock)
    exp = experiment()
    source = exp.run(SavingCfg())
    acquisition = exp.acquisition_for(source)
    saved = exp.save_run(point, source=source)
    first = saved.report
    now += timedelta(days=1)
    second = saved.save()
    reports = (first, second)
    labber_paths: list[Path] = []
    native_paths: list[Path] = []
    for report, expected in zip(reports, expected_dates, strict=True):
        assert report.run_id == acquisition.run_id
        assert all(item.status == "saved" for item in report.outputs)
        labber = next(
            item.output.path
            for item in report.outputs
            if item.output.format == "labber"
        )
        labber_paths.append(labber)
        assert labber.is_file()
        assert (
            labber.relative_to(tmp_path / "Database" / "entry" / "Labber").parts[:3]
            == expected
        )
        native_paths.append(
            next(
                item.output.path
                for item in report.outputs
                if item.output.format == "data_h5"
            )
        )
    assert labber_paths[0] != labber_paths[1]
    assert native_paths[0] == native_paths[1]
    loaded = load_run_data(native_paths[1])
    assert datetime.fromisoformat(loaded.metadata.started_at) == acquisition.started_at
    assert loaded.metadata.finished_at is not None
    assert (
        datetime.fromisoformat(loaded.metadata.finished_at) == acquisition.finished_at
    )
    assert loaded.metadata.labber_path == str(labber_paths[1])
    ledger = ResultEntry.open(
        "entry",
        result_root=tmp_path / "results",
        database_root=tmp_path / "Database",
    ).ledger
    events = ledger.events()
    assert [event.kind for event in events] == ["acquired", "saved", "saved"]
    assert all(
        datetime.fromisoformat(event.at).utcoffset() == timedelta(0) for event in events
    )


def test_partial_error_retains_handle_and_explicit_resave_reuses_directory(
    point: PointView,
    tmp_path: Path,
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    run_id = exp.acquisition_for(source).run_id
    blocker = tmp_path / "Database" / "entry" / "runs" / run_id / "figures"
    blocker.parent.mkdir(parents=True)
    blocker.write_text("block run PNGs")
    with pytest.raises(PartialSaveError, match="outputs failed") as caught:
        exp.save_run(point, source=source, figures={"fit": saving_figure()})
    partial = caught.value
    assert (
        len([item for item in partial.report.outputs if item.status == "failed"]) == 1
    )
    assert partial.run.run_id == run_id
    assert partial.run.report == partial.report
    blocker.unlink()
    rewritten = partial.run.save()
    assert rewritten.run_id == run_id
    assert all(item.status == "saved" for item in rewritten.outputs)
    native = next(
        item.output.path
        for item in rewritten.outputs
        if item.output.format == "data_h5"
    )
    assert native.parent == blocker.parent


@pytest.mark.parametrize("failing_member", ["figure", "analysis"])
def test_encoding_failure_retains_handle_and_saves_other_members(
    point: PointView, monkeypatch: pytest.MonkeyPatch, failing_member: str
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    acquisition = exp.acquisition_for(source)
    figure = saving_figure()
    document = analysis_document()
    if failing_member == "figure":
        monkeypatch.setattr(
            figure, "savefig", Mock(side_effect=ValueError("PNG encoder failed"))
        )
    else:
        document.summary["frequency"] = float("nan")
    with pytest.raises(PartialSaveError, match="outputs failed") as caught:
        exp.save_run(
            point, source=source, figures={"fit": figure}, analysis=(document,)
        )
    partial = caught.value
    assert partial.run.run_id == acquisition.run_id
    assert partial.run.report == partial.report
    failed = [item for item in partial.report.outputs if item.status == "failed"]
    assert len(failed) == (2 if failing_member == "figure" else 1)
    assert all(item.key.member == failing_member and item.error for item in failed)
    assert {"data_h5", "labber"} <= {
        item.output.format for item in partial.report.outputs if item.status == "saved"
    }
    assert all(path.is_file() for path in partial.run.paths)
    rewritten = partial.run.save(
        figures={"fit": saving_figure()}, analysis=(analysis_document(),)
    )
    assert rewritten.run_id == acquisition.run_id
    assert all(item.status == "saved" for item in rewritten.outputs)


def test_failed_new_labber_retains_native_historical_link(point: PointView) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    saved = exp.save_run(point, source=source)
    labber = next(
        item.output.path
        for item in saved.report.outputs
        if item.output.format == "labber"
    )
    # All files in this test's Labber directory belong to this run.
    labber.unlink()
    labber.parent.rmdir()
    labber.parent.write_text("block subsequent Labber output")
    with pytest.raises(PartialSaveError, match="outputs failed") as caught:
        saved.save()
    native = next(
        item.output.path
        for item in caught.value.report.outputs
        if item.output.format == "data_h5"
    )
    assert load_run_data(native).metadata.labber_path == str(labber)
    assert caught.value.run.paths == tuple(
        item.output.path
        for item in caught.value.report.outputs
        if item.status == "saved"
    )


def test_stale_or_unowned_source_fails_before_creating_run_outputs(
    point: PointView,
    tmp_path: Path,
) -> None:
    exp = experiment()
    stale = exp.run(SavingCfg())
    exp.run(SavingCfg(gain=2.0))
    with pytest.raises(ValueError, match="acquisition evidence"):
        exp.save_run(point, source=stale)
    manual = RunRecord(cfg=SavingCfg(), result=stale.result)
    with pytest.raises(ValueError, match="acquisition evidence"):
        exp.save_run(point, source=manual)
    assert not (tmp_path / "Database" / "entry" / "runs").exists()


def test_ledger_append_failure_propagates_after_files_are_written(
    point: PointView,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    original = RecordsLedger.append

    def reject_saved(
        self: RecordsLedger,
        event: LedgerEvent,
        *,
        record: JsonObject | None = None,
    ) -> None:
        if event.kind == "saved":
            raise OSError("ledger unavailable")
        original(self, event, record=record)

    monkeypatch.setattr(RecordsLedger, "append", reject_saved)
    with pytest.raises(OSError, match="ledger unavailable"):
        exp.save_run(point, source=source)
    native = (
        tmp_path
        / "Database"
        / "entry"
        / "runs"
        / exp.acquisition_for(source).run_id
        / "data.h5"
    )
    assert load_run_data(native).metadata.run_id == exp.acquisition_for(source).run_id
