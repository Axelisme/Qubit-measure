"""Real Notebook run/save caller over fake acquisition and production storage."""

from collections.abc import Generator, Mapping
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Literal, Self
from unittest.mock import Mock

import numpy as np
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
from zcu_tools.experiment.stop_signal import ScheduleOutcomeError
from zcu_tools.notebook import NotebookAdapter, NotebookExperiment, PartialSaveError
from zcu_tools.notebook import adapter as adapter_module
from zcu_tools.notebook import saving as saving_module
from zcu_tools.plotting.plots import NonPresentingHost
from zcu_tools.resources.document_store import DocumentStore
from zcu_tools.resources.entry import (
    AcceptedPayload,
    AcquiredPayload,
    JsonObject,
    JsonValue,
    LedgerEvent,
    PointView,
    RecordsLedger,
    ResultEntry,
    RoleSpec,
    capture_point_storage,
    component_registry,
    role_registry,
    snapshot,
    source_path,
)
from zcu_tools.resources.entry.schema import PointDocument

from tests._native_support import native_metadata
from tests.experiment._saving_support import SavingCfg, SavingCore, saving_figure
from tests.notebook import _interrupt_fakes
from tests.notebook._interrupt_fakes import InterruptedCore, InterruptedExecutorCore
from tests.resources.entry.fakes import Driver, registry_state, restore_registry


@pytest.fixture(scope="module", autouse=True)
def accept_registry() -> Generator[None]:
    before = registry_state()
    component_registry.register("notebook/accept-driver", Driver)
    role_registry.register("notebook_target", RoleSpec("notebook/accept-driver"))
    registered = registry_state()
    try:
        yield
        assert registry_state() == registered, (
            "notebook tests polluted shared registries"
        )
    finally:
        restore_registry(before)


@pytest.fixture(autouse=True)
def accept_registry_guard(request: pytest.FixtureRequest) -> Generator[None]:
    before = registry_state()
    yield
    assert registry_state() == before, f"registry polluter: {request.node.nodeid}"


@pytest.fixture
def accept_point(point: PointView) -> PointView:
    point.add_component("D1", kind="notebook/accept-driver", rate=1000.0)
    point.add_component("D2", kind="notebook/accept-driver", rate=2000.0)
    return point


@pytest.fixture
def ledger(point: PointView, tmp_path: Path) -> RecordsLedger:
    return ResultEntry.open(
        "entry",
        result_root=tmp_path / "results",
        database_root=tmp_path / "Database",
    ).ledger


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


def acquired_payload(ledger: RecordsLedger) -> AcquiredPayload:
    """Read the saved run's acquired payload and fail on a missing/wrong event."""
    event = next(event for event in ledger.events() if event.kind == "acquired")
    assert isinstance(event.payload, AcquiredPayload)
    return event.payload


def analysis_document() -> AnalysisDocument:
    return AnalysisDocument(
        pane="analysis", parameters={"skip": 1}, summary={"frequency": 5000.0}
    )


@pytest.mark.parametrize("mode", ["software", "program", "program-round", "executor"])
def test_interrupted_acquisition_preserves_saveable_partial(
    point: PointView, mode: Literal["software", "program", "program-round", "executor"]
) -> None:
    core = InterruptedExecutorCore() if mode == "executor" else InterruptedCore(mode)
    exp = NotebookAdapter(
        soc=object(),
        soccfg=object(),
        device_manager=DeviceManager(),
        host=NonPresentingHost(),
    )(core)
    record = exp.run(SavingCfg())
    expected = np.array([1.0, np.nan, np.nan], dtype=np.complex128)
    assert record is exp.last_run
    assert exp.acquisition_for(record).completion == "stopped"
    np.testing.assert_allclose(record.result.signal, expected, equal_nan=True)
    assert core.context is not None
    assert core.context.cancel_signal.error is not None
    assert core.context.cancel_signal.error.status == "interrupted"
    with pytest.raises(ScheduleOutcomeError):
        core.context.cancel_signal.raise_if_error()
    saved = exp.save_run(point, source=record)
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
    assert loaded.metadata.completion == "stopped"
    restored, _ = core.load_run(native)
    np.testing.assert_allclose(restored.result.signal, expected, equal_nan=True)
    np.testing.assert_allclose(
        load_labber_data(str(labber)).z, expected, equal_nan=True
    )


def test_setup_devices_interrupt_escapes_and_preserves_previous_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = InterruptedCore("setup")
    exp = NotebookAdapter(
        soc=object(),
        soccfg=object(),
        device_manager=DeviceManager(),
        host=NonPresentingHost(),
    )(core)
    previous = exp.run(SavingCfg())
    evidence = exp.acquisition_for(previous)
    interrupt = KeyboardInterrupt("setup_devices stopped")
    monkeypatch.setattr(_interrupt_fakes, "setup_devices", Mock(side_effect=interrupt))
    with pytest.raises(KeyboardInterrupt) as caught:
        exp.run(SavingCfg())
    assert caught.value is interrupt
    assert exp.last_run is previous
    assert exp.acquisition_for(previous) == evidence


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
        roles=None,
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
    assert acquired_payload(ledger).roles == {}


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


def test_save_run_rejects_cross_point_roles_before_outputs_and_events(
    accept_point: PointView,
    ledger: RecordsLedger,
    tmp_path: Path,
) -> None:
    entry = ResultEntry.open(
        "entry",
        result_root=tmp_path / "results",
        database_root=tmp_path / "Database",
    )
    other_point = entry.new_point("other", clone_from=accept_point)
    roles = other_point.resolve(("notebook_target",), notebook_target="D2")
    assert roles.components == {"notebook_target": "D2"}
    exp = experiment()
    source = exp.run(SavingCfg())
    events_before = tuple(ledger.events())

    with pytest.raises(ValueError):
        exp.save_run(accept_point, source=source, roles=roles)

    assert not any(entry.database_path.rglob("*"))
    assert tuple(ledger.events()) == events_before


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


@pytest.mark.parametrize("stderr", [0.0, 0.25, None])
def test_accept_writes_role_values_and_local_accepted_provenance(
    accept_point: PointView, ledger: RecordsLedger, stderr: float | None
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    roles = accept_point.resolve(("notebook_target",), notebook_target="D1")
    saved = exp.save_run(accept_point, source=source, roles=roles)
    native = next(
        item.output.path
        for item in saved.report.outputs
        if item.output.format == "data_h5"
    )
    assert load_run_data(native).metadata.snapshot.roles == {"notebook_target": "D1"}
    assert acquired_payload(ledger).roles == {"notebook_target": "D1"}
    event_id = saved.accept(
        {
            "notebook_target.rate": (1250.0, stderr),
            "general.description": "accepted fit",
        },
        kind="frequency",
        analysis_params={"skip": 2},
    )
    event = ledger.get(event_id).event
    assert event.kind == "accepted"
    assert isinstance(event.payload, AcceptedPayload)
    assert event.payload.run_id == saved.run_id
    assert event.payload.analysis_summary == {"skip": 2}
    assert [(write.path, write.value) for write in event.payload.writes] == [
        ("D1.rate", 1250.0),
        ("general.description", "accepted fit"),
    ]
    accept_point.refresh()
    assert accept_point.D1.rate == 1250.0
    assert accept_point.D2.rate == 2000.0
    assert accept_point.description == "accepted fit"
    metadata = accept_point.meta("D1.rate")
    assert metadata is not None
    assert (metadata.source, metadata.kind, metadata.run_id, metadata.at) == (
        event_id,
        "frequency",
        saved.run_id,
        event.at,
    )
    assert metadata.stderr == stderr
    description_source = accept_point.meta("general.description")
    assert description_source is not None
    assert description_source.source == event_id
    assert description_source.stderr is None


def test_accept_preserves_json_containers_and_unknown_stderr(
    point: PointView, ledger: RecordsLedger
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    saved = exp.save_run(point, source=source)
    event_id = saved.accept(
        {"general.ext": ({"series": [1, 2], "optional": None}, None)}, kind="fit"
    )
    point.refresh()
    assert point.general.ext.series == [1, 2]
    assert point.general.ext.optional is None
    event = ledger.get(event_id).event
    assert isinstance(event.payload, AcceptedPayload)
    assert event.payload.analysis_summary == {}
    assert event.payload.writes[0].value == {"series": [1, 2], "optional": None}
    metadata = point.meta("general.ext.series")
    assert metadata is not None
    assert metadata.source == event_id
    assert metadata.stderr is None


def test_accept_keeps_original_point_roles_and_run_after_later_acquisition(
    accept_point: PointView, tmp_path: Path, ledger: RecordsLedger
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    roles = accept_point.resolve(("notebook_target",), notebook_target="D1")
    saved = exp.save_run(accept_point, source=source, roles=roles)
    entry = ResultEntry.open(
        "entry",
        result_root=tmp_path / "results",
        database_root=tmp_path / "Database",
    )
    other_point = entry.new_point("other", description="untouched")
    next_source = exp.run(SavingCfg(gain=3.0))
    next_saved = exp.save_run(other_point, source=next_source)
    event_id = saved.accept({"notebook_target.rate": 1500.0}, kind="frequency")
    assert accept_point.D1.rate == 1500.0
    assert accept_point.D2.rate == 2000.0
    assert other_point.description == "untouched"
    event = ledger.get(event_id).event
    assert isinstance(event.payload, AcceptedPayload)
    assert event.payload.run_id == saved.run_id
    assert event.payload.run_id != next_saved.run_id


@pytest.mark.parametrize(
    ("values", "kind", "message"),
    [
        ({}, "fit", "values"),
        ({"general.description": "fit"}, "", "kind"),
        ({"general.description": "fit"}, "   ", "kind"),
        ({"general.flux_value": float("nan")}, "fit", "strict JSON"),
        ({"general.flux_value": float("inf")}, "fit", "strict JSON"),
        ({"general.flux_value": (1.0, -0.1)}, "fit", "stderr"),
        ({"general.flux_value": (1.0, float("nan"))}, "fit", "stderr"),
        ({"general.flux_value": (1.0, float("inf"))}, "fit", "stderr"),
    ],
)
def test_accept_rejects_invalid_input_without_point_writes(
    point: PointView,
    ledger: RecordsLedger,
    values: Mapping[str, JsonValue | tuple[JsonValue, float | None]],
    kind: str,
    message: str,
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    saved = exp.save_run(point, source=source)
    before = ledger.events()
    before_point = snapshot(point)
    with pytest.raises(ValueError, match=message):
        saved.accept(values, kind=kind)
    assert ledger.events() == before
    assert point.description == "save-time context"
    assert snapshot(point) == before_point


def test_accept_schema_failure_keeps_event_and_rolls_back_point_edit(
    accept_point: PointView, ledger: RecordsLedger
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    saved = exp.save_run(accept_point, source=source)
    before_source = accept_point.meta("D1.rate")
    before_point = snapshot(accept_point)
    with pytest.raises(ValueError, match="rate"):
        saved.accept(
            {"general.description": "candidate", "D1.rate": "not a number"},
            kind="frequency",
        )
    event = ledger.events()[-1]
    assert event.kind == "accepted"
    assert isinstance(event.payload, AcceptedPayload)
    assert event.payload.run_id == saved.run_id
    assert event.payload.writes[-1].value == "not a number"
    accept_point.refresh()
    assert accept_point.description == "save-time context"
    assert accept_point.D1.rate == 1000.0
    assert accept_point.meta("D1.rate") == before_source
    assert snapshot(accept_point) == before_point


def test_accept_commit_validator_error_propagates_and_keeps_event(
    point: PointView, tmp_path: Path, ledger: RecordsLedger
) -> None:
    failure = RuntimeError("point validation rejected the fit")

    def reject_fit(document: PointDocument) -> None:
        if document.general.description == "candidate":
            raise failure

    path = source_path(point)
    binding = capture_point_storage(point)
    bound = PointView(
        DocumentStore(
            path, PointDocument, format="zcu.parameter-container", validate=reject_fit
        ),
        path,
        ledger=tmp_path / "results" / "entry" / "records" / "ledger.jsonl",
        entry_id=str(binding.entry_id),
        result_root=tmp_path / "results",
        database_root=tmp_path / "Database",
    )
    exp = experiment()
    source = exp.run(SavingCfg())
    saved = exp.save_run(bound, source=source)
    before_point = snapshot(bound)
    with pytest.raises(RuntimeError, match="point validation rejected") as caught:
        saved.accept({"general.description": "candidate"}, kind="fit")
    assert caught.value is failure
    assert ledger.events()[-1].kind == "accepted"
    bound.refresh()
    assert bound.description == "save-time context"
    assert snapshot(bound) == before_point


def test_accept_ledger_error_propagates_without_point_writes(
    point: PointView, ledger: RecordsLedger, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    saved = exp.save_run(point, source=source)
    before = ledger.events()
    before_point = snapshot(point)
    original = RecordsLedger.append
    failure = OSError("accepted ledger unavailable")

    def reject_accepted(
        self: RecordsLedger, event: LedgerEvent, *, record: JsonObject | None = None
    ) -> None:
        if event.kind == "accepted":
            raise failure
        original(self, event, record=record)

    monkeypatch.setattr(RecordsLedger, "append", reject_accepted)
    with pytest.raises(OSError, match="accepted ledger unavailable") as caught:
        saved.accept({"general.description": "candidate"}, kind="fit")
    assert caught.value is failure
    assert ledger.events() == before
    point.refresh()
    assert point.description == "save-time context"
    assert snapshot(point) == before_point


def test_accept_rejects_non_json_analysis_parameters(
    point: PointView, ledger: RecordsLedger
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    saved = exp.save_run(point, source=source)
    before = ledger.events()
    before_point = snapshot(point)
    with pytest.raises(ValueError, match="strict JSON"):
        saved.accept(
            {"general.description": "candidate"},
            kind="fit",
            analysis_params={"skip": float("nan")},
        )
    assert ledger.events() == before
    assert snapshot(point) == before_point
