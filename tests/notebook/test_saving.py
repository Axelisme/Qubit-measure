"""Real Notebook run/save caller over fake acquisition and production storage."""

from collections.abc import Generator, Mapping
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import IO, Literal, Self
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseReadoutSpec, PulseSpec
from zcu_tools.datafile import (
    AnalysisDocument,
    load_labber_data,
    load_run_data,
)
from zcu_tools.device import DeviceManager, FakeDevice, FakeDeviceInfo
from zcu_tools.experiment import RunRecord
from zcu_tools.experiment.stop_signal import ScheduleOutcomeError
from zcu_tools.format_version import YamlMap
from zcu_tools.notebook import NotebookAdapter, NotebookExperiment, PartialSaveError
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
    writeback,
)
from zcu_tools.resources.entry.schema import PARAMETER_VERSION, PointDocument

from tests.device._snapshot_fakes import SnapshotDevice
from tests.experiment._saving_support import SavingCfg, SavingCore, saving_figure
from tests.experiment._snapshot_support import SnapshotCore
from tests.notebook._interrupt_fakes import InterruptedCore
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


def test_saved_run_keeps_its_device_snapshot_across_later_acquisitions(
    point: PointView,
) -> None:
    manager = DeviceManager()
    flux = SnapshotDevice()
    flux.set_value(0.25)
    flux.events.clear()
    manager.register_device("flux", flux)
    exp = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=manager, host=NonPresentingHost()
    )(SnapshotCore())
    first = exp.run(SavingCfg())
    assert flux.events == ["read", "write"]
    saved = exp.save_run(point, source=first)
    flux.set_value(0.9)
    second = exp.run(SavingCfg())
    second_evidence = exp.acquisition_for(second)
    assert second_evidence.device_snapshot is not None
    assert second_evidence.device_snapshot["flux"]["value"] == 0.9
    flux.read_failure = OSError("Save must not read hardware")
    read_count = flux.events.count("read")
    report = saved.save()
    path = next(
        item.output.path for item in report.outputs if item.output.format == "data_h5"
    )
    restored = load_run_data(path)
    assert restored.metadata.device_snapshot is not None
    assert restored.metadata.device_snapshot["flux"]["value"] == 0.25
    with h5py.File(path, "r") as file:
        dataset = file["device"]
        assert isinstance(dataset, h5py.Dataset)
        assert dataset.shape == ()
        string = h5py.check_string_dtype(dataset.dtype)
        assert string is not None and string.encoding == "utf-8"
    assert flux.events.count("read") == read_count


def test_saved_run_without_connected_devices_writes_empty_snapshot(
    point: PointView,
) -> None:
    exp = experiment()
    record = exp.run(SavingCfg())
    saved = exp.save_run(point, source=record)
    path = next(
        item.output.path
        for item in saved.report.outputs
        if item.output.format == "data_h5"
    )
    assert load_run_data(path).metadata.device_snapshot == {}


def acquired_payload(ledger: RecordsLedger) -> AcquiredPayload:
    """Read the saved run's acquired payload and fail on a missing/wrong event."""
    event = next(event for event in ledger.events() if event.kind == "acquired")
    assert isinstance(event.payload, AcquiredPayload)
    return event.payload


def analysis_document() -> AnalysisDocument:
    return AnalysisDocument(
        pane="analysis", parameters={"skip": 1}, summary={"frequency": 5000.0}
    )


@pytest.mark.parametrize("mode", ["software", "program", "program-round"])
def test_interrupted_acquisition_preserves_saveable_partial(
    point: PointView, mode: Literal["software", "program", "program-round"]
) -> None:
    core = InterruptedCore(mode)
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


def test_device_operation_interrupt_escapes_and_preserves_previous_record() -> None:
    core = InterruptedCore("device")
    manager = DeviceManager()
    exp = NotebookAdapter(
        soc=object(),
        soccfg=object(),
        device_manager=manager,
        host=NonPresentingHost(),
    )(core)
    previous = exp.run(SavingCfg())
    evidence = exp.acquisition_for(previous)
    interrupt = KeyboardInterrupt("Device operation stopped")

    class InterruptingDevice(FakeDevice):
        info_model = FakeDeviceInfo

        def set_value(self, value: float) -> float:
            del value
            raise interrupt

    manager.register_device("sensor", InterruptingDevice())
    with pytest.raises(KeyboardInterrupt) as caught:
        exp.run(SavingCfg(dev={"probe": "sensor"}))
    assert caught.value is interrupt
    assert exp.last_run is previous
    assert exp.acquisition_for(previous) == evidence


def test_preparation_failure_retains_partial_handle_and_saves_files(
    point: PointView, ledger: RecordsLedger
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    source.result.signal = source.result.signal[:1]
    try:
        exp.save_run(
            point,
            source=source,
            figures={"fit": saving_figure()},
            analysis=(analysis_document(),),
        )
    except PartialSaveError as error:
        partial = error
    except (TypeError, ValueError) as error:
        pytest.fail(f"Data preparation aborted independent files: {error}")
    else:
        pytest.fail("Invalid data must be reported as a partial save")
    assert partial.run.run_id == exp.acquisition_for(source).run_id
    assert partial.run.report == partial.report
    data = [item for item in partial.report.outputs if item.key.member == "data"]
    files = [item for item in partial.report.outputs if item.key.member != "data"]
    assert len(data) == 2
    assert all(item.status == "failed" and item.error for item in data)
    assert len(files) == 3
    assert all(item.status == "saved" and item.output.path.is_file() for item in files)
    assert [event.kind for event in ledger.events()] == ["acquired", "saved"]


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
    assert loaded.metadata.started_at is not None
    assert datetime.fromisoformat(loaded.metadata.started_at) == acquisition.started_at
    assert loaded.metadata.finished_at is not None
    assert (
        datetime.fromisoformat(loaded.metadata.finished_at) == acquisition.finished_at
    )
    assert loaded.metadata.completion == acquisition.completion
    assert loaded.metadata.provenance == acquisition.provenance
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
        assert labber.name == f"test_saving_{acquisition.run_id[-6:]}_1.hdf5"
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
    assert loaded.metadata.started_at is not None
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
    blocker = (
        tmp_path / "Database" / "entry" / "runs" / f"test_saving_{run_id}" / "figures"
    )
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


def test_analysis_encoding_failure_retains_handle_and_saves_other_members(
    point: PointView,
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    acquisition = exp.acquisition_for(source)
    figure = saving_figure()
    document = analysis_document()
    document.summary["frequency"] = float("nan")
    with pytest.raises(PartialSaveError, match="outputs failed") as caught:
        exp.save_run(
            point, source=source, figures={"fit": figure}, analysis=(document,)
        )
    partial = caught.value
    assert partial.run.run_id == acquisition.run_id
    assert partial.run.report == partial.report
    failed = [item for item in partial.report.outputs if item.status == "failed"]
    assert len(failed) == 1
    assert all(item.key.member == "analysis" and item.error for item in failed)
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

    with pytest.raises(ValueError, match="same bound point"):
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
        / f"test_saving_{exp.acquisition_for(source).run_id}"
        / "data.h5"
    )
    assert load_run_data(native).metadata.run_id == exp.acquisition_for(source).run_id


@pytest.mark.parametrize(
    "metadata", [{"stderr": 0.25}, {}, {"stderr": -1, "comment": ["caller", None]}]
)
def test_accept_writes_role_values_and_caller_metadata(
    accept_point: PointView, ledger: RecordsLedger, metadata: YamlMap
) -> None:
    with accept_point.edit() as draft:
        draft.set_meta("D1.rate", {"stderr": 9.0, "stale": True})
        draft.set_meta("general.description", {"comment": "kept"})
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
        parameter={
            "notebook_target.rate": (1250.0, metadata),
            "general.description": "accepted fit",
        },
    )
    event = ledger.get(event_id).event
    assert event.kind == "accepted"
    assert isinstance(event.payload, AcceptedPayload)
    assert event.payload.run_id == saved.run_id
    assert [(write.path, write.value) for write in event.payload.writes] == [
        ("D1.rate", 1250.0),
        ("general.description", "accepted fit"),
    ]
    accept_point.refresh()
    assert accept_point.D1.rate == 1250.0
    assert accept_point.D2.rate == 2000.0
    assert accept_point.description == "accepted fit"
    assert accept_point.meta("D1.rate") == metadata
    assert accept_point.meta("general.description") == {"comment": "kept"}


def test_accept_preserves_json_containers_and_existing_metadata(
    point: PointView, ledger: RecordsLedger
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    saved = exp.save_run(point, source=source)
    with point.edit() as draft:
        draft.set_meta("general.ext.series", {"comment": "kept"})
    event_id = saved.accept(
        parameter={"general.ext": {"series": [1, 2], "optional": None}}
    )
    point.refresh()
    assert point.general.ext.series == [1, 2]
    assert point.general.ext.optional is None
    event = ledger.get(event_id).event
    assert isinstance(event.payload, AcceptedPayload)
    assert event.payload.writes[0].value == {"series": [1, 2], "optional": None}
    assert point.meta("general.ext.series") == {"comment": "kept"}


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
    event_id = saved.accept(parameter={"notebook_target.rate": 1500.0})
    assert accept_point.D1.rate == 1500.0
    assert accept_point.D2.rate == 2000.0
    assert other_point.description == "untouched"
    event = ledger.get(event_id).event
    assert isinstance(event.payload, AcceptedPayload)
    assert event.payload.run_id == saved.run_id
    assert event.payload.run_id != next_saved.run_id


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({}, "nonempty"),
        ({"general.ext.flux_value": float("nan")}, "strict JSON"),
        ({"general.ext.flux_value": float("inf")}, "strict JSON"),
        ({"general.ext.flux_value": (1.0, 0.25)}, "dictionary"),
        ({"general.ext.flux_value": (1.0, None)}, "dictionary"),
        ({"general.ext.flux_value": (1.0, {"bad": (1, 2)})}, "bad"),
    ],
)
def test_accept_rejects_invalid_input_without_point_writes(
    point: PointView,
    ledger: RecordsLedger,
    values: Mapping[str, JsonValue | tuple[JsonValue, YamlMap]],
    message: str,
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    saved = exp.save_run(point, source=source)
    before = ledger.events()
    before_point = snapshot(point)
    with pytest.raises(ValueError, match=message):
        saved.accept(parameter=values)
    assert ledger.events() == before
    assert point.description == "save-time context"
    assert snapshot(point) == before_point


def test_accept_schema_failure_keeps_event_and_prior_writes_and_stops(
    accept_point: PointView, ledger: RecordsLedger
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    saved = exp.save_run(accept_point, source=source)
    before_source = accept_point.meta("D1.rate")
    with pytest.raises(writeback.WritebackApplyError) as caught:
        saved.accept(
            parameter={
                "general.description": "candidate",
                "D1.rate": "not a number",
                "D2.rate": 3000.0,
            },
            module={"D1.pulses.pi": pulse_spec()},
        )
    event = ledger.events()[-1]
    assert event.kind == "accepted"
    assert isinstance(event.payload, AcceptedPayload)
    assert event.payload.run_id == saved.run_id
    assert event.payload.writes[1].value == "not a number"
    assert caught.value.event_id == event.id
    assert caught.value.applied == 1
    assert isinstance(caught.value.__cause__, ValueError)
    assert "rate" in str(caught.value.__cause__)
    accept_point.refresh()
    assert accept_point.description == "candidate"
    assert accept_point.D1.rate == 1000.0
    assert accept_point.D2.rate == 2000.0
    assert accept_point.meta("D1.rate") == before_source
    binding = capture_point_storage(accept_point)
    entry = ResultEntry.open(
        binding.entry_name,
        result_root=binding.result_path.parent,
        database_root=binding.database_path.parent,
    )
    assert entry.module_library(binding.point).snapshot().module_keys() == ()


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
            path,
            PointDocument,
            format="zcu.parameter-container",
            supported_version=PARAMETER_VERSION,
            validate=reject_fit,
        ),
        path,
        entry_id=str(binding.entry_id),
        result_root=tmp_path / "results",
        database_root=tmp_path / "Database",
    )
    exp = experiment()
    source = exp.run(SavingCfg())
    saved = exp.save_run(bound, source=source)
    before_point = snapshot(bound)
    with pytest.raises(writeback.WritebackApplyError) as caught:
        saved.accept(parameter={"general.description": "candidate"})
    assert caught.value.__cause__ is failure
    assert caught.value.applied == 0
    assert caught.value.event_id == ledger.events()[-1].id
    assert ledger.events()[-1].kind == "accepted"
    bound.refresh()
    assert bound.description == "save-time context"
    assert snapshot(bound) == before_point


@pytest.mark.parametrize("failure_type", [IsADirectoryError, PermissionError])
def test_accept_ledger_error_propagates_without_point_writes(
    point: PointView, tmp_path: Path, failure_type: type[OSError]
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    saved = exp.save_run(point, source=source)
    before_point = snapshot(point)
    path = tmp_path / "results" / "entry" / "records" / "ledger.jsonl"
    path.unlink()
    path.mkdir()
    failure = failure_type("ledger cannot be opened")
    original_open = Path.open

    def open_path(
        file: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[str] | IO[bytes]:
        if file == path:
            raise failure
        return original_open(file, mode, buffering, encoding, errors, newline)

    # The filesystem reports EISDIR on POSIX and EACCES on Windows.
    with (
        patch.object(Path, "open", open_path),
        pytest.raises(failure_type) as caught,
    ):
        saved.accept(parameter={"general.description": "candidate"})
    assert caught.value is failure
    point.refresh()
    assert point.description == "save-time context"
    assert snapshot(point) == before_point


def pulse_spec(*, gain: float = 0.25) -> PulseSpec:
    return PulseSpec(
        ch=0, nqz=1, freq=100.0, gain=gain, waveform=ConstWaveformCfg(length=0.05)
    )


def test_accept_writes_parameter_module_waveform_with_one_event(
    accept_point: PointView, tmp_path: Path, ledger: RecordsLedger
) -> None:
    exp = experiment()
    source = exp.run(SavingCfg())
    saved = exp.save_run(
        accept_point,
        source=source,
        roles=accept_point.resolve(("notebook_target",), notebook_target="D1"),
    )
    # Later role resolutions and acquisitions cannot retarget the saved handle.
    accept_point.resolve(("notebook_target",), notebook_target="D2")
    exp.run(SavingCfg(gain=2.0))
    pulse = pulse_spec()
    waveform = ConstWaveformCfg(length=0.075)
    before = len(ledger.events())
    event_id = saved.accept(
        parameter={"notebook_target.rate": (1500.0, {"stderr": 0.5})},
        module={"notebook_target.pulses.pi": pulse},
        waveform={"notebook_target.waveforms.base": waveform},
    )
    assert len(ledger.events()) == before + 1
    event = ledger.get(event_id).event
    assert isinstance(event.payload, AcceptedPayload)
    assert [(w.path, w.value) for w in event.payload.writes] == [
        ("D1.rate", 1500.0),
        ("D1.pulses.pi", pulse.to_dict()),
        ("D1.waveforms.base", waveform.to_dict()),
    ]
    assert event.origin == "notebook"
    assert event.payload.run_id == saved.run_id
    assert event.payload.source is None
    entry = ResultEntry.open(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "Database"
    )
    library = entry.module_library("sweet").snapshot()
    assert library.module_spec("D1.pulses.pi") == pulse
    assert library.waveform_spec("D1.waveforms.base") == waveform
    assert accept_point.D1.rate == 1500.0
    for metadata in (
        library.meta("D1.pulses.pi.gain"),
        library.meta("D1.waveforms.base.length"),
    ):
        assert metadata is not None
        assert (metadata.source, metadata.kind, metadata.run_id, metadata.at) == (
            event_id,
            "test/saving",
            saved.run_id,
            event.at,
        )
    module_meta = library.meta("D1.pulses.pi.gain")
    assert module_meta is not None and module_meta.stderr is None
    assert accept_point.meta("D1.rate") == {"stderr": 0.5}


@pytest.mark.parametrize("family", ["module", "waveform"])
def test_accept_library_only_is_nonempty(
    accept_point: PointView, tmp_path: Path, family: str
) -> None:
    exp = experiment()
    saved = exp.save_run(accept_point, source=exp.run(SavingCfg()))
    if family == "module":
        saved.accept(module={"D1.pulses.pi": pulse_spec()})
    else:
        saved.accept(waveform={"D1.waveforms.base": ConstWaveformCfg(length=0.1)})
    entry = ResultEntry.open(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "Database"
    )
    library = entry.module_library("sweet").snapshot()
    if family == "module":
        assert library.module_keys() == ("D1.pulses.pi",)
    else:
        assert library.waveform_keys() == ("D1.waveforms.base",)


def test_accept_library_failure_keeps_prior_commits_and_stops(
    accept_point: PointView, tmp_path: Path, ledger: RecordsLedger
) -> None:
    exp = experiment()
    saved = exp.save_run(accept_point, source=exp.run(SavingCfg()))
    before = len(ledger.events())
    with pytest.raises(writeback.WritebackApplyError) as caught:
        saved.accept(
            parameter={"D1.rate": 1700.0},
            module={
                "D1.pulses.good": pulse_spec(),
                "D1.pulses.bad": PulseReadoutSpec(
                    gen_ch=0,
                    nqz=1,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=1.0),
                    ro_ch=0,
                    ro_length=0.1,
                    freq=100.0,
                ),
                "D1.pulses.later": pulse_spec(gain=0.5),
            },
            waveform={"D1.waveforms.later": ConstWaveformCfg(length=0.1)},
        )
    assert len(ledger.events()) == before + 1
    accept_point.refresh()
    assert accept_point.D1.rate == 1700.0
    entry = ResultEntry.open(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "Database"
    )
    library = entry.module_library("sweet").snapshot()
    assert library.module_keys() == ("D1.pulses.good",)
    assert caught.value.event_id == ledger.events()[-1].id
    assert caught.value.applied == 2
    assert isinstance(caught.value.__cause__, ValueError)
    assert "Wrong module family" in str(caught.value.__cause__)
    assert library.waveform_keys() == ()
    module_meta = library.meta("D1.pulses.good.gain")
    assert module_meta is not None and module_meta.source == ledger.events()[-1].id


def test_accept_rejects_empty_writeback_before_ledger(
    point: PointView, ledger: RecordsLedger
) -> None:
    exp = experiment()
    saved = exp.save_run(point, source=exp.run(SavingCfg()))
    before = ledger.events()
    with pytest.raises(ValueError, match="nonempty"):
        saved.accept()
    with pytest.raises(ValueError, match="nonempty"):
        saved.accept(parameter={}, module={}, waveform={}, samples={})
    assert ledger.events() == before
    assert point.description == "save-time context"
