"""Run record/metadata publication and fixed connected-device snapshots."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import h5py
import numpy as np
import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import RunSnapshot, SoftwareProvenance, load_run_data
from zcu_tools.device import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment import ExpCfgModel
from zcu_tools.experiment.acquisition import new_run_id
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    RunRequest,
)
from zcu_tools.gui.app.measure.services.guard import RunInputs
from zcu_tools.gui.app.measure.services.operation_gate import (
    OperationKind,
)
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.state import DeviceState, DeviceStatus

from tests.device._snapshot_fakes import SnapshotDevice
from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui.app.measure._cfg_fakes import parameter_capture
from tests.gui.app.measure._form_fakes import make_form
from tests.gui.app.measure._native_save_support import (
    SavingAdapter,
    native_save_session,
)
from tests.gui.app.measure._synthetic import SampleResult
from tests.gui.app.measure.services._snapshot_adapter import SnapshotAdapter

from ._run_support import (
    make_inputs,
    make_plots,
    make_run_service,
    make_state,
)


def _run_snapshot() -> RunSnapshot:
    return RunSnapshot(
        entry_id=UUID("00000000-0000-0000-0000-000000000001"),
        entry_name="original-entry",
        point="original-point",
        description="original description",
        roles={"qubit": "Q1", "resonator": "R1"},
    )


def _provenance() -> SoftwareProvenance:
    return SoftwareProvenance(
        software_versions={"python": "3.13"},
        git_commit="a" * 40,
        git_dirty=False,
        qick_version=None,
        soc_fingerprint="cached-config",
        hostname="test-host",
    )


@pytest.mark.parametrize("stopped", [False, True])
def test_run_publishes_record_and_historical_metadata_together(
    monkeypatch: pytest.MonkeyPatch, stopped: bool
) -> None:
    state, tab_id, adapter = make_state()
    record = RunRecord(cfg=ExpCfgModel(), result=SampleResult(np.array([1.0])))
    adapter.run.return_value = record
    provenance = _provenance()
    capture = MagicMock(return_value=provenance)
    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.run.capture_software_provenance", capture
    )
    snapshot = _run_snapshot()
    inputs = replace(
        make_inputs(state, tab_id, adapter),
        snapshot=snapshot,
        experiment_tag="test/native",
    )
    settings = FakeDeviceInfo(address="offline", value=1.25)
    state.put_device(
        DeviceState(
            name="flux",
            type_name="FakeDevice",
            address="offline",
            status=DeviceStatus.CONNECTED,
            remember=False,
            info=settings,
        )
    )
    identity = MagicMock(wraps=new_run_id)
    monkeypatch.setattr("zcu_tools.gui.app.measure.services.run.new_run_id", identity)
    driver = FakeDevice()
    driver.set_value(1.25)
    service, _gate, background, handles = make_run_service(
        state, devices={"flux": driver}
    )
    before_run = datetime.now(UTC)
    token = service.start_run(inputs, plots=make_plots())
    after_admission = datetime.now(UTC)
    assert identity.called  # Identity is already fixed before acquisition starts.
    settings.value = 9.5
    state.set_context(
        replace(state.session_env, parameters=parameter_capture(gain=0.8))
    )
    if stopped:
        assert service.cancel_run()
    background.run_work()
    after_acquire = datetime.now(UTC)
    pane = state.get_tab(tab_id).run
    assert pane.result is record
    assert pane.source_operation_id == token
    metadata = pane.metadata
    assert metadata is not None
    assert metadata.snapshot == snapshot
    assert metadata.experiment == "test/native"
    assert metadata.provenance == provenance
    assert metadata.completion == ("stopped" if stopped else "complete")
    assert metadata.labber_path is None
    assert metadata.finished_at is not None
    assert metadata.device_snapshot == {
        "flux": FakeDeviceInfo(address="none", value=1.25).model_dump(mode="json")
    }
    assert metadata.started_at is not None
    assert before_run <= datetime.fromisoformat(metadata.started_at) <= after_admission
    assert (
        datetime.fromisoformat(metadata.started_at)
        <= datetime.fromisoformat(metadata.finished_at)
        <= after_acquire
    )
    assert metadata.run_id.startswith(
        datetime.fromisoformat(metadata.started_at).strftime("%Y%m%dT%H%M%SZ-")
    )
    capture.assert_called_once()
    outcome = handles.poll(token)
    assert outcome is not None
    assert outcome.status == ("cancelled" if stopped else "finished")


@pytest.mark.parametrize("provenance_failure", [False, True])
def test_new_run_clears_old_evidence_and_failure_never_publishes_success(
    monkeypatch: pytest.MonkeyPatch, provenance_failure: bool
) -> None:
    state, tab_id, adapter = make_state()
    adapter.run.return_value = RunRecord(
        cfg=ExpCfgModel(), result=SampleResult(np.array([1.0]))
    )
    capture = MagicMock(return_value=_provenance())
    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.run.capture_software_provenance", capture
    )
    inputs = replace(
        make_inputs(state, tab_id, adapter),
        snapshot=_run_snapshot(),
        experiment_tag="test/native",
    )
    service, _gate, background, handles = make_run_service(state)
    service.start_run(inputs, plots=make_plots())
    background.run_work()
    assert state.get_tab(tab_id).run.metadata is not None
    if provenance_failure:
        capture.side_effect = RuntimeError("provenance failed")
    else:
        adapter.run.side_effect = RuntimeError("acquire failed")
    token = service.start_run(inputs, plots=make_plots())
    assert state.get_tab(tab_id).run.result is None
    assert state.get_tab(tab_id).run.metadata is None
    background.run_work()
    assert state.get_tab(tab_id).run.result is None
    assert state.get_tab(tab_id).run.metadata is None
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "failed"


@pytest.mark.parametrize("has_snapshot", [False, True])
def test_run_without_native_declaration_or_saved_origin_does_not_invent_evidence(
    monkeypatch: pytest.MonkeyPatch, has_snapshot: bool
) -> None:
    state, tab_id, adapter = make_state()
    capture = MagicMock(side_effect=RuntimeError("must not capture"))
    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.run.capture_software_provenance", capture
    )
    inputs = replace(
        make_inputs(state, tab_id, adapter),
        snapshot=_run_snapshot() if has_snapshot else None,
        experiment_tag=None if has_snapshot else "test/native",
    )
    service, _gate, background, _handles = make_run_service(state)
    service.start_run(inputs, plots=make_plots())
    background.run_work()
    assert state.get_tab(tab_id).run.metadata is None
    capture.assert_not_called()


def test_gui_run_publishes_the_typed_record_and_declared_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This producer seam substitutes the public cfg assembly/acquisition
    # collaborators. It does not claim the pending assembler/core integration.
    state, tab_id, _mock_adapter = make_state()
    adapter = SavingAdapter()
    cfg = SavingCfg(gain=2.0)
    result = SavingResult(
        frequency=np.array([0.0, 1.0]), signal=np.array([1.0 + 0j, 0.5 + 0j])
    )

    def acquire(
        self: SavingCore, config: SavingCfg, *, context: RunContext
    ) -> SavingResult:
        assert config is cfg
        assert not context.cancel_signal.is_set()
        return result

    monkeypatch.setattr(
        SavingAdapter,
        "run",
        lambda self, req, raw_cfg, *, context: RunRecord(
            cfg, acquire(SavingCore(), cfg, context=context)
        ),
    )
    monkeypatch.setattr(SavingCore, "run", acquire)
    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.run.capture_software_provenance",
        MagicMock(return_value=_provenance()),
    )
    tab = state.get_tab(tab_id)
    tab.adapter = adapter
    inputs = RunInputs(
        tab_id=tab_id,
        adapter_name="test/saving",
        adapter=adapter,
        request=RunRequest(soc=None, soccfg=None, cfg_context=None),
        accepted_cfg=tab.cfg.accept(),
        snapshot=_run_snapshot(),
        experiment_tag=adapter.experiment_tag,
    )
    service, _gate, background, handles = make_run_service(state)
    token = service.start_run(inputs, plots=make_plots())
    background.run_work()
    record = state.get_tab(tab_id).run.result
    metadata = state.get_tab(tab_id).run.metadata
    assert isinstance(record, RunRecord)
    assert isinstance(record.cfg, SavingCfg)
    assert record.cfg == cfg
    assert record.result is result
    assert metadata is not None
    assert metadata.experiment == SavingCore().persistence_spec().tag
    assert metadata.completion == "complete"
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "finished"


def test_cancel_after_acquisition_does_not_relabel_completed_measurement_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, tab_id, adapter = make_state()
    adapter.run.return_value = RunRecord(
        cfg=ExpCfgModel(), result=SampleResult(np.array([1.0]))
    )
    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.run.capture_software_provenance",
        MagicMock(return_value=_provenance()),
    )
    inputs = replace(
        make_inputs(state, tab_id, adapter),
        snapshot=_run_snapshot(),
        experiment_tag="test/native",
    )
    service, _gate, background, handles = make_run_service(state)
    token = service.start_run(inputs, plots=make_plots())
    assert background.last_work is not None
    acquired = background.last_work()
    assert service.cancel_run()
    assert background.last_on_done is not None
    background.last_on_done(acquired)
    metadata = state.get_tab(tab_id).run.metadata
    assert metadata is not None and metadata.completion == "complete"
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "cancelled"


def test_run_reads_all_connected_devices_after_lease_before_core() -> None:
    state, tab_id, _ = make_state()
    adapter = SnapshotAdapter()
    state.get_tab(tab_id).cfg = make_form(
        adapter.make_form(FormContext(state.session_env.parameters)), state=state
    )
    flux, unused = SnapshotDevice(), SnapshotDevice()
    flux.set_value(0.25)
    unused.set_value(0.6)
    flux.events.clear()
    unused.events.clear()
    service, gate, background, handles = make_run_service(
        state, devices={"flux": flux, "unused": unused}
    )

    def require_lease() -> None:
        assert gate.has_active(OperationKind.RUN)

    flux.before_read = unused.before_read = require_lease
    inputs = make_inputs(state, tab_id, adapter)
    inputs = replace(
        inputs,
        request=replace(inputs.request, soc=None, soccfg=None),
        snapshot=_run_snapshot(),
        experiment_tag=adapter.experiment_tag,
    )
    token = service.start_run(inputs, plots=make_plots())
    assert flux.events == unused.events == []
    background.run_work()
    assert flux.events == ["read", "write"]
    assert unused.events == ["read"]
    metadata = state.get_tab(tab_id).run.metadata
    assert metadata is not None
    assert metadata.device_snapshot is not None
    assert metadata.device_snapshot["flux"]["value"] == 0.25
    assert metadata.device_snapshot["unused"]["value"] == 0.6
    flux.set_value(0.9)
    assert metadata.device_snapshot["flux"]["value"] == 0.25
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "finished"


def test_device_read_failure_rejects_run_before_core() -> None:
    state, tab_id, _ = make_state()
    adapter = SnapshotAdapter()
    flux, unused = SnapshotDevice(), SnapshotDevice()
    unused.read_failure = OSError("snapshot read failed")
    service, gate, background, handles = make_run_service(
        state, devices={"flux": flux, "unused": unused}
    )
    token = service.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())
    background.run_work()
    assert state.get_tab(tab_id).run.result is None
    assert flux.events == unused.events == ["read"]
    assert flux.get_value() == 0.0
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "failed"
    assert outcome.error == "snapshot read failed"
    assert not gate.has_active(OperationKind.RUN)


def test_run_and_native_save_keep_admitted_device_alias_after_rebinding(
    tmp_path: Path, qapp: QApplication
) -> None:
    from zcu_tools.gui.session.aliases import AliasRow

    with native_save_session(tmp_path, qapp) as session:
        session.context.set_alias_rows((AliasRow("flux_dev", "yoko1"),))
        captured = session.context.read_context()
        state = session.state
        adapter = SnapshotAdapter()
        state.get_tab("tab").adapter = adapter
        resource = TabFormResource(
            lambda _ctx: Form(SavingCfg, {}),
            capture=lambda _aliases: TabFormSources(
                FormEnv(parameters=captured, library=captured.library)
            ),
            mutation_allowed=lambda: True,
        )
        admitted_flux = captured.value("$flux_dev")
        assert isinstance(admitted_flux, str)
        resource.write({("dev",): {"flux_dev": admitted_flux}})
        state.get_tab("tab").cfg = resource
        service, _gate, background, handles = make_run_service(state, devices={})
        inputs = replace(
            make_inputs(state, "tab", adapter),
            parameters=captured,
            snapshot=replace(
                session.metadata.snapshot, roles=captured.capture_storage().roles
            ),
            experiment_tag=adapter.experiment_tag,
        )
        token = service.start_run(inputs, plots=make_plots())
        session.context.set_alias_rows((AliasRow("flux_dev", "yoko2"),))
        background.run_work()
        outcome = handles.poll(token)
        assert outcome is not None and outcome.status == "finished"
        result = state.get_tab("tab").run.result
        assert isinstance(result, RunRecord)
        assert result.cfg is not None
        assert result.cfg.dev == {"flux_dev": "yoko1"}
        metadata = state.get_tab("tab").run.metadata
        assert metadata is not None and metadata.snapshot.roles["flux_dev"] == "yoko1"
        report = session.save()
        path = next(
            item.output.path
            for item in report.outputs
            if item.output.format == "data_h5"
        )
        restored = load_run_data(path)
        assert restored.metadata.snapshot.roles["flux_dev"] == "yoko1"
        assert restored.cfg.values["dev"] == {"flux_dev": "yoko1"}
        assert session.context.read_context().value("$flux_dev") == "yoko2"


def test_gui_save_uses_fixed_run_device_snapshot(
    tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        state = session.state
        adapter = SnapshotAdapter()
        state.get_tab("tab").adapter = adapter
        flux = SnapshotDevice()
        flux.set_value(0.25)
        flux.events.clear()
        service, _gate, background, handles = make_run_service(
            state, devices={"flux": flux}
        )
        inputs = replace(
            make_inputs(state, "tab", adapter),
            snapshot=session.metadata.snapshot,
            experiment_tag=adapter.experiment_tag,
        )
        token = service.start_run(inputs, plots=make_plots())
        background.run_work()
        outcome = handles.poll(token)
        assert outcome is not None and outcome.status == "finished"
        assert flux.events == ["read", "write"]
        flux.read_failure = OSError("Save must not read hardware")
        report = session.save()
        path = next(
            item.output.path
            for item in report.outputs
            if item.output.format == "data_h5"
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
        assert flux.events == ["read", "write"]
