"""Measure terminal delivery refreshes observations, never acquisition evidence."""

from dataclasses import replace
from pathlib import Path
from threading import get_ident
from uuid import UUID

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.app_services import build_app_services
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form_adapter import FormContext
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.ui.device_dialog import DeviceDialog

from tests.device._snapshot_fakes import SnapshotDevice
from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._form_fakes import make_form
from tests.gui.app.measure._record_support import require_record

from ._run_support import make_inputs, make_plots, make_state
from ._snapshot_adapter import SnapshotAdapter


def readback_adapter(unused: SnapshotDevice, fail_run: bool):
    """Synthetic acquisition leaves one failed observation after successful capture."""

    def run(
        cfg: SavingCfg, *, context: RunContext
    ) -> RunRecord[SavingCfg, SavingResult]:
        result = SnapshotAdapter.run(cfg, context=context)
        unused.set_value(0.9)
        unused.read_failure = OSError("post-run observation failed")
        if fail_run:
            raise OSError("acquisition failed after write")
        return result

    return Adapter(
        spec=SnapshotAdapter.spec,
        make_form=SnapshotAdapter.make_form,
        run=run,
        guide=None,
    )


def _connected_devices() -> tuple[DeviceManager, SnapshotDevice, SnapshotDevice]:
    flux, unused = SnapshotDevice(), SnapshotDevice()
    flux.set_value(0.25)
    unused.set_value(0.6)
    manager = DeviceManager()
    manager.register_device("flux", flux)
    manager.register_device("unused", unused)
    return manager, flux, unused


@pytest.mark.parametrize("fail_run", [False, True], ids=["finished", "failed"])
def test_measure_terminal_reads_back_on_worker_and_commits_on_owner(
    qapp: QApplication, tmp_path: Path, fail_run: bool
) -> None:
    manager, flux, unused = _connected_devices()
    state, tab_id, _ = make_state()
    adapter = readback_adapter(unused, fail_run)
    state.get_tab(tab_id).cfg = make_form(
        adapter.make_form(FormContext(state.session_env.parameters)), state=state
    )
    services = build_app_services(
        state=state,
        bus=BaseEventBus(),
        registry=Registry(),
        progress_transport=DirectProgressTransport(),
        resource_versions=lambda: {},
        render_host=lambda: None,
        owner_scheduler=QtOwnerScheduler(),
        project_root=str(tmp_path),
        device_registry=manager,
    )
    owner = get_ident()
    read_threads: list[int] = []
    commits: list[int] = []
    flux.events.clear()
    unused.events.clear()
    flux.before_read = unused.before_read = lambda: read_threads.append(get_ident())
    services.device_control.on_device_changed(lambda _: commits.append(get_ident()))
    dialog = DeviceDialog(services.device_control, parameter_provider=lambda: None)
    try:
        inputs = make_inputs(state, tab_id, adapter)
        inputs = replace(
            inputs,
            snapshot=RunSnapshot(
                entry_id=UUID(int=1),
                point="point",
                roles={},
            ),
            experiment_tag=adapter.experiment_tag,
        )
        token = services.run.start_run(inputs, plots=make_plots())
        # This submitted, unblocked synthetic acquisition must deliver its terminal.
        qt_until(lambda: services.handles.poll(token) is not None, label="run terminal")
        assert services.background.quiesce(timeout_ms=30_000)
        assert services.device_control.get_cached_device_value("flux") == 0.75
        assert services.device_control.get_cached_device_value("unused") == 0.6
        assert flux.events == unused.events == ["read", "write", "read"]
        assert read_threads and all(thread != owner for thread in read_threads)
        assert commits == [owner]
        outcome = services.handles.poll(token)
        assert outcome is not None and outcome.status == (
            "failed" if fail_run else "finished"
        )
        assert not services.operation_gate.has_active("run")
        if not fail_run:
            metadata = require_record(state.get_tab(tab_id).run).metadata
            assert metadata is not None and metadata.device_snapshot is not None
            assert metadata.device_snapshot["flux"]["value"] == 0.25
            assert metadata.device_snapshot["unused"]["value"] == 0.6
        dialog.show()
        qapp.processEvents()
        ui = ComponentDriver(dialog)
        ui.select_list_row(
            next(row.text for row in ui.list_rows() if "flux (" in row.text)
        )
        assert any(w.text == "0.75" for w in ui.visible().widgets)
        ui.screenshot(tmp_path / f"measure-{fail_run}-readback.png")
        ui.click("Close")
    finally:
        services.background.stop_accepting()
        assert services.background.quiesce(timeout_ms=30_000)
        dialog.close()
        dialog.deleteLater()
        manager.close_all_devices()
