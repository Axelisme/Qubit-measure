from dataclasses import dataclass
from typing import ClassVar

import pytest
from pydantic import TypeAdapter
from qtpy.QtCore import QMetaObject, Qt
from qtpy.QtWidgets import QFileDialog, QPushButton, QTabWidget
from zcu_tools.datafile import Container
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.metadata import CfgSnapshot
from zcu_tools.experiment.native_persistence import RestoredRun, metadata_from_container
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    SessionEnv,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.services.load import LoadDataError
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.mcp.measure.session import GuiRpcError

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.measure._load_support import write_native_file
from tests.gui.app.measure._reload_fakes import Loader, OldAdapter
from tests.gui.app.measure.remote._helpers import (
    call,
    mcp_client,
    open_client,
)
from tests.gui.app.measure.ui._qt_interactions import modal_action


@dataclass
class Result:
    data_path: str


class RuntimeCfg(ExpCfgModel):
    """The experiment cfg this adapter runs and stores with its results."""

    knob: int = 7


class LoadAdapter(OldAdapter):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        load_data=True, analysis=AnalysisMode.NONE
    )

    @staticmethod
    def make_form(ctx: FormContext) -> Form[RuntimeCfg]:
        del ctx
        return Form(RuntimeCfg, {})

    experiment_tag: ClassVar[str | None] = "test/load"

    def restore(self, stored: Container) -> RestoredRun[RunRecord[RuntimeCfg, Result]]:
        cfg = (
            None
            if TypeAdapter(CfgSnapshot)
            .validate_python(stored.docs["cfg"])
            .values["gain"]
            == 2.0
            else RuntimeCfg(knob=42)
        )
        return RestoredRun(
            RunRecord(
                cfg=cfg,
                result=Result("native"),
                metadata=metadata_from_container(stored),
            ),
            "cfg unavailable" if cfg is None else None,
        )


@pytest.fixture
def app(qapp, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_native_file(tmp_path / "result.hdf5")
    write_native_file(tmp_path / "missing.hdf5", gain=2.0)
    write_native_file(tmp_path / "bad.hdf5")
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    registry = Registry()
    registry.register("demo", LoadAdapter)
    ctrl = Controller(
        state,
        registry,
        None,
        BaseEventBus(),
        catalog_loader=Loader(),
        project_root=str(tmp_path),
    )
    from zcu_tools.gui.session.context_types import (
        CommandOrigin,
        EntryRequest,
        PointCreateRequest,
    )

    ctrl.setup_control.new_entry(EntryRequest("lab"))
    ctrl.setup_control.new_point(
        PointCreateRequest("working", "", CommandOrigin("gui"))
    )
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    ctrl.add_view(window)
    window.show()
    tab_id = ctrl.new_tab("demo")
    try:
        yield ctrl, window, state, tab_id
    finally:
        ctrl.close()
        window.deleteLater()
        qapp.processEvents()


def test_mcp_tab_open_from_file_loads_the_result_into_a_new_active_tab(
    app, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZCU_MCP_CALL_LOG", "0")
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    ctrl, window, state, previous = app
    remote = RemoteControlAdapter(
        controller=ctrl,
        opts=ControlOptions(port=0),
        owner_scheduler=QtOwnerScheduler(),
        render_view=window,
    )
    port = remote.start()
    bridge, invoke = mcp_client(port, tmp_path)
    try:
        invoke("connect", {"port": port})
        invoke("rpc_call", {"method": "context.snapshot"})
        tab = invoke(
            "rpc_call",
            {
                "method": "tab.open_file",
                "params": {"adapter_name": "demo", "data_path": "result.hdf5"},
            },
        )["tab_id"]
        assert tab != previous
        assert state.active_tab_id == tab
        summary = invoke(
            "rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab}}
        )["tabs"][0]
        assert summary["interaction"]["has_run_result"] is True
        assert state.active_tab_id == tab
    finally:
        bridge.disconnect()
        remote.stop()


def test_failed_mcp_load_restores_non_neighbor_visible_tab(
    app, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZCU_MCP_CALL_LOG", "0")
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    ctrl, window, state, focused = app
    middle = ctrl.new_tab("demo")
    neighbor = ctrl.new_tab("demo")
    old_tabs = tuple(ctrl.list_tab_ids())
    assert old_tabs == (focused, middle, neighbor)
    tabs = next(
        widget
        for widget in window.findChildren(QTabWidget)
        if widget.count() == 3 and isinstance(widget.widget(0), ExpTabWidget)
    )
    tabs.setCurrentIndex(0)
    assert state.active_tab_id == focused
    assert window.get_view_snapshot()["active_tab_id"] == focused

    def reject_load(
        self: LoadAdapter, stored: Container
    ) -> RestoredRun[RunRecord[RuntimeCfg, Result]]:
        raise LoadDataError("bad data", reason_code="invalid_data_file")

    monkeypatch.setattr(LoadAdapter, "restore", reject_load)
    remote = RemoteControlAdapter(
        controller=ctrl,
        opts=ControlOptions(port=0),
        owner_scheduler=QtOwnerScheduler(),
        render_view=window,
    )
    port = remote.start()
    bridge, invoke = mcp_client(port, tmp_path)
    try:
        invoke("connect", {"port": port})
        invoke("rpc_call", {"method": "context.snapshot"})
        with pytest.raises(GuiRpcError, match="bad data"):
            invoke(
                "rpc_call",
                {
                    "method": "tab.open_file",
                    "params": {"adapter_name": "demo", "data_path": "bad.hdf5"},
                },
            )
        assert tuple(ctrl.list_tab_ids()) == old_tabs
        assert state.active_tab_id == focused
        with open_client(port) as sock:
            snapshot = call(sock, "view.snapshot", {})
            assert snapshot["ok"], snapshot
            assert snapshot["result"]["active_tab_id"] == focused
        assert tabs.currentIndex() == 0
    finally:
        bridge.disconnect()
        remote.stop()


@pytest.mark.parametrize("path", ["result.hdf5", "missing.hdf5"])
def test_qt_load_status_reports_the_loaded_path(app, qapp, tmp_path, path):
    """Load shows results only, so the status names the file and nothing else."""
    _, window, _, _ = app
    selected = tmp_path / path
    if not selected.exists():
        write_native_file(selected, gain=2.0 if path == "missing.hdf5" else 1.0)
    expected = f"Loaded data from {selected}"
    tab = window.findChild(ExpTabWidget)
    assert tab is not None
    load_button = next(
        button
        for button in tab.findChildren(QPushButton)
        if button.text() == "Load Data"
    )
    assert load_button.isEnabled()

    from zcu_tools.gui.app.measure.ui.run_load_dialog import RunLoadDialog

    def open_file(dialog: RunLoadDialog) -> None:
        file_button = dialog.findChild(QPushButton, "load_file")
        assert file_button is not None
        file_button.click()

    def choose_file(dialog: QFileDialog) -> None:
        dialog.selectFile(str(selected))
        QMetaObject.invokeMethod(dialog, "accept", Qt.ConnectionType.DirectConnection)

    modal_action(
        qapp,
        load_button.click,
        [(RunLoadDialog, open_file), (QFileDialog, choose_file)],
    )
    assert window.get_view_snapshot()["status"] == expected
