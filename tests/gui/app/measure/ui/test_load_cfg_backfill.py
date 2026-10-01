from dataclasses import dataclass, replace
from typing import ClassVar

import pytest
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QFileDialog, QLineEdit, QPushButton, QTabWidget
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    LoadDataRequest,
)
from zcu_tools.gui.app.measure.app import _make_empty_ctx
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.services.load import LoadDataError
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.cfg import DirectValue
from zcu_tools.gui.cfg.resource import CfgEdit
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.gui.session.types import ContextReadiness
from zcu_tools.gui.widgets.cfg.resource_form import ResourceCfgFormWidget
from zcu_tools.mcp.measure.session import GuiRpcError

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.measure._reload_fakes import Loader, OldAdapter
from tests.gui.app.measure.remote._helpers import (
    call,
    mcp_client,
    open_client,
)


class RuntimeCfg(ExpCfgModel):
    knob: int = 42


@dataclass
class Result:
    data_path: str


class LoadAdapter(OldAdapter):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        load_data=True, analysis=AnalysisMode.NONE
    )

    def load(self, req: LoadDataRequest) -> RunRecord[RuntimeCfg, Result]:
        cfg = None if req.data_path == "missing.hdf5" else RuntimeCfg()
        return RunRecord(cfg=cfg, result=Result(req.data_path))


@pytest.fixture
def app(qapp):
    state = State(replace(_make_empty_ctx(), readiness=ContextReadiness.DRAFT))
    registry = Registry()
    registry.register("demo", LoadAdapter)
    ctrl = Controller(
        state, registry, IOManager(), None, BaseEventBus(), catalog_loader=Loader()
    )
    window = MainWindow(ctrl, dialog_presenter=RecordingDialogPresenter())
    ctrl.add_view(window)
    window.show()
    tab_id = ctrl.new_tab("demo")
    try:
        yield ctrl, window, state, tab_id
    finally:
        ctrl._background_svc.quiesce()
        window.deleteLater()
        qapp.processEvents()


def flush_form_timers():
    loop = QEventLoop()
    QTimer.singleShot(30, loop.quit)
    loop.exec()


def test_local_load_publishes_to_the_same_resource_and_form(app):
    ctrl, window, state, tab_id = app
    resource = ctrl.cfg_resources.lookup(tab_id)
    resource.edit(resource.observe().ref.revision, (CfgEdit(("knob",), 99),))
    before = resource.observe().ref
    outcome = ctrl.load_tab_result(tab_id, "result.hdf5")
    assert outcome.cfg_backfill == "applied"
    after = resource.observe().ref
    assert after.cfg_id == before.cfg_id
    assert after.revision == before.revision + 1
    version = state.version.get(f"tab:{tab_id}:cfg")
    form = window.findChild(ResourceCfgFormWidget)
    assert form is not None
    entry = form.findChild(QLineEdit)
    assert entry is not None and entry.text() == "42"
    flush_form_timers()
    assert resource.observe().ref == after
    assert state.version.get(f"tab:{tab_id}:cfg") == version
    resource.edit(after.revision, (CfgEdit(("knob",), 43),))
    assert entry.text() == "43"
    assert state.get_tab(tab_id).cfg.snapshot_inputs().value.fields[
        "knob"
    ] == DirectValue(43)


@pytest.mark.parametrize(
    "path, disposition", [("result.hdf5", "applied"), ("missing.hdf5", "not_applied")]
)
def test_remote_load_reports_same_result_and_refreshes_live_qt(app, path, disposition):
    ctrl, window, _, tab_id = app
    resource = ctrl.cfg_resources.lookup(tab_id)
    before = resource.observe().ref
    remote = RemoteControlAdapter(
        controller=ctrl,
        opts=ControlOptions(port=0),
        owner_scheduler=QtOwnerScheduler(),
        render_view=window,
    )
    port = remote.start()
    try:
        with open_client(port) as client:
            for method, params in (
                ("tab.snapshot", {"tab_id": tab_id}),
                ("context.snapshot", {}),
            ):
                assert call(client, method, params)["ok"]
            reply = call(client, "tab.load_data", {"tab_id": tab_id, "data_path": path})
            assert reply["ok"], reply
            assert reply["result"]["cfg_backfill"] == disposition
            form = window.findChild(ResourceCfgFormWidget)
            assert form is not None
            entry = form.findChild(QLineEdit)
            assert entry is not None
            assert entry.text() == ("42" if disposition == "applied" else "7")
            after = resource.observe().ref
            assert after.cfg_id == before.cfg_id
            if disposition == "applied":
                assert after.revision == before.revision + 1
                rejected = call(
                    client,
                    "tab.edit_cfg",
                    {
                        "tab_id": tab_id,
                        "expected": {
                            "cfg_id": str(before.cfg_id),
                            "revision": str(before.revision),
                        },
                        "edits": [{"path": ["knob"], "value": 99}],
                    },
                )
                assert rejected["error"]["reason"] == "stale_revision"
                assert entry.text() == "42"
            else:
                assert after == before
    finally:
        remote.stop()


def test_mcp_tab_open_from_file_loads_and_backfills_gui(
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
        tab = invoke("tab_open", {"experiment": "demo", "from_file": "result.hdf5"})[
            "tab"
        ]
        assert tab != previous
        assert state.active_tab_id == tab
        assert state.get_tab(tab).cfg.snapshot_inputs().value.fields[
            "knob"
        ] == DirectValue(42)
        summary = invoke("tab_get", {"tab": tab, "include": ["summary"]})["summary"]
        assert summary["state"]["has_result"] is True
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
        self: LoadAdapter, req: LoadDataRequest
    ) -> RunRecord[RuntimeCfg, Result]:
        raise LoadDataError("bad data", reason_code="invalid_data_file")

    monkeypatch.setattr(LoadAdapter, "load", reject_load)
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
            invoke("tab_open", {"experiment": "demo", "from_file": "bad.hdf5"})
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


@pytest.mark.parametrize(
    "path, expected",
    [
        ("result.hdf5", "Loaded data from result.hdf5"),
        ("missing.hdf5", "Loaded data from missing.hdf5; Config was not backfilled"),
    ],
)
def test_qt_load_status_reports_one_overall_disposition(
    app, monkeypatch, path, expected
):
    _, window, _, _ = app
    messages: list[str] = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: (path, ""))
    monkeypatch.setattr(window, "show_status_message", messages.append)
    tab = window.findChild(ExpTabWidget)
    assert tab is not None
    load_button = next(
        button
        for button in tab.findChildren(QPushButton)
        if button.text() == "Load Data"
    )
    assert load_button.isEnabled()
    load_button.click()
    assert messages == [expected]


def test_reload_captures_backfilled_config_not_retired_editor(app):
    ctrl, _, state, tab_id = app
    ctrl.load_tab_result(tab_id, "result.hdf5")
    preview = ctrl.prepare_experiment_reload()
    ctrl.reload_experiments(preview)
    restored = ctrl.list_tab_ids()
    assert len(restored) == 1
    assert restored[0] != tab_id
    assert state.get_tab(restored[0]).cfg.snapshot_inputs().value.fields[
        "knob"
    ] == DirectValue(42)
