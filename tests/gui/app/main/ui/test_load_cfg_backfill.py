from dataclasses import dataclass, replace
from typing import ClassVar

import pytest
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QFileDialog, QPushButton
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    LoadDataRequest,
)
from zcu_tools.gui.app.main.app import _make_empty_ctx
from zcu_tools.gui.app.main.controller import Controller
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError
from zcu_tools.gui.app.main.services.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.app.main.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.main.ui.main_window import MainWindow
from zcu_tools.gui.cfg import DirectValue
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.gui.session.types import ContextReadiness
from zcu_tools.gui.widgets.cfg import CfgFormWidget

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.main.services.remote._helpers import open_client, recv_response, send
from tests.gui.services.test_experiment_reload import Loader, OldAdapter


class RuntimeCfg(ExpCfgModel):
    knob: int = 42


@dataclass
class Result:
    cfg_snapshot: RuntimeCfg | None


class LoadAdapter(OldAdapter):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        load_data=True, analysis=AnalysisMode.NONE
    )

    def load(self, req: LoadDataRequest) -> Result:
        return Result(None if req.data_path == "missing.hdf5" else RuntimeCfg())


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


def test_local_load_rebinds_form_and_drops_pending_old_snapshot(app):
    ctrl, window, state, tab_id = app
    original = ctrl.editor_id_for_owner(tab_id)
    assert original is not None
    ctrl.cfg_editor_set_field(original, "knob", 99)
    outcome = ctrl.load_tab_result(tab_id, "result.hdf5")
    assert outcome.cfg_backfill == "applied"
    version = state.version.get(f"tab:{tab_id}:cfg")
    tab = window.findChild(ExpTabWidget)
    assert tab is not None
    form = tab.findChild(CfgFormWidget)
    assert form is not None
    assert form.read_schema().value.fields["knob"] == DirectValue(42)
    flush_form_timers()
    assert state.get_tab(tab_id).cfg_schema.value.fields["knob"] == DirectValue(42)
    assert state.version.get(f"tab:{tab_id}:cfg") == version
    with pytest.raises(CfgEditorError):
        ctrl.cfg_editor_set_field(original, "knob", 100)
    replacement = ctrl.editor_id_for_owner(tab_id)
    assert replacement is not None
    ctrl.cfg_editor_set_field(replacement, "knob", 43)
    flush_form_timers()
    assert form.read_schema().value.fields["knob"] == DirectValue(43)
    assert state.get_tab(tab_id).cfg_schema.value.fields["knob"] == DirectValue(43)


@pytest.mark.parametrize(
    "path, disposition", [("result.hdf5", "applied"), ("missing.hdf5", "not_applied")]
)
def test_remote_load_reports_same_result_and_refreshes_live_qt(app, path, disposition):
    ctrl, window, state, tab_id = app
    original = ctrl.editor_id_for_owner(tab_id)
    assert original is not None
    remote = RemoteControlAdapter(
        controller=ctrl,
        opts=ControlOptions(port=0),
        owner_scheduler=QtOwnerScheduler(),
        render_view=window,
    )
    port = remote.start()
    try:
        with open_client(port) as client:
            send(
                client,
                {
                    "id": "load",
                    "method": "tab.load_data",
                    "params": {"tab_id": tab_id, "data_path": path},
                },
            )
            reply = recv_response(client, "load")
            assert reply["ok"], reply
            assert reply["result"]["cfg_backfill"] == disposition
            form = window.findChild(CfgFormWidget)
            assert form is not None
            assert form.read_schema() == state.get_tab(tab_id).cfg_schema
            assert form.read_schema().value.fields["knob"] == DirectValue(
                42 if disposition == "applied" else 7
            )
            current = ctrl.editor_id_for_owner(tab_id)
            if disposition == "applied":
                assert current is not None and current != original
                send(
                    client,
                    {
                        "id": "stale",
                        "method": "editor.set_field",
                        "params": {
                            "editor_id": original,
                            "path": "knob",
                            "value": 99,
                        },
                    },
                )
                assert not recv_response(client, "stale")["ok"]
                assert form.read_schema().value.fields["knob"] == DirectValue(42)
            else:
                assert current == original
    finally:
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
    assert state.get_tab(restored[0]).cfg_schema.value.fields["knob"] == DirectValue(42)
