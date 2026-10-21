"""Screenshots of measure-gui named dialogs through the public registry."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QPushButton,
    QTabWidget,
    QWidget,
)
from zcu_tools.cfg_model import PulseSpec, TemplateDef
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.main_dialog_registry import MainDialogRegistry
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.inspect_dialog import InspectDialog
from zcu_tools.gui.widgets import DialogRefStore
from zcu_tools.mcp.measure.session import GuiRpcError

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._app_launch import (
    LaunchedMeasureApp,
    click_toolbar_setup,
    launched_measure_app,
    visible_setup_dialogs,
)


def test_arb_waveform_named_dialog_captures_only_while_open(qapp) -> None:
    parent = QWidget()
    ctrl = MagicMock()
    ctrl.arb_waveforms.list_infos.return_value = []
    ctrl.arb_waveforms.list_data_keys.return_value = []
    registry = MainDialogRegistry(
        ctrl,
        parent=parent,
        dialog_refs=DialogRefStore(),
        templates=lambda: {},
        dialog_presenter=RecordingDialogPresenter(),
        alias_presets=None,
    )

    with pytest.raises(RuntimeError, match="not currently open"):
        registry.take_screenshot(DialogName.ARB_WAVEFORM)

    try:
        registry.open(DialogName.ARB_WAVEFORM)
        qapp.processEvents()
        assert DialogName.ARB_WAVEFORM in registry.visible_names()
        assert registry.take_screenshot(DialogName.ARB_WAVEFORM).startswith(b"\x89PNG")
    finally:
        registry.close(DialogName.ARB_WAVEFORM)
        qapp.processEvents()

    with pytest.raises(RuntimeError, match="not currently open"):
        registry.take_screenshot(DialogName.ARB_WAVEFORM)


@pytest.mark.uses_wall_clock
@pytest.mark.parametrize("opening", ["launch", "toolbar_focus", "toolbar_reopen"])
def test_setup_screenshot_uses_visible_gui_dialog_through_mcp(
    qapp: QApplication, tmp_path: Path, opening: str
) -> None:
    with launched_measure_app(qapp, tmp_path) as app:
        launched = visible_setup_dialogs(app.window)
        assert len(launched) == 1
        if opening != "launch":
            if opening == "toolbar_reopen":
                app.window.close_dialog(DialogName.SETUP)
                qapp.processEvents()
            click_toolbar_setup(app.window)
            qapp.processEvents()
            if opening == "toolbar_focus":
                assert visible_setup_dialogs(app.window) == launched
        assert len(visible_setup_dialogs(app.window)) == 1
        expected = [DialogName.SETUP]
        assert app.window.list_open_dialogs() == expected

        path = Path(
            app.invoke(
                "rpc_call",
                {
                    "method": "dialog.screenshot",
                    "params": {
                        "name": "setup",
                        "out_path": str(tmp_path / "setup.png"),
                    },
                },
            )["saved_to"]
        )
        assert path.is_file() and path.read_bytes().startswith(b"\x89PNG")
        assert app.window.list_open_dialogs() == expected


@pytest.mark.uses_wall_clock
def test_soc_named_screenshot_through_mcp(qapp: QApplication, tmp_path: Path) -> None:
    with launched_measure_app(qapp, tmp_path) as app:
        ui = ComponentDriver(app.window)
        ui.click("Close")
        assert any(w.text == "SoC…" for w in ui.visible().widgets), (
            "SoC toolbar entry missing"
        )
        ui.click("SoC…")
        reply = app.invoke(
            "rpc_call",
            {
                "method": "dialog.screenshot",
                "params": {"name": "soc", "out_path": str(tmp_path / "soc.png")},
            },
        )
        assert Path(reply["saved_to"]).read_bytes().startswith(b"\x89PNG")
        ui.click("Close")
        with pytest.raises(RuntimeError, match="not currently open"):
            app.invoke(
                "rpc_call",
                {"method": "dialog.screenshot", "params": {"name": "soc"}},
            )


def _inspect_screenshot(app: LaunchedMeasureApp, tmp_path: Path) -> Path:
    reply = app.invoke(
        "rpc_call",
        {
            "method": "dialog.screenshot",
            "params": {"name": "inspect", "out_path": str(tmp_path / "inspect.png")},
        },
    )
    return Path(reply["saved_to"])


def _inspect_tabs(app: LaunchedMeasureApp) -> QTabWidget:
    (inspect,) = [d for d in app.window.findChildren(InspectDialog) if d.isVisible()]
    tabs = inspect.findChild(QTabWidget)
    assert tabs is not None
    return tabs


def _click_inspect_tab(app: LaunchedMeasureApp, title: str) -> None:
    tabs = _inspect_tabs(app)
    (index,) = [i for i in range(tabs.count()) if tabs.tabText(i) == title]
    tabs.setCurrentIndex(index)


def _current_inspect_page(app: LaunchedMeasureApp) -> str:
    tabs = _inspect_tabs(app)
    return tabs.tabText(tabs.currentIndex())


@pytest.mark.uses_wall_clock
def test_inspect_screenshot_captures_the_shown_page_without_switching_through_mcp(
    qapp: QApplication, tmp_path: Path
) -> None:
    with launched_measure_app(qapp, tmp_path) as app:
        (startup,) = visible_setup_dialogs(app.window)
        startup.reject()
        app.window.open_dialog(DialogName.INSPECT)
        qapp.processEvents()

        for title in ("Aliases", "Modules", "Parameters"):
            _click_inspect_tab(app, title)
            qapp.processEvents()
            path = _inspect_screenshot(app, tmp_path)
            assert path.read_bytes().startswith(b"\x89PNG")
            assert _current_inspect_page(app) == title

        with pytest.raises(GuiRpcError, match="unknown dialog name"):
            app.invoke(
                "rpc_call",
                {"method": "dialog.screenshot", "params": {"name": "aliases"}},
            )
        app.window.close_dialog(DialogName.INSPECT)
        qapp.processEvents()


def test_inspect_uses_composition_templates_and_the_named_arb_dialog(
    qapp: QApplication, tmp_path: Path
) -> None:
    registry = Registry(
        templates={
            "probe": TemplateDef(
                PulseSpec,
                "Probe pulse",
                {"ch": 0, "nqz": 1, "freq": 1.0, "gain": 0.1, "waveform": "w"},
                default_name="probe",
            )
        }
    )
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        BaseEventBus(),
        progress_transport=DirectProgressTransport(),
        project_root=str(tmp_path),
    )
    parent = QWidget()
    dialogs = MainDialogRegistry(
        ctrl,
        parent=parent,
        dialog_refs=DialogRefStore(),
        templates=lambda: registry.templates,
        dialog_presenter=RecordingDialogPresenter(),
        alias_presets=None,
    )
    try:
        dialogs.open(DialogName.INSPECT)
        inspect = dialogs.dialog(DialogName.INSPECT)
        assert inspect is not None
        next(b for b in inspect.findChildren(QPushButton) if b.text() == "New").click()
        new = inspect.findChild(QDialog, "inspect.new_entry")
        assert new is not None
        combo = new.findChild(QComboBox, "inspect.new_entry.template")
        assert combo is not None
        assert [combo.itemText(i) for i in range(combo.count())] == [
            "Probe pulse  (module)"
        ]
        next(
            b for b in inspect.findChildren(QPushButton) if b.text() == "Arb Waveforms…"
        ).click()
        qapp.processEvents()
        assert DialogName.ARB_WAVEFORM in dialogs.visible_names()
        assert dialogs.take_screenshot(DialogName.ARB_WAVEFORM).startswith(b"\x89PNG")
    finally:
        dialogs.close(DialogName.ARB_WAVEFORM)
        dialogs.close(DialogName.INSPECT)
        parent.deleteLater()
        qapp.processEvents()
