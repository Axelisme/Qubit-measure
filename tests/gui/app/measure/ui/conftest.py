"""Shared real measure UI composition and teardown for public view contracts."""

from collections.abc import Generator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.ui.main_window import MainWindow

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    NativeSaveSession,
    native_save_session,
)


@pytest.fixture
def live_ui(
    tmp_path: Path, qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[NativeSaveSession, Controller, MainWindow, str]]:
    """Open a real tab and view over the existing offline native Save fixture."""
    with native_save_session(tmp_path, qapp) as session:
        registry = Registry()
        registry.register("saving", AnalysisSavingAdapter)
        monkeypatch.setattr(
            "zcu_tools.gui.app.measure.services.app_services.BackgroundRunner",
            lambda: session.background,
        )
        ctrl = Controller(
            session.state,
            registry,
            None,
            session.bus,
            progress_transport=DirectProgressTransport(),
            project_root=str(tmp_path),
        )
        tab_id = ctrl.new_tab("saving")
        original = session.state.get_tab("tab")
        session.state.update_tab_result(
            tab_id,
            original.run.result,
            metadata=session.metadata,
            parameters=original.run.parameters,
        )
        window = MainWindow(ctrl, dialog_presenter=RecordingDialogPresenter())
        window.add_tab_widget(tab_id, "saving")
        try:
            yield session, ctrl, window, tab_id
        finally:
            assert session.background.quiesce()
            ctrl.tab_control.close_tab(tab_id)
            window.deleteLater()
            qapp.processEvents()
