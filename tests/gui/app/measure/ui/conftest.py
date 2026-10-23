"""Shared real measure UI composition and teardown for public view contracts."""

from collections.abc import Callable, Generator, Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.control_identity import ControlIdentity

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    NativeSaveSession,
    native_save_session,
)
from tests.gui.app.measure._record_support import require_record


@pytest.fixture
def main_window_control_factory(
    qapp: QApplication,
) -> Iterator[Callable[[], ControlIdentity]]:
    """Keep fake application authority alive until its Qt view tree is deleted."""
    baseline = set(qapp.topLevelWidgets())
    authorities: list[ControlIdentity] = []

    def create() -> ControlIdentity:
        authority = ControlIdentity()
        authorities.append(authority)
        return authority

    yield create
    for widget in qapp.topLevelWidgets():
        if widget not in baseline:
            widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    authorities.clear()


@pytest.fixture
def live_registry() -> Registry:
    """Inject the synthetic adapter catalog used by real Controller UI scenarios."""
    registry = Registry()
    registry.register("saving", AnalysisSavingAdapter)
    return registry


@pytest.fixture
def live_ui(
    tmp_path: Path, qapp: QApplication, monkeypatch: pytest.MonkeyPatch,
    live_registry: Registry,
) -> Generator[tuple[NativeSaveSession, Controller, MainWindow, str]]:
    """Open a real tab and view over the existing offline native Save fixture."""
    with native_save_session(tmp_path, qapp) as session:
        monkeypatch.setattr(
            "zcu_tools.gui.app.measure.services.app_services.BackgroundRunner",
            lambda: session.background,
        )
        ctrl = Controller(
            session.state,
            live_registry,
            None,
            session.bus,
            progress_transport=DirectProgressTransport(),
            project_root=str(tmp_path),
        )
        tab_id = ctrl.new_tab("saving")
        original = session.state.get_tab("tab")
        session.state.update_tab_result(
            tab_id,
            replace(require_record(original.run), metadata=session.metadata),
            parameters=original.run.parameters,
        )
        window = MainWindow(
            ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
        )
        window.add_tab_widget(tab_id, "saving")
        try:
            yield session, ctrl, window, tab_id
        finally:
            assert session.background.quiesce()
            ctrl.tab_control.close_tab(tab_id)
            window.deleteLater()
            qapp.processEvents()
