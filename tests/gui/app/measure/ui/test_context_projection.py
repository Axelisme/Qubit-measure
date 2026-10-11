"""Existing tab views follow the current session source without changing identity."""

from collections.abc import Mapping
from typing import NoReturn

from qtpy.QtWidgets import QTabWidget
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import RunRequest
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
    PointUseRequest,
)
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from zcu_lab.v2.fake.stub.gui import FakeAdapter


class ProjectionAdapter(FakeAdapter):
    """Keep the source/view seam real while forbidding execution."""

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, object],
        *,
        context: RunContext,
    ) -> NoReturn:
        raise AssertionError("This view contract must not start a Run")


def test_existing_tab_tracks_empty_draft_and_active_source(
    qapp, tmp_path, monkeypatch
) -> None:
    background = BackgroundRunner()
    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.app_services.BackgroundRunner",
        lambda: background,
    )
    registry = Registry()
    registry.register("fake", ProjectionAdapter)
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        BaseEventBus(),
        progress_transport=DirectProgressTransport(),
        project_root=str(tmp_path),
    )
    window = MainWindow(ctrl, dialog_presenter=RecordingDialogPresenter())
    ctrl.add_view(window)
    try:
        tab_id = ctrl.tab_control.new_tab("fake")
        tabs = window.findChild(QTabWidget)
        assert tabs is not None
        title = tabs.tabText(0)
        cfg_id = ctrl.cfg_resources.lookup(tab_id).observe().ref.cfg_id
        interaction = ctrl.get_tab_snapshot(tab_id).interaction
        assert interaction is not None
        assert not interaction.has_context
        assert not interaction.has_active_context
        assert window.get_view_snapshot()["point_label"] == "No point selected"
        ctrl.setup_control.new_entry(EntryRequest("entry"))
        interaction = ctrl.get_tab_snapshot(tab_id).interaction
        assert interaction is not None
        assert interaction.has_context
        assert not interaction.has_active_context
        assert window.get_view_snapshot()["point_label"] == "Draft (in-memory)"
        for label in ("first", "second"):
            ctrl.setup_control.new_point(
                PointCreateRequest(label, "", CommandOrigin("gui"))
            )
            interaction = ctrl.get_tab_snapshot(tab_id).interaction
            assert interaction is not None
            assert interaction.has_context and interaction.has_active_context
            assert window.get_view_snapshot()["point_label"] == label
        ctrl.setup_control.use_point(PointUseRequest("first"))
        assert window.get_view_snapshot()["point_label"] == "first"
        assert ctrl.tab_control.list_tab_ids() == [tab_id]
        assert ctrl.cfg_resources.lookup(tab_id).observe().ref.cfg_id == cfg_id
        assert tabs.tabText(0) == title
    finally:
        assert background.quiesce()
        for tab_id in tuple(ctrl.tab_control.list_tab_ids()):
            ctrl.tab_control.close_tab(tab_id)
        window.deleteLater()
        qapp.processEvents()


def test_empty_tab_area_shows_next_step_hint_until_a_tab_opens(
    qapp, tmp_path, monkeypatch
) -> None:
    from qtpy.QtWidgets import QLabel

    background = BackgroundRunner()
    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.app_services.BackgroundRunner",
        lambda: background,
    )
    registry = Registry()
    registry.register("fake", ProjectionAdapter)
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        BaseEventBus(),
        progress_transport=DirectProgressTransport(),
        project_root=str(tmp_path),
    )
    window = MainWindow(ctrl, dialog_presenter=RecordingDialogPresenter())
    ctrl.add_view(window)
    window.show()
    qapp.processEvents()
    try:
        hint = window.findChild(QLabel, "empty_tabs_hint")
        tabs = window.findChild(QTabWidget)
        assert hint is not None and tabs is not None
        assert hint.isVisibleTo(window) and not tabs.isVisibleTo(window)
        without_entry = hint.text()
        ctrl.setup_control.new_entry(EntryRequest("entry"))
        qapp.processEvents()
        assert hint.isVisibleTo(window)
        assert hint.text() and hint.text() != without_entry
        tab_id = ctrl.tab_control.new_tab("fake")
        qapp.processEvents()
        assert tabs.isVisibleTo(window) and not hint.isVisibleTo(window)
        ctrl.tab_control.close_tab(tab_id)
        qapp.processEvents()
        assert hint.isVisibleTo(window) and not tabs.isVisibleTo(window)
    finally:
        assert background.quiesce()
        for tab_id in tuple(ctrl.tab_control.list_tab_ids()):
            ctrl.tab_control.close_tab(tab_id)
        window.close()
        window.deleteLater()
        qapp.processEvents()
