"""Existing tab views follow the current session source without changing identity."""

import os
from pathlib import Path
from typing import NoReturn

import pytest
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QTabWidget, QWidget
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import Adapter
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
)
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._synthetic import SampleAdapter, SampleCfg


def ProjectionAdapter_run(cfg: SampleCfg, *, context: RunContext) -> NoReturn:
    raise AssertionError("This view contract must not start a Run")


ProjectionAdapter = Adapter(
    spec=SampleAdapter.spec,
    make_form=SampleAdapter.make_form,
    run=ProjectionAdapter_run,
    guide=None,
    analysis=SampleAdapter.analysis,
    post_analysis=SampleAdapter.post_analysis,
)


@pytest.mark.parametrize("point", [None, "working"])
def test_document_set_scope_is_visible_when_opening_inspect(
    qapp, tmp_path, point: str | None
) -> None:
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        Registry(),
        None,
        BaseEventBus(),
        progress_transport=DirectProgressTransport(),
        project_root=str(tmp_path),
    )
    ctrl.entry_control.new_entry(EntryRequest("entry"))
    if point is not None:
        ctrl.entry_control.new_point(
            PointCreateRequest(point, "", CommandOrigin("gui"))
        )
    window = MainWindow(ctrl, templates=lambda: {})
    ctrl.add_view(window)
    window.refresh_context_panel()
    window.resize(1300, 850)
    window.show()
    evidence = Path(os.environ.get("CONTEXT_PROJECTION_EVIDENCE", str(tmp_path)))
    scope = "global" if point is None else "point"
    driver = ComponentDriver(window)
    try:
        qapp.processEvents()
        assert driver.observe(object_name="current_entry").text == "entry"
        assert driver.observe(object_name="current_point").text == (
            "Global" if point is None else point
        )
        driver.screenshot(evidence / scope / "01-scope.png")
        driver.click("Inspect…")
        assert driver.dialogs()
        driver.screenshot(evidence / scope / "02-inspect.png")
        driver.click("Close")
        assert driver.observe(object_name="current_point").text == (
            "Global" if point is None else point
        )
        driver.screenshot(evidence / scope / "03-returned.png")
    finally:
        ctrl.close()
        window.dispose()
        qapp.processEvents()


def _set_window_colours(widget: QWidget, background: str, text: str) -> None:
    palette = widget.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(background))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(text))
    widget.setPalette(palette)


def test_existing_tab_tracks_unselected_and_active_source(
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
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    ctrl.add_view(window)
    try:
        from qtpy.QtWidgets import QLabel

        _set_window_colours(window, "white", "black")
        window.refresh_context_panel()
        tab_id = ctrl.tab_control.new_tab("fake")
        tabs = window.findChild(QTabWidget, "experiment.tabs")
        assert tabs is not None
        title = tabs.tabText(0)
        cfg = ctrl.cfg_resources.lookup(tab_id)
        interaction = ctrl.tab_control.get_tab_snapshot(tab_id).interaction
        assert interaction is not None
        assert not interaction.has_context
        assert not interaction.has_active_context
        assert window.get_view_snapshot()["point_label"] == "No point selected"
        ctrl.entry_control.new_entry(EntryRequest("entry"))
        interaction = ctrl.tab_control.get_tab_snapshot(tab_id).interaction
        assert interaction is not None
        assert interaction.has_context and interaction.has_active_context
        assert window.get_view_snapshot()["point_label"] == "Global"
        for label in ("first", "second"):
            ctrl.entry_control.new_point(
                PointCreateRequest(label, "", CommandOrigin("gui"))
            )
            interaction = ctrl.tab_control.get_tab_snapshot(tab_id).interaction
            assert interaction is not None
            assert interaction.has_context and interaction.has_active_context
            assert window.get_view_snapshot()["point_label"] == label
        ctrl.entry_control.use_point("first")
        assert window.get_view_snapshot()["point_label"] == "first"
        assert ctrl.tab_control.list_tab_ids() == [tab_id]
        assert ctrl.cfg_resources.lookup(tab_id) is cfg
        assert tabs.tabText(0) == title
        _set_window_colours(window, "#353535", "white")
        qapp.processEvents()
        for name in ("current_entry", "current_point"):
            label = window.findChild(QLabel, name)
            assert label is not None
            assert label.palette().color(
                label.foregroundRole()
            ) == window.palette().color(QPalette.ColorRole.WindowText), name
    finally:
        assert background.quiesce()
        for tab_id in tuple(ctrl.tab_control.list_tab_ids()):
            ctrl.tab_control.close_tab(tab_id)
        ctrl.close()
        window.dispose()
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
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    ctrl.add_view(window)
    window.show()
    qapp.processEvents()
    try:
        hint = window.findChild(QLabel, "empty_tabs_hint")
        tabs = window.findChild(QTabWidget, "experiment.tabs")
        assert hint is not None and tabs is not None
        assert hint.isVisibleTo(window) and not tabs.isVisibleTo(window)
        without_entry = hint.text()
        ctrl.entry_control.new_entry(EntryRequest("entry"))
        qapp.processEvents()
        assert hint.isVisibleTo(window)
        without_point = hint.text()
        assert without_point and without_point != without_entry
        ctrl.entry_control.new_point(
            PointCreateRequest("first", "", CommandOrigin("gui"))
        )
        qapp.processEvents()
        with_point = hint.text()
        assert with_point == without_point
        ctrl.entry_control.open_entry(EntryRequest("entry"))
        qapp.processEvents()
        assert hint.text() == without_point
        ctrl.entry_control.use_point("first")
        qapp.processEvents()
        assert hint.text() == with_point
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
        ctrl.close()
        window.dispose()
        qapp.processEvents()
