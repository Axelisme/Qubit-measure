"""Window layout: session strip above a form column and a run/figure column.

Synthetic workflows only; the assertions read widget geometry, visibility,
ancestry and enabled state, never workflow names inside the host.
"""

import pytest
from qtpy.QtCore import QPoint, QRect, Qt
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QWidget,
)
from zcu_tools.gui.app.workflows.panel import WorkflowFormPanel
from zcu_tools.gui.app.workflows.window import WorkflowWindow

from ._window_support import WindowSession, bind_point, expand_all

EXISTING_NAMES = (
    "workflow.select",
    "workflow.run.message",
    "workflow.run.start",
    "workflow.run.pause",
    "workflow.run.resume",
    "workflow.run.stop",
    "workflow.source.setup",
    "workflow.source.aliases",
    "workflow.source.status",
    "workflow.source.use",
    "workflow.source.reload",
)
RUN_ROW = (
    "workflow.run.start",
    "workflow.run.pause",
    "workflow.run.resume",
    "workflow.run.stop",
)
SOURCE_BUTTONS = (
    "workflow.source.setup",
    "workflow.source.aliases",
    "workflow.source.use",
    "workflow.source.reload",
)


def child[W: QWidget](window: QWidget, kind: type[W], name: str) -> W:
    found = window.findChildren(kind, name)
    assert len(found) == 1, name
    return found[0]


def area(window: QWidget, widget: QWidget) -> QRect:
    return QRect(widget.mapTo(window, QPoint(0, 0)), widget.size())


def columns(window: WorkflowWindow[None]) -> tuple[QSplitter, QWidget, QWidget]:
    splitter = child(window, QSplitter, "workflow.split")
    assert splitter.orientation() == Qt.Orientation.Horizontal
    assert splitter.count() == 2
    left, right = splitter.widget(0), splitter.widget(1)
    assert left is not None and right is not None
    return splitter, left, right


def show(window: WorkflowWindow[None], qapp: QApplication, w: int, h: int) -> None:
    window.resize(w, h)
    window.show()
    qapp.processEvents()


def expand_forms(window: WorkflowWindow[None], qapp: QApplication) -> None:
    expand_all(window.plan_panel, qapp)
    expand_all(window.tunables_panel, qapp)
    if window.tuning_panel is not None:
        expand_all(window.tuning_panel, qapp)


def tab_page(tabs: QTabWidget, index: int) -> QWidget:
    page = tabs.widget(index)
    assert page is not None
    return page


def assert_scrolled_in_tab(tabs: QTabWidget, index: int, panel: QWidget) -> None:
    """panel is shown inside a scroll area that belongs to that tab."""
    page = tab_page(tabs, index)
    assert page.isAncestorOf(panel) and panel.isVisibleTo(page)
    scroll = panel.parentWidget()
    while scroll is not None and not isinstance(scroll, QScrollArea):
        scroll = scroll.parentWidget()
    assert scroll is not None
    assert scroll is page or page.isAncestorOf(scroll)


def assert_only_form_in_tab(tabs: QTabWidget, index: int, panel: QWidget) -> None:
    """panel is that tab's scrolled form and no replaced form is still shown."""
    assert_scrolled_in_tab(tabs, index, panel)
    page = tab_page(tabs, index)
    shown = [
        widget
        for widget in page.findChildren(QWidget)
        if isinstance(widget, WorkflowFormPanel)
        and widget.isVisibleTo(page)
        and widget is not panel
        and not panel.isAncestorOf(widget)
    ]
    assert shown == [], "a replaced or second form remains shown in this tab"


def assert_d_layout(window: WorkflowWindow[None], qapp: QApplication) -> None:
    """Strip on top; selector, tabs and fixed run row left; run status/figures right."""
    show(window, qapp, 1600, 1000)
    assert (window.width(), window.height()) == (1600, 1000)
    splitter, left, right = columns(window)
    strip = child(window, QWidget, "workflow.session")
    assert area(window, strip).bottom() < area(window, splitter).top()
    for name in SOURCE_BUTTONS:
        button = child(window, QPushButton, name)
        assert button.isVisible()
        assert area(window, button).bottom() < area(window, splitter).top()
        assert area(window, button).left() > area(window, strip).right()

    selector = child(window, QComboBox, "workflow.select")
    tabs = child(window, QTabWidget, "workflow.forms")
    for widget in (selector, tabs):
        assert left.isAncestorOf(widget) and widget.isVisible()
    assert area(window, selector).bottom() < area(window, tabs).top()
    assert (tabs.tabText(0), tabs.tabText(1)) == ("Plan", "Tunables")
    row = [child(window, QPushButton, name) for name in RUN_ROW]
    for button in row:
        assert left.isAncestorOf(button) and button.isVisible()
        assert area(window, button).top() > area(window, tabs).bottom()
        assert area(window, button).bottom() <= window.height()
        # One row: styled buttons may differ in height, not in centre line.
        assert (
            abs(area(window, button).center().y() - area(window, row[0]).center().y())
            <= 2
        )
    lefts = [area(window, button).left() for button in row]
    assert lefts == sorted(lefts)

    message = child(window, QLabel, "workflow.run.message")
    figures = child(window, QStackedWidget, "workflow.figures")
    for widget in (message, figures):
        assert right.isAncestorOf(widget) and widget.isVisible()
    assert area(window, message).bottom() <= area(window, figures).top()
    assert figures.height() >= 600


def test_existing_object_names_are_unique_and_visible(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = window_session.window
    show(window, qapp, 1600, 1000)
    for name in EXISTING_NAMES:
        assert child(window, QWidget, name).isVisible(), name


def test_expanded_forms_keep_the_run_row_and_tall_figures_at_1600x1000(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = window_session.window
    expand_forms(window, qapp)
    assert_d_layout(window, qapp)
    tabs = child(window, QTabWidget, "workflow.forms")
    assert_only_form_in_tab(tabs, 0, window.plan_panel)
    assert_only_form_in_tab(tabs, 1, window.tunables_panel)


def test_splitter_resizes_the_form_column_against_the_figures(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = window_session.window
    show(window, qapp, 1600, 1000)
    splitter, left, right = columns(window)
    splitter.setSizes([700, 900])
    qapp.processEvents()
    wide_form = left.width()
    splitter.setSizes([500, 1100])
    qapp.processEvents()
    assert left.width() < wide_form
    assert right.width() >= 1000


@pytest.mark.parametrize("name", ["synthetic_wide", "synthetic_small"])
def test_minimum_height_stays_within_800_with_every_section_expanded(
    window_session: WindowSession, qapp: QApplication, name: str
) -> None:
    window = window_session.window
    window.select_workflow(name)
    expand_forms(window, qapp)
    assert window.minimumSizeHint().height() <= 800


def test_replaced_panels_appear_in_their_own_tabs(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = window_session.window
    bind_point(window_session.harness, window_session.root)
    tabs = child(window, QTabWidget, "workflow.forms")
    show(window, qapp, 1600, 1000)
    tabs.setCurrentIndex(1)

    def replace_with(action: str) -> None:
        old = window.plan_panel, window.tunables_panel
        if action == "select":
            window.select_workflow("synthetic_small")
        elif action == "prepare":
            window.prepare_forms()
        else:
            window_session.source.catalog = window_session.fresh_catalog()
            window.reload_source()
        qapp.processEvents()
        assert window.plan_panel is not old[0]
        assert window.tunables_panel is not old[1]
        assert_only_form_in_tab(tabs, 0, window.plan_panel)
        assert_only_form_in_tab(tabs, 1, window.tunables_panel)
        assert tabs.currentIndex() == 1 and window.tunables_panel.isVisible()

    for action in ("select", "prepare", "reload"):
        replace_with(action)
    assert_d_layout(window, qapp)
