"""Figure area: one focused figure, a thumbnail strip and Follow latest.

A synthetic workflow draws named figures from a per-test script. The
assertions read the strip's buttons in on-screen order, their checked frame,
the focused figure's title, the Follow latest box and the strip's scroll
position; the host never sees a workflow name.
"""

from collections.abc import Callable, Sequence
from time import monotonic

from qtpy.QtCore import QEventLoop, QPoint, QRect, QTimer
from qtpy.QtGui import QImage
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QScrollArea,
    QStackedWidget,
    QToolButton,
    QWidget,
)
from zcu_tools.gui.app.workflows.window import WorkflowWindow

from ._window_support import FigurePoint, WindowSession, bind_point, install_soc

RUN_ID = "figures"


def child[W: QWidget](window: QWidget, kind: type[W], name: str) -> W:
    found = window.findChildren(kind, name)
    assert len(found) == 1, name
    return found[0]


def wait_for(qapp: QApplication, condition: Callable[[], bool]) -> None:
    """Run the Qt event loop in short slices until condition holds or 5 s pass."""
    deadline = monotonic() + 5.0
    while not condition():
        assert monotonic() < deadline, "condition not reached within 5 s"
        loop = QEventLoop()
        QTimer.singleShot(20, loop.quit)
        loop.exec()
    qapp.processEvents()


def start_figures(
    session: WindowSession,
    qapp: QApplication,
    points: Sequence[FigurePoint],
    *,
    pauses: Sequence[int] = (),
    run_id: str = RUN_ID,
) -> None:
    """Show the Window at 1600x1000 and run the script until Done or a pause."""
    window = session.window
    window.resize(1600, 1000)
    window.show()
    qapp.processEvents()
    session.figure_points = tuple(points)
    session.figure_pauses = set(pauses)
    window.select_workflow("synthetic_figures")
    assert isinstance(window.start(run_id), int)
    session.harness.complete_segment()
    qapp.processEvents()


def bound(session: WindowSession) -> WindowSession:
    """Bind the point and SoC once per test; every run reuses them."""
    bind_point(session.harness, session.root)
    install_soc(session.harness)
    return session


def resume(session: WindowSession, qapp: QApplication, run_id: str = RUN_ID) -> None:
    session.window.resume(run_id)
    session.harness.complete_segment()
    qapp.processEvents()


def lifecycle(window: WorkflowWindow[None]) -> str:
    status = window.controller.status()
    assert status is not None
    return status.lifecycle


def thumbnails(window: WorkflowWindow[None]) -> list[QToolButton]:
    """The strip's thumbnails, left to right as the user sees them."""
    strip = child(window, QScrollArea, "workflow.figures.thumbnails")
    content = strip.widget()
    assert content is not None
    buttons = content.findChildren(QToolButton, "workflow.figures.thumbnail")
    return sorted(buttons, key=lambda button: button.mapTo(content, QPoint(0, 0)).x())


def names(window: WorkflowWindow[None]) -> list[str]:
    return [button.text() for button in thumbnails(window)]


def thumbnail(window: WorkflowWindow[None], name: str) -> QToolButton:
    found = [button for button in thumbnails(window) if button.text() == name]
    assert len(found) == 1, name
    return found[0]


def focused(window: WorkflowWindow[None]) -> str:
    """The one framed thumbnail; it must name the figure shown large."""
    framed = [button.text() for button in thumbnails(window) if button.isChecked()]
    assert len(framed) == 1, framed
    stack = child(window, QStackedWidget, "workflow.figures")
    figure = getattr(stack.currentWidget(), "figure", None)
    assert figure is not None, "the large view shows no figure"
    assert figure.axes[0].get_title() == framed[0]
    return framed[0]


def title(window: WorkflowWindow[None]) -> str:
    return child(window, QLabel, "workflow.figures.title").text()


def follow(window: WorkflowWindow[None]) -> QCheckBox:
    return child(window, QCheckBox, "workflow.figures.follow")


def assert_focus(window: WorkflowWindow[None], name: str, place: str) -> None:
    """name is framed and shown large; the title gives its place, e.g. "2 / 3"."""
    assert focused(window) == name
    assert name in title(window) and f"({place})" in title(window)


def in_view(strip: QScrollArea, button: QToolButton) -> bool:
    viewport = strip.viewport()
    assert viewport is not None
    shown = QRect(button.mapTo(viewport, QPoint(0, 0)), button.size())
    return viewport.rect().contains(shown)


def picture(button: QToolButton) -> QImage:
    return button.icon().pixmap(button.iconSize()).toImage()


def edge(button: QToolButton) -> str:
    """Colour of the thumbnail's left frame, as painted on screen."""
    return button.grab().toImage().pixelColor(1, button.height() // 2).name()


def test_follow_latest_focuses_the_most_recently_updated_figure(
    window_session: WindowSession, qapp: QApplication
) -> None:
    session = bound(window_session)
    window = session.window
    start_figures(
        session,
        qapp,
        [
            FigurePoint(("a/one",)),
            FigurePoint(("b/two",)),
            FigurePoint(("c/three",)),
            FigurePoint(("b/two",)),
        ],
    )

    assert lifecycle(window) == "done"
    assert names(window) == ["a/one", "b/two", "c/three"]
    assert names(window) == list(window.figure_names(RUN_ID))
    assert follow(window).isChecked()
    # The final flush after the last point is not an update: focus stays put.
    assert_focus(window, "b/two", "2 / 3")
    assert edge(thumbnail(window, "b/two")) != edge(thumbnail(window, "a/one"))


def test_thumbnails_keep_creation_order_and_append_new_figures(
    window_session: WindowSession, qapp: QApplication
) -> None:
    session = bound(window_session)
    window = session.window
    start_figures(
        session,
        qapp,
        [
            FigurePoint(draw=("b/two",), create=("a/one",)),
            FigurePoint(("a/one",)),
            FigurePoint(("c/three",)),
        ],
        pauses=[0],
    )

    assert lifecycle(window) == "paused"
    assert names(window) == ["b/two"], "an undrawn figure has nothing to show"
    assert_focus(window, "b/two", "1 / 1")
    resume(session, qapp)
    assert lifecycle(window) == "done"
    assert names(window) == ["a/one", "b/two", "c/three"]
    assert_focus(window, "c/three", "3 / 3")


def test_clicked_thumbnail_keeps_focus_until_follow_latest_is_checked(
    window_session: WindowSession, qapp: QApplication
) -> None:
    session = bound(window_session)
    window = session.window
    start_figures(
        session,
        qapp,
        [
            FigurePoint(("a/one",)),
            FigurePoint(("b/two",)),
            FigurePoint(("c/three",)),
            FigurePoint(("d/four",)),
            FigurePoint(("b/two",)),
        ],
        pauses=[2],
    )
    assert lifecycle(window) == "paused"
    assert_focus(window, "c/three", "3 / 3")
    assert follow(window).isChecked()

    thumbnail(window, "a/one").click()
    assert not follow(window).isChecked()
    assert_focus(window, "a/one", "1 / 3")
    resume(session, qapp)
    assert lifecycle(window) == "done"
    assert names(window) == ["a/one", "b/two", "c/three", "d/four"]
    assert_focus(window, "a/one", "1 / 4")
    assert not follow(window).isChecked()

    follow(window).click()
    assert follow(window).isChecked()
    assert_focus(window, "b/two", "2 / 4")
    thumbnail(window, "d/four").click()
    assert not follow(window).isChecked()
    assert_focus(window, "d/four", "4 / 4")


def test_new_run_turns_follow_latest_back_on_with_only_its_figures(
    window_session: WindowSession, qapp: QApplication
) -> None:
    session = bound(window_session)
    window = session.window
    start_figures(session, qapp, [FigurePoint(("a/one", "b/two"))])
    thumbnail(window, "a/one").click()
    assert not follow(window).isChecked()

    start_figures(session, qapp, [FigurePoint(("c/three",))], run_id="second")
    assert lifecycle(window) == "done"
    assert follow(window).isChecked()
    assert names(window) == ["c/three"]
    assert_focus(window, "c/three", "1 / 1")


def test_thumbnail_strip_scrolls_to_the_focused_figure(
    window_session: WindowSession, qapp: QApplication
) -> None:
    session = bound(window_session)
    window = session.window
    many = tuple(f"fig/{index:02d}" for index in range(14))
    start_figures(session, qapp, [FigurePoint(many)])
    strip = child(window, QScrollArea, "workflow.figures.thumbnails")
    bar = strip.horizontalScrollBar()
    assert bar is not None

    assert names(window) == list(many)
    assert bar.maximum() > 0, "fourteen thumbnails must overflow the strip"
    assert_focus(window, "fig/13", "14 / 14")
    assert in_view(strip, thumbnail(window, "fig/13"))
    assert not in_view(strip, thumbnail(window, "fig/00"))

    thumbnail(window, "fig/00").click()
    assert_focus(window, "fig/00", "1 / 14")
    assert in_view(strip, thumbnail(window, "fig/00"))
    assert not in_view(strip, thumbnail(window, "fig/13"))


def test_thumbnails_show_the_live_figure_content(
    window_session: WindowSession, qapp: QApplication
) -> None:
    session = bound(window_session)
    window = session.window
    start_figures(
        session,
        qapp,
        [FigurePoint(("a/one",)), FigurePoint(("a/one",)), FigurePoint(("a/one",))],
        pauses=[1],
    )
    assert lifecycle(window) == "paused"
    button = thumbnail(window, "a/one")
    wait_for(qapp, lambda: not picture(button).isNull())
    paused = picture(button)

    resume(session, qapp)
    assert lifecycle(window) == "done"
    wait_for(qapp, lambda: picture(button) != paused)
