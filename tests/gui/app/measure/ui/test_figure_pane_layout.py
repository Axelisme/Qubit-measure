"""Selection and geometry at the shared figure pane boundary."""

from pathlib import Path

from qtpy.QtCore import QCoreApplication, QEvent, QObject, QRect
from qtpy.QtWidgets import QApplication, QLabel, QWidget
from zcu_tools.gui.app.measure.ui.figure_pane_layout import FigurePaneLayout

from tests.gui._user_input import ComponentDriver


class _MemberBounds(QObject):
    def __init__(self, members: tuple[QLabel, ...]) -> None:
        super().__init__()
        self.before = {member.objectName(): member.geometry() for member in members}
        self.changes: list[tuple[str, QRect, QRect]] = []
        for member in members:
            member.installEventFilter(self)

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if (
            isinstance(a0, QWidget)
            and a0.isVisible()
            and a1 is not None
            and a1.type() in (QEvent.Type.Move, QEvent.Type.Resize)
        ):
            name = a0.objectName()
            before, during = self.before[name], a0.geometry()
            if before != during:
                self.changes.append((name, before, during))
        return False


def test_selecting_a_tile_keeps_every_visible_member_in_its_bounds(
    qapp: QApplication, tmp_path: Path
) -> None:
    pane = FigurePaneLayout()
    try:
        pane.addWidget(QLabel("No figure yet"))
        members = tuple(QLabel(f"Figure {index}") for index in range(1, 5))
        for index, member in enumerate(members, start=1):
            member.setObjectName(f"figure-{index}")
            pane.addWidget(member)
        pane.set_tiled(tiled=True)
        pane.set_page(0)
        pane.resize(640, 480)
        pane.show()
        driver = ComponentDriver(pane)
        driver.screenshot(tmp_path / "before-selection.png")
        assert all(
            driver.observe(object_name=tile.objectName()).visible for tile in members
        )
        bounds = _MemberBounds(members)
        assert not bounds.before["figure-1"].intersects(bounds.before["figure-4"])

        driver.click(object_name="figure-4")
        driver.screenshot(tmp_path / "after-selection.png")
        assert pane.currentWidget() is members[3]
        assert {tile.objectName(): tile.geometry() for tile in members} == bounds.before
        assert not bounds.changes, bounds.changes
    finally:
        pane.close()
        pane.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
