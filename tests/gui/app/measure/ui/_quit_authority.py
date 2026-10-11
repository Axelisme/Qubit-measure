"""Process-boundary probe for Measure's real QApplication exit lifecycle."""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.figure import Figure
from qtpy.QtCore import QCoreApplication, QEvent, QTimer
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.template_catalog import TemplateCatalog
from zcu_tools.gui.app.measure.ui.interactive_frontend import InteractiveFrontend
from zcu_tools.gui.app.measure.ui.main_window import MainWindow

from tests.gui.app.measure._native_save_support import AnalysisSavingAdapter
from tests.gui.app.measure._reload_fakes import Loader


class RecordingFrontend(InteractiveFrontend):
    """Record only the authority projections crossing the plugin view boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.read_only_changes: list[bool] = []
        self._figure = Figure()

    def set_read_only(self, *, read_only: bool) -> None:
        self.read_only_changes.append(read_only)

    @property
    def figure(self) -> Figure:
        return self._figure

    @property
    def preview_active(self) -> bool:
        return False

    def cancel_preview(self) -> None:
        pass

    def teardown(self) -> None:
        pass


def main(project_root: Path) -> int:
    """Return a failure with explicit stderr if Qt exit leaves a live projection."""
    app = QApplication([])
    registry = Registry()
    registry.register("saving", AnalysisSavingAdapter)
    behavior = MeasureGuiBehavior(
        lambda: (registry, TemplateCatalog(), Loader()),
        clean=True,
        project_root=str(project_root),
    )
    assembly = behavior.assemble(None)
    ctrl, window = assembly.controller, assembly.window
    if not isinstance(ctrl, Controller) or not isinstance(window, MainWindow):
        raise TypeError("Measure composition did not produce its public owners")
    behavior.before_show(assembly)
    tab_id = ctrl.new_tab("saving")
    window.add_tab_widget(tab_id, "saving")
    frontend = RecordingFrontend()
    window.mount_interactive_analysis(tab_id, lambda _env: frontend)
    window.show()
    if frontend.read_only_changes != [False]:
        raise RuntimeError("frontend did not receive the initial Local projection")
    QTimer.singleShot(0, app.quit)
    app.exec()
    ctrl.control_identity.connect("agent", "after-real-qt-exit", "owner")
    late_projection = frontend.read_only_changes != [False]
    ctrl.tab_control.close_tab(tab_id)
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    if late_projection:
        print("late authority projection after application quit", file=sys.stderr)
        return 1
    print("application quit retired authority projection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
