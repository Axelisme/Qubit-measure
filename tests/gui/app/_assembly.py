"""Typed controller access and Qt teardown for real behavior assemblies."""

from collections.abc import Generator
from contextlib import contextmanager

from qtpy.QtWidgets import QApplication, QWidget
from zcu_tools.gui.runtime import GuiRuntimeBehavior


@contextmanager
def assembled_controller[C](
    behavior: GuiRuntimeBehavior, controller_type: type[C], *, qapp: QApplication
) -> Generator[C]:
    """Assemble without a control socket and dispose the real Qt window on exit."""
    assembly = behavior.assemble(None)
    window = assembly.window
    if not isinstance(window, QWidget):
        raise TypeError("GUI composition must supply a QWidget")
    try:
        controller = assembly.controller
        if not isinstance(controller, controller_type):
            raise TypeError(f"GUI composition must supply {controller_type.__name__}")
        yield controller
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
