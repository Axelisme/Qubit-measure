"""Capture the focused document set in session and application fixtures."""

from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.types import DocumentSetCapture
from zcu_tools.resources.entry.schema import ParameterDocument


def capture_focus(control: ContextControlPort) -> DocumentSetCapture:
    """Read the currently focused global/point set without changing selection."""
    return control.read_context(point=control.current_point_label())


def inspect_focus(control: ContextControlPort) -> ParameterDocument:
    """Inspect the currently focused document without changing selection."""
    return control.inspect_document(point=control.current_point_label())
