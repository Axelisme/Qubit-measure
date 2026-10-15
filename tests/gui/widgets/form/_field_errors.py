"""Shared Qt seam for a field's error mark and error-first row tooltip."""

from qtpy.QtWidgets import QLabel, QWidget


def error_widget(view: QWidget, path: str) -> QWidget:
    """Return the input or name label carrying the row's formError property."""
    value = view.findChild(QLabel, f"form.value.{path}")
    assert value is not None, path
    row = value.parentWidget()
    assert row is not None, path
    marked = [
        w for w in row.findChildren(QWidget) if w.property("formError") is not None
    ]
    # Nested model rows carry their own marks; only this row's children count.
    marked = [w for w in marked if w.parentWidget() is row]
    assert len(marked) == 1, (path, marked)
    return marked[0]


def field_error(view: QWidget, path: str) -> str:
    """Read the error paragraph, or empty text for an unmarked field row."""
    widget = error_widget(view, path)
    active = widget.property("formError")
    assert isinstance(active, bool), (path, active)
    return row_tooltip(view, path).split("\n\n", 1)[0] if active else ""


def row_tooltip(view: QWidget, path: str) -> str:
    """Read the tooltip of the row owning this field's error-marked widget."""
    row = error_widget(view, path).parentWidget()
    assert row is not None, path
    return row.toolTip()
