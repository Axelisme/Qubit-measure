"""Set up expanded sections through the editor's public fold controls."""

from qtpy.QtWidgets import QLabel, QWidget

from tests.gui._user_input import mouse_click


def expand_sections(view: QWidget, *paths: str) -> None:
    """Open each named section if folded, leaving existing expanded rows alone."""
    for path in paths:
        title = view.findChild(QLabel, f"form.fold.{path}")
        assert title is not None, path
        if title.text().startswith("▸ "):
            mouse_click(title, title.rect().center())
