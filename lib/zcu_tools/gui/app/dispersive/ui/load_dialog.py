"""LoadOnetoneDialog — pick a one-tone hdf5 + axis orientation with a live preview.

dispersive loads a single one-tone spectrum, so it just reuses the shared
``LoadDataDialog`` base (file row + Browse, transpose toggle, best-effort read,
preview) with no extra options. The chosen ``transpose_axes`` flag is passed to the
Controller only on Load, so State always stores the canonical (x=flux, y=freq)
layout.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QHBoxLayout,
    QLabel,
    QWidget,
)

from zcu_tools.gui.widgets.load_dialog import LoadDataDialog


class LoadOnetoneRequest(NamedTuple):
    filepath: str
    transpose_axes: bool


class LoadOnetoneDialog(LoadDataDialog):
    """Modal dialog returning a LoadOnetoneRequest (or None on cancel)."""

    def __init__(self, parent: QWidget | None = None, start_dir: str = "") -> None:
        super().__init__(
            window_title="Load one-tone spectrum",
            file_placeholder="Choose a one-tone hdf5…",
            browse_caption="Load one-tone",
            no_preview_text="No preview\n(choose a 2D one-tone file)",
            parent=parent,
            start_dir=start_dir,
        )

    # --- construction ----------------------------------------------------

    def _build_options(self, opt_row: QHBoxLayout) -> None:
        # Just the transpose toggle plus a hint — no type / inherit.
        opt_row.addWidget(self._transpose_btn)
        opt_row.addWidget(
            QLabel("(toggle if the preview shows frequency on x instead of flux)")
        )
        opt_row.addStretch(1)

    # --- result ----------------------------------------------------------

    def result_request(self) -> LoadOnetoneRequest | None:
        """The chosen LoadOnetoneRequest, or None if no file was selected."""
        if not self._filepath:
            return None
        return LoadOnetoneRequest(
            filepath=self._filepath, transpose_axes=self._transpose
        )
