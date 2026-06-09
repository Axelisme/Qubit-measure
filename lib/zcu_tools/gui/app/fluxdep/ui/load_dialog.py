"""LoadSpectrumDialog — pick a file + type + axis orientation with a live preview.

Adds fluxdep's extras (a Type combo and an "Inherit from" combo) onto the shared
``LoadDataDialog`` base, which owns the file row + Browse, the transpose toggle, the
best-effort read and the preview. ``inherit_from`` optionally seeds the new
spectrum's flux alignment from an already-loaded one. The chosen ``transpose_axes``
flag is passed to the Controller only on Load, so State always stores the canonical
(x=flux, y=freq) layout.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from zcu_tools.gui.app.fluxdep.state import SpecType
from zcu_tools.gui.widgets.load_dialog import LoadDataDialog


class LoadRequest(NamedTuple):
    filepath: str
    spec_type: SpecType
    inherit_from: Optional[str]
    transpose_axes: bool


class LoadSpectrumDialog(LoadDataDialog):
    """Modal dialog returning a LoadRequest (or None on cancel)."""

    def __init__(
        self,
        loaded_names: list[str],
        parent: Optional[QWidget] = None,
        start_dir: str = "",
    ) -> None:
        self._loaded_names = loaded_names
        super().__init__(
            window_title="Load spectrum",
            file_placeholder="Choose a spectrum hdf5…",
            browse_caption="Load spectrum",
            no_preview_text="No preview\n(choose a 2D spectrum file)",
            parent=parent,
            start_dir=start_dir,
        )

    # --- construction ----------------------------------------------------

    def _build_options(self, opt_row: QHBoxLayout) -> None:
        # Type + transpose + inherit row.
        opt_row.addWidget(QLabel("Type:"))
        self._type = QComboBox()
        self._type.addItems(["OneTone", "TwoTone"])
        opt_row.addWidget(self._type)

        opt_row.addWidget(self._transpose_btn)

        opt_row.addWidget(QLabel("Inherit from:"))
        self._inherit = QComboBox()
        self._inherit.addItem("(none)", userData=None)
        for n in self._loaded_names:
            self._inherit.addItem(n, userData=n)
        opt_row.addWidget(self._inherit, stretch=1)

    # --- result ----------------------------------------------------------

    def result_request(self) -> Optional[LoadRequest]:
        """The chosen LoadRequest, or None if no file was selected."""
        if not self._filepath:
            return None
        spec_type: SpecType = (
            "OneTone" if self._type.currentText() == "OneTone" else "TwoTone"
        )
        inherit = self._inherit.currentData()
        return LoadRequest(
            filepath=self._filepath,
            spec_type=spec_type,
            inherit_from=inherit,
            transpose_axes=self._transpose,
        )
