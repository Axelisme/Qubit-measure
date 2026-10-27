"""LoadSpectrumDialog — pick a file + type + axis orientation with a live preview.

Adds fluxdep's extras (a Type combo and an "Inherit from" combo) onto the shared
``LoadDataDialog`` base, which owns the editable file row + Browse, the axis-mapping dropdown, the
best-effort read and the preview. ``inherit_from`` optionally seeds the new
spectrum's flux alignment from an already-loaded one; while one is chosen the
preview draws its half/integer lines, so the inherited alignment can be judged
before loading. The chosen ``transpose_axes``
flag is passed to the Controller only on Load, so State always stores the canonical
(x=flux, y=freq) layout.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from matplotlib.axes import Axes
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from zcu_tools.analysis.fluxdep.line_picker import (
    HALF_FLUX_COLOR,
    INTEGER_FLUX_COLOR,
)
from zcu_tools.gui.app.fluxdep.state import SpecType
from zcu_tools.gui.widgets.load_dialog import LoadDataDialog


class LoadRequest(NamedTuple):
    filepath: str
    spec_type: SpecType
    inherit_from: str | None
    transpose_axes: bool


class LoadSpectrumDialog(LoadDataDialog):
    """Modal dialog returning a LoadRequest (or None on cancel)."""

    def __init__(
        self,
        inherit_sources: Mapping[str, tuple[float, float]],
        parent: QWidget | None = None,
        start_dir: str = "",
    ) -> None:
        """Build the dialog over the spectra a new one may inherit from.

        inherit_sources maps eligible aligned or seeded spectrum names, in list
        order, to the (flux_half, flux_int) device positions it would seed; choosing it in
        "Inherit from" draws those lines on the preview. start_dir is where
        Browse opens.
        """
        self._inherit_sources = dict(inherit_sources)
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

        opt_row.addWidget(self._axis_combo)

        opt_row.addWidget(QLabel("Inherit from:"))
        self._inherit = QComboBox()
        self._inherit.addItem("(none)", userData=None)
        for n in self._inherit_sources:
            self._inherit.addItem(n, userData=n)
        self._scope.connect(
            self._inherit.currentIndexChanged, lambda _index: self._render_preview()
        )
        opt_row.addWidget(self._inherit, stretch=1)

    def _draw_preview_overlay(self, ax: Axes) -> None:
        """Draw the chosen inherit source's half/integer lines, as its picker does.

        The lines mark device positions: vertical on the original x axis,
        horizontal on y after transposing. The spectrum keeps its extent, so
        positions outside the displayed device range remain clipped.
        """
        source = self._inherit.currentData()
        if source is None:
            return
        flux_half, flux_int = self._inherit_sources[source]
        if self._transpose:
            extent = ax.get_ylim()
            ax.axhline(flux_half, color=HALF_FLUX_COLOR, linestyle="--", linewidth=1)
            ax.axhline(flux_int, color=INTEGER_FLUX_COLOR, linestyle="--", linewidth=1)
            ax.set_ylim(extent)
        else:
            extent = ax.get_xlim()
            ax.axvline(flux_half, color=HALF_FLUX_COLOR, linestyle="--", linewidth=1)
            ax.axvline(flux_int, color=INTEGER_FLUX_COLOR, linestyle="--", linewidth=1)
            ax.set_xlim(extent)

    # --- result ----------------------------------------------------------

    def result_request(self) -> LoadRequest | None:
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
