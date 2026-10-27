"""Fluxdep Type/Inherit controls and alignment overlay over the shared chooser."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from matplotlib.axes import Axes
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from zcu_tools.analysis.flux import ComponentFluxContext
from zcu_tools.analysis.fluxdep.line_picker import HALF_FLUX_COLOR, INTEGER_FLUX_COLOR
from zcu_tools.gui.app.analysis_project.native import NativeSpectrumSelection
from zcu_tools.gui.app.analysis_project.native_dialog import (
    NativeDeviceChoices,
    NativeSpectrumDialog,
)
from zcu_tools.gui.app.fluxdep.state import SpecType


class LoadRequest(NamedTuple):
    """Exact source path, tooling type, optional alignment seed and native choice.

    transpose_axes applies only to Labber. selection explicitly names native
    variable, axis and device; inherit_from names a previously seeded spectrum.
    """

    filepath: str
    spec_type: SpecType
    inherit_from: str | None
    transpose_axes: bool
    selection: NativeSpectrumSelection | None


class LoadSpectrumDialog(NativeSpectrumDialog):
    """Return a source request with fluxdep type and optional alignment inheritance."""

    def __init__(
        self,
        inherit_sources: Mapping[str, tuple[float, float]],
        parent: QWidget | None = None,
        start_dir: str = "",
        *,
        flux_context: ComponentFluxContext | None = None,
        context_error: str | None = None,
    ) -> None:
        """Offer seeded/aligned sources and caller-projected device options.

        inherit_sources maps names to half/integer positions for the preview.
        flux_context supplies calibrated devices/default/units. Without a frame
        only unnamed_device is available. context_error remains visible.
        """
        self._inherit_sources = dict(inherit_sources)
        choices = NativeDeviceChoices(
            units=flux_context.units
            if flux_context is not None and flux_context.frame is not None
            else {},
            default_device=flux_context.default_device
            if flux_context is not None
            else None,
            error=context_error,
        )
        super().__init__(choices, parent, start_dir)

    def _build_source_options(self, row: QHBoxLayout) -> None:
        row.addWidget(QLabel("Type:"))
        self._type = QComboBox()
        self._type.addItems(["OneTone", "TwoTone"])
        row.addWidget(self._type)
        row.addWidget(self._axis_combo)
        row.addWidget(QLabel("Inherit from:"))
        self._inherit = QComboBox()
        self._inherit.addItem("(none)", userData=None)
        for name in self._inherit_sources:
            self._inherit.addItem(name, userData=name)
        self._scope.connect(
            self._inherit.currentIndexChanged, lambda _index: self._render_preview()
        )
        row.addWidget(self._inherit, stretch=1)

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

    def result_request(self) -> LoadRequest | None:
        """Return the chosen source/type/seed, or None without required fields."""
        if not self._filepath:
            return None
        selection = self.native_selection()
        if self.is_native_source and selection is None:
            return None
        spec_type: SpecType = (
            "OneTone" if self._type.currentText() == "OneTone" else "TwoTone"
        )
        return LoadRequest(
            self._filepath,
            spec_type,
            self._inherit.currentData(),
            self._transpose if not self.is_native_source else False,
            selection,
        )
