"""LoadSpectrumDialog — pick a file + type + axis orientation with a live preview.

Adds fluxdep's extras (a Type combo and an "Inherit from" combo) onto the shared
``LoadDataDialog`` base, which owns the file row + Browse, the transpose toggle, the
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

import numpy as np
from matplotlib.axes import Axes
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.analysis.flux import ComponentFluxContext
from zcu_tools.analysis.fluxdep import project_native_spectrum
from zcu_tools.analysis.fluxdep.line_picker import (
    HALF_FLUX_COLOR,
    INTEGER_FLUX_COLOR,
)
from zcu_tools.datafile import Container
from zcu_tools.gui.app.fluxdep.services.load import (
    NativeSpectrumSelection,
    read_spectrum_source,
)
from zcu_tools.gui.app.fluxdep.state import SpecType
from zcu_tools.gui.lifetime import LifetimeScope
from zcu_tools.gui.widgets.load_dialog import LoadDataDialog


class LoadRequest(NamedTuple):
    """Chosen path/type, optional alignment seed and source-specific axes.

    filepath is exact; spec_type chooses tooling. inherit_from names the seed.
    transpose_axes applies only to Labber; native inputs use explicit selection.
    """

    filepath: str
    spec_type: SpecType
    inherit_from: str | None
    transpose_axes: bool
    selection: NativeSpectrumSelection | None


class LoadSpectrumDialog(LoadDataDialog):
    """Modal dialog returning a LoadRequest (or None on cancel)."""

    def __init__(
        self,
        inherit_sources: Mapping[str, tuple[float, float]],
        parent: QWidget | None = None,
        start_dir: str = "",
        *,
        flux_context: ComponentFluxContext | None = None,
        context_error: str | None = None,
    ) -> None:
        """Build the dialog over the spectra a new one may inherit from.

        inherit_sources maps eligible aligned or seeded spectrum names, in list
        order, to the (flux_half, flux_int) device positions it would seed; choosing it in
        "Inherit from" draws those lines on the preview. start_dir is where
        Browse opens. flux_context supplies native device choices/default/unit
        without parsing component fields. context_error is a visible provider
        failure; native loading remains possible only as unnamed_device.
        """
        self._scope = LifetimeScope(self)
        self._native_container: Container | None = None
        self._flux_context = flux_context
        self._context_error = context_error
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
        rows = QVBoxLayout()
        opt_row.addLayout(rows)
        basic = QHBoxLayout()
        rows.addLayout(basic)
        basic.addWidget(QLabel("Type:"))
        self._type = QComboBox()
        self._type.addItems(["OneTone", "TwoTone"])
        basic.addWidget(self._type)

        basic.addWidget(self._transpose_btn)

        basic.addWidget(QLabel("Inherit from:"))
        self._inherit = QComboBox()
        self._inherit.addItem("(none)", userData=None)
        for n in self._inherit_sources:
            self._inherit.addItem(n, userData=n)
        self._scope.connect(self._inherit.currentIndexChanged, self._render_preview)
        basic.addWidget(self._inherit, stretch=1)

        native = QHBoxLayout()
        rows.addLayout(native)
        self._variable = QComboBox()
        self._variable.setObjectName("native_variable")
        self._device_axis = QComboBox()
        self._device_axis.setObjectName("native_device_axis")
        self._device = QComboBox()
        self._device.setObjectName("native_device")
        self._device.addItem("unnamed_device")
        self._unit_label = QLabel("native")
        for caption, combo in (
            ("Variable:", self._variable),
            ("Device axis:", self._device_axis),
            ("Device:", self._device),
        ):
            native.addWidget(QLabel(caption))
            native.addWidget(combo, stretch=1)
        native.addWidget(self._unit_label)
        self._native_error = QLabel()
        self._native_error.setObjectName("native_input_error")
        self._native_error.setWordWrap(True)
        rows.addWidget(self._native_error)
        self._variable.setEnabled(False)
        self._device_axis.setEnabled(False)
        self._device.setEnabled(False)
        self._scope.connect(self._variable.currentIndexChanged, self._choose_variable)
        self._scope.connect(self._device_axis.currentIndexChanged, self._native_preview)
        self._scope.connect(self._device.currentIndexChanged, self._show_device_unit)

    def _read_source(
        self, filepath: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        self._native_container = None
        self._native_error.clear()
        try:
            source = read_spectrum_source(filepath)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._variable.clear()
            self._device_axis.clear()
            self._device.clear()
            self._device.addItem("unnamed_device")
            for control in (
                self._variable,
                self._device_axis,
                self._device,
                self._transpose_btn,
            ):
                control.setEnabled(False)
            self._unit_label.setText("native")
            self._native_error.setText(str(error))
            if self._ok_button is not None:
                self._ok_button.setEnabled(False)
            return None
        native = isinstance(source, Container)
        self._variable.setEnabled(native)
        self._device_axis.setEnabled(native)
        self._device.setEnabled(native)
        self._transpose_btn.setEnabled(not native)
        if native:
            self._transpose_btn.setChecked(False)
        self._variable.blockSignals(True)
        self._variable.clear()
        self._device.clear()
        self._device.addItem("unnamed_device")
        if not native:
            self._variable.blockSignals(False)
            self._device_axis.clear()
            self._unit_label.setText("native")
            return self._read_best_effort(filepath)
        self._native_container = source
        self._variable.addItem("(choose variable)", userData=None)
        for name in source.vars:
            self._variable.addItem(name, userData=name)
        candidates = [
            name
            for name, item in source.vars.items()
            if item.values.ndim == 2 and np.iscomplexobj(item.values)
        ]
        if len(candidates) == 1:
            self._variable.setCurrentText(candidates[0])
        self._variable.blockSignals(False)
        context = self._flux_context
        if context is not None and context.frame is not None:
            self._device.clear()
            self._device.addItem("(choose device)", userData=None)
            self._device.addItems(list(context.frame.devices))
            if (
                context.default_device is not None
                and context.default_device in context.frame.devices
            ):
                self._device.setCurrentText(context.default_device)
        self._choose_variable()
        return self._raw

    def _show_device_unit(self) -> None:
        context = self._flux_context
        self._unit_label.setText(
            context.units.get(self._device.currentText(), "native")
            if context is not None
            else "native"
        )
        if self._native_container is not None:
            self._native_preview()

    def _choose_variable(self) -> None:
        source = self._native_container
        if source is None:
            return
        self._device_axis.blockSignals(True)
        self._device_axis.clear()
        self._device_axis.addItem("(choose axis)", userData=None)
        name = self._variable.currentData()
        if name is not None:
            variable = source.vars[name]
            for axis in variable.axes:
                self._device_axis.addItem(axis.name, userData=axis.name)
            frequencies = [
                axis
                for axis in variable.axes
                if axis.unit in ("Hz", "kHz", "MHz", "GHz")
            ]
            if len(variable.axes) == 2 and len(frequencies) == 1:
                device_axis = next(
                    axis for axis in variable.axes if axis.name != frequencies[0].name
                )
                self._device_axis.setCurrentText(device_axis.name)
        self._device_axis.blockSignals(False)
        self._native_preview()

    def _native_preview(self) -> None:
        source = self._native_container
        if source is None:
            return
        self._raw = None
        valid = False
        variable = self._variable.currentData()
        axis = self._device_axis.currentData()
        message = "Choose a variable, device axis and device."
        if variable is not None and axis is not None:
            try:
                spectrum = project_native_spectrum(
                    source, variable=variable, device_axis=axis
                )
                self._raw = (spectrum.signals, spectrum.dev_values, spectrum.freqs_mhz)
                if self._flux_context is None or self._flux_context.frame is None:
                    self._unit_label.setText(spectrum.unit)
                valid = self._device.currentText() != "(choose device)"
                message = "" if valid else message
            except ValueError as error:
                message = str(error)
        if self._context_error:
            message = self._context_error + ("\n" + message if message else "")
        self._native_error.setText(message)
        if self._ok_button is not None:
            self._ok_button.setEnabled(valid)
        self._render_preview()

    def done(self, a0: int) -> None:
        """Retire selection callbacks before closing the dialog."""
        self._scope.close()
        super().done(a0)

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
        """Return the selected request, or None without required path/native axes.

        Only enabled OK submits it. Native selections preserve variable, axis and
        device names; the disabled transpose toggle never overrides those axes.
        """
        if not self._filepath:
            return None
        spec_type: SpecType = (
            "OneTone" if self._type.currentText() == "OneTone" else "TwoTone"
        )
        inherit = self._inherit.currentData()
        variable = self._variable.currentData()
        axis = self._device_axis.currentData()
        selection = None
        if self._native_container is not None:
            if not isinstance(variable, str) or not isinstance(axis, str):
                return None
            selection = NativeSpectrumSelection(
                variable, axis, self._device.currentText()
            )
        return LoadRequest(
            filepath=self._filepath,
            spec_type=spec_type,
            inherit_from=inherit,
            transpose_axes=self._transpose if self._native_container is None else False,
            selection=selection,
        )
