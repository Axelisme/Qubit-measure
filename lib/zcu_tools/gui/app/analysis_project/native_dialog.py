"""Shared native spectrum chooser; callers supply device identities and units."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from zcu_tools.analysis.fluxdep import project_native_spectrum
from zcu_tools.datafile import Container
from zcu_tools.gui.lifetime import LifetimeScope
from zcu_tools.gui.widgets.load_dialog import LoadDataDialog

from .native import NativeSpectrumSelection, read_spectrum_source


@dataclass(frozen=True)
class NativeDeviceChoices:
    """Caller-selected device options, without component or predictor access.

    units maps selectable instrument names to native units. An empty mapping
    offers only unnamed_device, using the source axis unit. default_device is
    preselected only when in units. error is an optional visible caller failure.
    """

    units: Mapping[str, str]
    default_device: str | None = None
    error: str | None = None


class NativeSpectrumDialog(LoadDataDialog):
    """File preview and explicit native selection over caller-supplied devices.

    Unique 2-D complex variables and unambiguous frequency/device axes are only
    preselected; all native choices remain editable. Labber retains transpose
    and unnamed_device. Invalid native reads/selections disable submission.
    Subclasses add domain controls in _build_source_options and return their
    own result_request, using native_selection for the explicit native fields.
    """

    def __init__(
        self,
        choices: NativeDeviceChoices,
        parent: QWidget | None = None,
        start_dir: str = "",
    ) -> None:
        self._scope = LifetimeScope(self)
        self._native_container: Container | None = None
        self._device_choices = choices
        super().__init__(
            window_title="Load spectrum",
            file_placeholder="Choose a spectrum hdf5…",
            browse_caption="Load spectrum",
            no_preview_text="No preview\n(choose a 2D spectrum file)",
            parent=parent,
            start_dir=start_dir,
        )

    def _build_options(self, opt_row: QHBoxLayout) -> None:
        rows = QVBoxLayout()
        opt_row.addLayout(rows)
        basic = QHBoxLayout()
        rows.addLayout(basic)
        self._build_source_options(basic)

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

    def _build_source_options(self, row: QHBoxLayout) -> None:
        """Place the Labber axis mapping; subclasses may add domain options."""
        row.addWidget(self._axis_combo)

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
                self._axis_combo,
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
        self._axis_combo.setEnabled(not native)
        if native:
            self._axis_combo.setCurrentIndex(0)
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
        context = self._device_choices
        if context.units:
            self._device.clear()
            self._device.addItem("(choose device)", userData=None)
            self._device.addItems(list(context.units))
            if (
                context.default_device is not None
                and context.default_device in context.units
            ):
                self._device.setCurrentText(context.default_device)
        self._choose_variable()
        return self._raw

    def _show_device_unit(self) -> None:
        context = self._device_choices
        self._unit_label.setText(
            context.units.get(self._device.currentText(), "native")
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
                if not self._device_choices.units:
                    self._unit_label.setText(spectrum.unit)
                valid = self._device.currentText() != "(choose device)"
                message = "" if valid else message
            except ValueError as error:
                message = str(error)
        if self._device_choices.error:
            message = self._device_choices.error + ("\n" + message if message else "")
        self._native_error.setText(message)
        if self._ok_button is not None:
            self._ok_button.setEnabled(valid)
        self._render_preview()

    def done(self, a0: int) -> None:
        """Retire selection callbacks before closing the dialog."""
        self._scope.close()
        super().done(a0)

    def native_selection(self) -> NativeSpectrumSelection | None:
        """Return explicit native choices, or None for Labber/unselected fields.

        Native callers must reject None when is_native_source is true. This
        method never infers variable, axis or device at submission time.
        """
        if self._native_container is None:
            return None
        variable = self._variable.currentData()
        axis = self._device_axis.currentData()
        if not isinstance(variable, str) or not isinstance(axis, str):
            return None
        if self._device.currentText() == "(choose device)":
            return None
        return NativeSpectrumSelection(variable, axis, self._device.currentText())

    @property
    def is_native_source(self) -> bool:
        """Whether the current successfully decoded source is a Container."""
        return self._native_container is not None
