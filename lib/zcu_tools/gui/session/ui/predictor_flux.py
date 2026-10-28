"""Detached multi-instrument flux controls for the predictor trial."""

from collections.abc import Callable, Mapping

from qtpy.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QRadioButton,
)

from zcu_tools.gui.lifetime import LifetimeScope
from zcu_tools.gui.session.predictor_trial import PredictorTrial
from zcu_tools.resources.predictor_file import PredictorFlux


class PredictorFluxEditor(QGroupBox):
    """Present trial's frame offset and native drafts without source writes.

    on_edited notifies the owner after text or axis edits. A loaded frame owns
    the device list, periods and units; only offset and values are editable.
    Missing cached values stay blank and block queries until entered.
    """

    def __init__(self, on_edited: Callable[[], None], *, trial: PredictorTrial) -> None:
        super().__init__("Flux frame")
        self._scope = LifetimeScope(self)
        self._rows_scope = LifetimeScope(self)
        self._on_edited = on_edited
        self._trial = trial
        self._values: dict[str, QLineEdit] = {}
        self._axes: dict[str, QRadioButton] = {}
        self._layout = QGridLayout(self)
        self._offset = QLineEdit(self._trial.offset_text)
        self._offset.setObjectName("predictor_offset")
        self._layout.addWidget(QLabel("Offset (Φ0):"), 0, 0)
        self._layout.addWidget(self._offset, 0, 1, 1, 2)
        self._scope.connect(self._offset.textChanged, self._notify)
        self._axis_group = QButtonGroup(self)
        self._hint = QLabel("Load a v2 predictor file to set the flux frame.")
        self._layout.addWidget(self._hint, 1, 0, 1, 3)

    def retire_callbacks(self) -> None:
        """Disconnect row and offset callbacks after the dialog accepts final close."""
        self._rows_scope.close()
        self._scope.close()

    def _notify(self, *_args: object) -> None:
        self._trial.set_offset_text(self._offset.text())
        for name, field in self._values.items():
            self._trial.set_device_text(name, field.text())
        for name, axis in self._axes.items():
            if axis.isChecked():
                self._trial.set_axis_device(name)
        self._on_edited()

    def set_flux(self, flux: PredictorFlux | None, values: Mapping[str, float]) -> None:
        """Replace frame and rows, prefilling known native values, without notification.

        Unknown devices in values are ignored. Missing values remain blank.
        The first device becomes the axis; empty frames have no device axis.
        None clears a previously loaded frame and shows the load-file prompt.
        """
        self._rows_scope.close()
        self._rows_scope = LifetimeScope(self)
        while self._layout.count() > 2:
            item = self._layout.takeAt(2)
            if item is not None and (widget := item.widget()) is not None:
                widget.deleteLater()
        self._values.clear()
        self._axes.clear()
        self._trial.set_flux(flux, values)
        self._offset.blockSignals(True)
        self._offset.setText(self._trial.offset_text)
        self._offset.blockSignals(False)
        if flux is None:
            self._layout.addWidget(
                QLabel("Load a v2 predictor file to set the flux frame."), 1, 0, 1, 3
            )
            return
        for row, name in enumerate(flux.period, start=1):
            axis = QRadioButton(name)
            axis.setObjectName(f"predictor_axis_{name}")
            axis.setToolTip("Use this instrument as the curve's horizontal axis")
            self._axis_group.addButton(axis)
            field = QLineEdit(self._trial.device_text(name))
            field.setObjectName(f"predictor_value_{name}")
            field.setPlaceholderText("enter value")
            self._layout.addWidget(axis, row, 0)
            self._layout.addWidget(field, row, 1)
            self._layout.addWidget(QLabel(flux.unit[name]), row, 2)
            if row == 1:
                axis.setChecked(True)
            self._values[name] = field
            self._axes[name] = axis
            self._rows_scope.connect(field.textChanged, self._notify)
            self._rows_scope.connect(axis.toggled, self._notify)

    def flux(self) -> PredictorFlux:
        """Return the local frame; missing frame or invalid offset raises PredictorInputError."""
        return self._trial.flux()

    def has_frame(self) -> bool:
        """Return whether a file or caller has supplied a detached flux frame."""
        return self._trial.has_frame()

    def known_device_values(self) -> dict[str, float]:
        """Return nonblank native inputs for cache; invalid drafts raise a named error."""
        return self._trial.known_device_values()

    def device_values(self) -> dict[str, float]:
        """Return every native device value; blank/invalid text raises a named error."""
        return self._trial.device_values()

    def axis_device(self) -> str:
        """Return the selected instrument; no selection raises PredictorInputError."""
        return self._trial.axis_device()

    def set_axis_device(self, name: str) -> None:
        """Select a loaded device and notify on change; unknown names raise ValueError."""
        if name not in self._axes:
            raise ValueError(f"Unknown axis device {name!r}")
        self._axes[name].setChecked(True)

    def set_offset(self, offset: float) -> None:
        """Set local Phi0 offset text, notifying the owner on change."""
        self._trial.set_offset(offset)
        self.refresh_offset()

    def refresh_offset(self) -> None:
        """Present the model's Phi0 offset text, notifying the owner if it changed.

        Call after model calibration, before publishing a success status. The
        synchronous text-change callback keeps normal edit/debounce/cache order.
        """
        self._offset.setText(self._trial.offset_text)

    def set_device_value(self, name: str, value: float, *, notify: bool = True) -> None:
        """Set one native value; notify=False only moves the marker without replotting.

        Unknown names raise KeyError. The owner settles table/cache state after
        a silent update, avoiding a full curve query on every mouse motion.
        """
        self._trial.set_device_value(name, value)
        field = self._values[name]
        field.blockSignals(not notify)
        try:
            field.setText(self._trial.device_text(name))
        finally:
            field.blockSignals(False)
