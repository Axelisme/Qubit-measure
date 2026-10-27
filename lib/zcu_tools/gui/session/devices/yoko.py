"""YOKOGS200 adapter and its existing Qt control panel."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from qtpy.QtWidgets import QComboBox, QFormLayout, QLabel, QWidget

from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.device.yoko import YOKOGS200, YOKOGS200Info
from zcu_tools.gui.session.devices.base import BaseDeviceAdapter, DeviceCapabilities
from zcu_tools.gui.session.ui.eval_field import EvalNumericField
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

if TYPE_CHECKING:
    from zcu_tools.gui.session.parameters import ParameterContext


class _YOKOGS200Panel(QWidget):
    """Info + control panel for YOKOGS200."""

    def __init__(
        self,
        parameter_provider: Callable[[], ParameterContext | None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        form = QFormLayout(self)

        self._type_label = QLabel()
        make_label_selectable(self._type_label)
        form.addRow("Type:", self._type_label)

        self._address_label = QLabel()
        make_label_selectable(self._address_label)
        form.addRow("Address:", self._address_label)

        self._mode_label = QLabel()
        make_label_selectable(self._mode_label)
        form.addRow("Mode:", self._mode_label)

        self._output_combo = QComboBox()
        self._output_combo.addItems(["on", "off"])
        form.addRow("Output:", self._output_combo)

        self._value_field = EvalNumericField(
            minimum=-1e9, maximum=1e9, decimals=6, parameter_provider=parameter_provider
        )
        form.addRow("Value:", self._value_field)

    def load(self, info: Any) -> None:
        from zcu_tools.device.yoko import YOKOGS200Info

        assert isinstance(info, YOKOGS200Info)
        self._type_label.setText(info.type)
        self._address_label.setText(info.address)
        self._mode_label.setText(info.mode)
        self._output_combo.setCurrentText(info.output)
        self._value_field.load_direct(info.value)

    def dispose(self) -> None:
        """Retire field callbacks at terminal teardown, not on hide/reopen.

        Repeated calls are safe; cleanup failures propagate for retry before
        deleting Qt children. Success prevents further field callback delivery.
        """
        self._value_field.dispose()

    def reset_eval_fields(self) -> None:
        """Reset all eval-mode fields to direct mode (R3 device-switch guard)."""
        self._value_field.reset_to_direct()

    def set_editing_enabled(self, *, enabled: bool) -> None:
        self._output_combo.setEnabled(enabled)
        self._value_field.setEnabled(enabled)

    def read(self) -> Any:
        return {
            "output": self._output_combo.currentText(),
            "value": self._value_field.read_raw(),
        }


class YOKOGS200Adapter(BaseDeviceAdapter):
    """Construct YOKOGS200 drivers and provide their dedicated panel."""

    driver_cls = YOKOGS200
    panel_cls = _YOKOGS200Panel
    capabilities = DeviceCapabilities(bias_source=True, user_connectable=True)

    def connect(self, address: str) -> YOKOGS200:
        """Open a VISA driver at address; backend/driver errors propagate."""
        import pyvisa

        return YOKOGS200(address, pyvisa.ResourceManager())

    def bias_unit(self, info: BaseDeviceInfo | None) -> str | None:
        """Return this instrument's cached unit, or None for mismatched Info."""
        return info.value_unit if isinstance(info, YOKOGS200Info) else None
