"""FakeDevice adapter and its existing Qt control panel."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from qtpy.QtWidgets import QComboBox, QFormLayout, QLabel, QWidget

from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.gui.session.devices.base import BaseDeviceAdapter, DeviceCapabilities
from zcu_tools.gui.session.ui.eval_field import EvalNumericField
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

if TYPE_CHECKING:
    from zcu_tools.gui.session.parameters import ParameterContext


class _FakeDevicePanel(QWidget):
    """Info + control panel for FakeDevice."""

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

        self._output_combo = QComboBox()
        self._output_combo.addItems(["on", "off"])
        form.addRow("Output:", self._output_combo)

        self._value_field = EvalNumericField(
            minimum=-1e9, maximum=1e9, decimals=6, parameter_provider=parameter_provider
        )
        form.addRow("Value:", self._value_field)

        self._rampstep_field = EvalNumericField(
            minimum=1e-9, maximum=1e9, decimals=9, parameter_provider=parameter_provider
        )
        form.addRow("Ramp step:", self._rampstep_field)

    def load(self, info: Any) -> None:
        from zcu_tools.device.fake import FakeDeviceInfo

        assert isinstance(info, FakeDeviceInfo)
        self._type_label.setText(info.type)
        self._address_label.setText(info.address)
        self._output_combo.setCurrentText(info.output)
        self._value_field.load_direct(info.value)
        self._rampstep_field.load_direct(info.rampstep)

    def reset_eval_fields(self) -> None:
        """Reset all eval-mode fields to direct mode.

        Called when a different device is loaded into this panel to prevent an
        expression from persisting into a different device's context (R3).
        """
        self._value_field.reset_to_direct()
        self._rampstep_field.reset_to_direct()

    def set_editing_enabled(self, *, enabled: bool) -> None:
        self._output_combo.setEnabled(enabled)
        self._value_field.setEnabled(enabled)
        self._rampstep_field.setEnabled(enabled)

    def read(self) -> Any:
        return {
            "output": self._output_combo.currentText(),
            "value": self._value_field.read_raw(),
            "rampstep": self._rampstep_field.read_raw(),
        }


class FakeDeviceAdapter(BaseDeviceAdapter):
    """Construct FakeDevice drivers and provide their dedicated panel."""

    driver_cls = FakeDevice
    panel_cls = _FakeDevicePanel
    capabilities = DeviceCapabilities(bias_source=True, user_connectable=False)

    def connect(self, address: str) -> FakeDevice:
        """Retrieve the addressed FakeDevice without resetting its settings or binding."""
        return FakeDevice(address=address)

    def bias_unit(self, info: BaseDeviceInfo | None) -> str | None:
        """Return this instrument's cached unit, or None for mismatched Info."""
        return info.value_unit if isinstance(info, FakeDeviceInfo) else None
