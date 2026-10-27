"""RohdeSchwarzSGS100A adapter and its existing Qt control panel."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from qtpy.QtWidgets import QComboBox, QFormLayout, QLabel, QWidget

from zcu_tools.device.sgs100a import RohdeSchwarzSGS100A
from zcu_tools.gui.session.devices.base import BaseDeviceAdapter, DeviceCapabilities
from zcu_tools.gui.session.ui.eval_field import EvalNumericField
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

if TYPE_CHECKING:
    from zcu_tools.gui.session.parameters import ParameterContext


class _SGS100APanel(QWidget):
    """Info + control panel for SGS100A."""

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

        self._freq_field = EvalNumericField(
            minimum=1e6, maximum=20e9, decimals=3, parameter_provider=parameter_provider
        )
        form.addRow("Freq (Hz):", self._freq_field)

        self._power_field = EvalNumericField(
            minimum=-120, maximum=30, decimals=2, parameter_provider=parameter_provider
        )
        form.addRow("Power (dBm):", self._power_field)

    def load(self, info: Any) -> None:
        from zcu_tools.device.sgs100a import RohdeSchwarzSGS100AInfo

        assert isinstance(info, RohdeSchwarzSGS100AInfo)
        self._type_label.setText(info.type)
        self._address_label.setText(info.address)
        self._output_combo.setCurrentText(info.output)
        self._freq_field.load_direct(info.freq_Hz)
        self._power_field.load_direct(info.power_dBm)

    def dispose(self) -> None:
        """Retire field callbacks at terminal teardown, not on hide/reopen.

        Repeated calls are safe; cleanup failures propagate for retry before
        deleting Qt children. Success prevents further field callback delivery.
        """
        self._freq_field.dispose()
        self._power_field.dispose()

    def reset_eval_fields(self) -> None:
        """Reset all eval-mode fields to direct mode (R3 device-switch guard)."""
        self._freq_field.reset_to_direct()
        self._power_field.reset_to_direct()

    def set_editing_enabled(self, *, enabled: bool) -> None:
        self._output_combo.setEnabled(enabled)
        self._freq_field.setEnabled(enabled)
        self._power_field.setEnabled(enabled)

    def read(self) -> Any:
        return {
            "output": self._output_combo.currentText(),
            "freq_Hz": self._freq_field.read_raw(),
            "power_dBm": self._power_field.read_raw(),
        }


class SGS100AAdapter(BaseDeviceAdapter):
    """Construct RohdeSchwarzSGS100A drivers and provide their dedicated panel."""

    driver_cls = RohdeSchwarzSGS100A
    panel_cls = _SGS100APanel
    capabilities = DeviceCapabilities(bias_source=False, user_connectable=True)

    def connect(self, address: str) -> RohdeSchwarzSGS100A:
        """Open a VISA driver at address; backend/driver errors propagate."""
        import pyvisa

        return RohdeSchwarzSGS100A(address, pyvisa.ResourceManager())
