"""ConnectionDialog — configure ZCU connection or use MockSoc."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from zcu_tools.gui.controller import Controller


class ConnectionDialog(QDialog):
    """Modal dialog for ZCU connection setup.

    Either connects to a real ZCU (ip/port) or instantiates a MockQickSoc
    for offline testing.
    """

    def __init__(
        self, controller: "Controller", parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self.setWindowTitle("Connection")
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._ip_edit = QLineEdit("192.168.1.1")
        form.addRow("IP address:", self._ip_edit)

        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(9898)
        form.addRow("Port:", self._port_spin)

        self._mock_check = QCheckBox("Use MockSoc (offline, no hardware)")
        self._mock_check.stateChanged.connect(self._on_mock_toggled)

        layout.addLayout(form)
        layout.addWidget(self._mock_check)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel  # type: ignore[attr-defined]
        )
        btn_box.accepted.connect(self._on_accepted)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_mock_toggled(self, state: int) -> None:
        use_mock = bool(state)
        self._ip_edit.setEnabled(not use_mock)
        self._port_spin.setEnabled(not use_mock)

    def _on_accepted(self) -> None:
        use_mock = self._mock_check.isChecked()
        try:
            if use_mock:
                from zcu_tools.program.v2.mocksoc import make_mock_soc, make_mock_soccfg

                soc = make_mock_soc()
                soccfg = make_mock_soccfg()
                logger.info("ConnectionDialog: using MockQickSoc")
            else:
                ip = self._ip_edit.text().strip()
                port = self._port_spin.value()
                # Real connection deferred to hardware-available environment.
                # Import here to avoid hard dependency in offline use.
                from qick import QickSoc  # type: ignore[import-untyped]

                soc = QickSoc(host=ip, port=port)
                soccfg = soc
                logger.info("ConnectionDialog: connected to %s:%d", ip, port)
            self._ctrl.set_connection(soc, soccfg)
            self._set_status("Connected", error=False)
            self.accept()
        except Exception as exc:
            self._set_status(str(exc), error=True)
            logger.warning("ConnectionDialog: connection failed: %r", exc)

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status_label.setText(msg)
        color = "red" if error else "green"
        self._status_label.setStyleSheet(f"color: {color};")
