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
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


if TYPE_CHECKING:
    from zcu_tools.gui.controller import Controller


class ConnectionDialog(QDialog):
    """Modal dialog for ZCU connection setup.

    Either connects to a real ZCU (ip/port) or instantiates a MockQickSoc
    for offline testing.  After a successful connection, displays the soccfg
    description so the user can verify channel assignments.
    """

    def __init__(
        self, controller: "Controller", parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self.setWindowTitle("Connection")
        self.setMinimumWidth(420)

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

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        layout.addWidget(self._connect_btn)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # soccfg description area — hidden until connection succeeds
        self._cfg_text = QPlainTextEdit()
        self._cfg_text.setReadOnly(True)
        self._cfg_text.setMinimumHeight(200)
        self._cfg_text.setLineWrapMode(QPlainTextEdit.NoWrap)  # type: ignore[attr-defined]
        self._cfg_text.setVisible(False)
        layout.addWidget(self._cfg_text)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)  # type: ignore[attr-defined]
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # show current soccfg if already connected
        self._maybe_show_current_cfg()

    def _maybe_show_current_cfg(self) -> None:
        soccfg = self._ctrl._state.exp_context.soccfg
        if soccfg is not None and hasattr(soccfg, "description"):
            try:
                self._show_cfg(soccfg.description())
                self._set_status("Currently connected", error=False)
            except Exception:
                pass

    def _on_mock_toggled(self, state: int) -> None:
        use_mock = bool(state)
        self._ip_edit.setEnabled(not use_mock)
        self._port_spin.setEnabled(not use_mock)

    def _on_connect_clicked(self) -> None:
        use_mock = self._mock_check.isChecked()
        self._connect_btn.setEnabled(False)
        self._set_status("Connecting…", error=False)
        try:
            if use_mock:
                from zcu_tools.program.v2.mocksoc import make_mock_soc, make_mock_soccfg

                soc = make_mock_soc()
                soccfg = make_mock_soccfg()
                logger.info("ConnectionDialog: using MockQickSoc")
            else:
                ip = self._ip_edit.text().strip()
                port = self._port_spin.value()
                try:
                    from zcu_tools.remote import make_soc_proxy
                except ImportError as e:
                    raise RuntimeError(
                        f"Cannot import ZCU client libraries: {e}\n"
                        "Use MockSoc for offline testing."
                    ) from e
                soc, soccfg = make_soc_proxy(ip, port)
                logger.info("ConnectionDialog: connected to %s:%d", ip, port)

            self._ctrl.set_connection(soc, soccfg)
            self._set_status("Connected", error=False)

            if hasattr(soccfg, "description"):
                self._show_cfg(soccfg.description())

        except Exception as exc:
            self._set_status(str(exc), error=True)
            logger.warning("ConnectionDialog: connection failed: %r", exc)
        finally:
            self._connect_btn.setEnabled(True)

    def _show_cfg(self, text: str) -> None:
        self._cfg_text.setPlainText(text)
        self._cfg_text.setVisible(True)
        self.adjustSize()

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status_label.setText(msg)
        color = "red" if error else "green"
        self._status_label.setStyleSheet(f"color: {color};")
