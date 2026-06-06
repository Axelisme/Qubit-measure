"""SetupDialog — build the run prerequisites (MockSoc, or a real ZCU + YOKO).

Mirrors measure-gui's setup format (grouped Project / ZCU / Flux-device sections
+ a "Use MockSoc" toggle that greys out the remote fields), but without any
context concept — autofluxdep has no per-flux contexts, only a single Setup that
wires the soc + flux device + predictor for the whole sweep.

On OK the dialog builds a ``SetupRequest`` and calls ``Controller.setup``:
- **Mock** (default): MockSoc + a FakeDevice flux board + a SimplePredictor.
- **Real**: ``make_soc_proxy(ip, port)`` + a YOKOGS200 at the given address +
  a FluxoniumPredictor loaded from the project's params.json.

A real connection that fails surfaces the error inline and keeps the dialog open
(Fast Fail), so the user can fix the address / path and retry.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.state import SetupRequest

logger = logging.getLogger(__name__)


class SetupDialog(QDialog):
    """Build run resources from a MockSoc or a real ZCU + YOKO + predictor."""

    def __init__(self, ctrl: Controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self.setWindowTitle("Setup")
        self.resize(460, 420)

        root = QVBoxLayout(self)

        # Use MockSoc — greys out the ZCU + flux-device fields when ticked
        self._mock = QCheckBox("Use MockSoc + FakeDevice (offline, no hardware)")
        self._mock.setChecked(True)
        self._mock.stateChanged.connect(self._on_mock_toggled)
        root.addWidget(self._mock)

        # Project (params.json drives the FluxoniumPredictor)
        proj = QGroupBox("Project")
        pform = QFormLayout(proj)
        self._chip = QLineEdit("Q5_2D")
        self._qub = QLineEdit("Q1")
        pform.addRow("chip_name", self._chip)
        pform.addRow("qub_name", self._qub)
        params_row = QHBoxLayout()
        self._params = QLineEdit()
        self._params.setPlaceholderText("/path/to/result/params.json")
        params_row.addWidget(self._params)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse_params)
        params_row.addWidget(browse)
        pform.addRow("params.json", params_row)
        root.addWidget(proj)

        # ZCU216 (real soc connection)
        self._zcu = QGroupBox("ZCU216")
        zform = QFormLayout(self._zcu)
        self._ip = QLineEdit("192.168.10.179")
        self._port = QSpinBox()
        self._port.setRange(1, 65535)
        self._port.setValue(8887)
        zform.addRow("ip", self._ip)
        zform.addRow("port", self._port)
        root.addWidget(self._zcu)

        # Flux device (real YOKOGS200 current source)
        self._flux = QGroupBox("Flux device (YOKOGS200)")
        fform = QFormLayout(self._flux)
        self._flux_addr = QLineEdit()
        self._flux_addr.setPlaceholderText("VISA address, e.g. USB0::...::INSTR")
        fform.addRow("address", self._flux_addr)
        root.addWidget(self._flux)

        self._status = QLabel("")
        root.addWidget(self._status)
        root.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._on_mock_toggled()

    def _on_mock_toggled(self) -> None:
        mock = self._mock.isChecked()
        # under mock, the real-connection groups are informational only
        self._zcu.setEnabled(not mock)
        self._flux.setEnabled(not mock)
        self._set_status(
            "MockSoc: a fake board + FakeDevice flux. Run uses synthetic signals."
            if mock
            else "Real connect: ZCU proxy + YOKOGS200 + predictor from params.json.",
            error=False,
        )

    def _on_browse_params(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select params.json", "", "JSON files (*.json);;All files (*)"
        )
        if path:
            self._params.setText(path)

    def _build_request(self) -> SetupRequest:
        return SetupRequest(
            use_mock=self._mock.isChecked(),
            ip=self._ip.text().strip(),
            port=self._port.value(),
            flux_device_address=self._flux_addr.text().strip(),
            params_path=self._params.text().strip(),
        )

    def _accept(self) -> None:
        request = self._build_request()
        if not request.use_mock and not os.path.isfile(request.params_path):
            # a real setup without a readable params.json still runs (predictor
            # degrades to the stand-in), but warn so it is not a silent surprise
            self._set_status(
                "No readable params.json — predictor will be a linear stand-in.",
                error=False,
            )
        try:
            self._ctrl.setup(request)
        except Exception as exc:  # Fast Fail — keep the dialog open on failure
            logger.warning("setup failed: %s", exc)
            self._set_status(f"Setup failed: {exc}", error=True)
            return
        self.accept()

    def _set_status(self, msg: str, *, error: bool) -> None:
        self._status.setText(msg)
        self._status.setStyleSheet(f"color: {'red' if error else 'green'};")
