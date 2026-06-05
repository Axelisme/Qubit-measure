"""SetupDialog — build the run prerequisites (prototype: MockSoc + FakeDevice).

A single dialog with a "Use MockSoc" path (the only one wired in the prototype):
ticking it and pressing OK builds a MockSoc + soccfg + a FakeDevice flux board
and stores them in State via ``Controller.setup``. The project / ZCU-ip / device
fields are shown for shape (matching measure-gui's setup_dialog) but, with mock
ticked, are informational — real make_soc_proxy / use_flux wiring is Phase B.
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.autofluxdep.controller import Controller


class SetupDialog(QDialog):
    """Build run resources. Prototype: MockSoc + FakeDevice on OK."""

    def __init__(self, ctrl: Controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self.setWindowTitle("Setup")
        self.resize(420, 360)

        root = QVBoxLayout(self)

        # Use MockSoc — the only path wired in the prototype
        self._mock = QCheckBox("Use MockSoc + FakeDevice (offline, no hardware)")
        self._mock.setChecked(True)
        self._mock.stateChanged.connect(self._on_mock_toggled)
        root.addWidget(self._mock)

        # Project (informational under mock)
        proj = QGroupBox("Project")
        pform = QFormLayout(proj)
        self._chip = QLineEdit("Q5_2D")
        self._qub = QLineEdit("Q1")
        self._flux_label = QLineEdit("051115_2.000mA")
        pform.addRow("chip_name", self._chip)
        pform.addRow("qub_name", self._qub)
        pform.addRow("flux label", self._flux_label)
        root.addWidget(proj)

        # ZCU216 (informational under mock)
        self._zcu = QGroupBox("ZCU216")
        zform = QFormLayout(self._zcu)
        self._ip = QLineEdit("192.168.10.179")
        self._port = QLineEdit("8887")
        zform.addRow("ip", self._ip)
        zform.addRow("port", self._port)
        root.addWidget(self._zcu)

        self._note = QLabel("")
        root.addWidget(self._note)
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
        # under mock, the ZCU fields are informational
        self._zcu.setEnabled(not mock)
        self._note.setText(
            "MockSoc: a fake board + FakeDevice flux. Run uses synthetic signals."
            if mock
            else "Real connect is Phase B; tick MockSoc to proceed in the prototype."
        )

    def _accept(self) -> None:
        if not self._mock.isChecked():
            # prototype: only the mock path is wired
            self._note.setText("Only MockSoc is wired in the prototype — tick it.")
            return
        self._ctrl.setup(use_mock=True)
        self.accept()
