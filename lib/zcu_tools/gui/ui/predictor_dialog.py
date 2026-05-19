"""PredictorDialog — load FluxoniumPredictor from params.json."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from zcu_tools.gui.controller import Controller


class PredictorDialog(QDialog):
    """Modal dialog for loading a FluxoniumPredictor."""

    def __init__(
        self, controller: "Controller", parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self.setWindowTitle("Predictor")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("/path/to/params.json")
        path_row.addWidget(self._path_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_file)
        path_row.addWidget(browse_btn)
        form.addRow("params.json:", path_row)

        self._flux_bias_spin = QDoubleSpinBox()
        self._flux_bias_spin.setRange(-1e6, 1e6)
        self._flux_bias_spin.setDecimals(6)
        self._flux_bias_spin.setValue(0.0)
        form.addRow("flux_bias:", self._flux_bias_spin)

        layout.addLayout(form)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        clear_btn = QPushButton("Clear predictor")
        clear_btn.clicked.connect(self._on_clear)
        layout.addWidget(clear_btn)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel  # type: ignore[attr-defined]
        )
        btn_box.accepted.connect(self._on_accepted)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select params.json", "", "JSON files (*.json);;All files (*)"
        )
        if path:
            self._path_edit.setText(path)

    def _on_accepted(self) -> None:
        path = self._path_edit.text().strip()
        flux_bias = self._flux_bias_spin.value()
        try:
            from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

            predictor = FluxoniumPredictor.from_file(path, flux_bias=flux_bias)
            self._ctrl.set_predictor(predictor)
            self._set_status("Predictor loaded", error=False)
            logger.info("PredictorDialog: loaded path=%r", path)
            self.accept()
        except Exception as exc:
            self._set_status(str(exc), error=True)
            logger.warning("PredictorDialog: load failed: %r", exc)

    def _on_clear(self) -> None:
        self._ctrl.set_predictor(None)
        self._set_status("Predictor cleared")
        logger.info("PredictorDialog: predictor cleared")

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status_label.setText(msg)
        color = "red" if error else "green"
        self._status_label.setStyleSheet(f"color: {color};")
