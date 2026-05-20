"""PredictorDialog — load FluxoniumPredictor from params.json and predict frequencies."""

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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from zcu_tools.gui.controller import Controller


class PredictorDialog(QDialog):
    """Modal dialog for loading a FluxoniumPredictor and predicting frequencies."""

    def __init__(
        self, controller: "Controller", parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self.setWindowTitle("Predictor")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # ── Load predictor ────────────────────────────────────────────────
        load_group = QGroupBox("Load predictor")
        load_form = QFormLayout(load_group)

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("/path/to/params.json")
        path_row.addWidget(self._path_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_file)
        path_row.addWidget(browse_btn)
        load_form.addRow("params.json:", path_row)

        self._flux_bias_spin = QDoubleSpinBox()
        self._flux_bias_spin.setRange(-1e6, 1e6)
        self._flux_bias_spin.setDecimals(6)
        self._flux_bias_spin.setValue(0.0)
        load_form.addRow("flux_bias:", self._flux_bias_spin)

        load_btn_row = QHBoxLayout()
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._on_accepted)
        load_btn_row.addWidget(load_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._on_clear)
        load_btn_row.addWidget(clear_btn)
        load_form.addRow("", load_btn_row)

        layout.addWidget(load_group)

        # ── Predict frequency ─────────────────────────────────────────────
        predict_group = QGroupBox("Predict frequency")
        predict_form = QFormLayout(predict_group)

        self._predict_value_spin = QDoubleSpinBox()
        self._predict_value_spin.setRange(-1e6, 1e6)
        self._predict_value_spin.setDecimals(6)
        self._predict_value_spin.setValue(0.0)
        predict_form.addRow("Flux value (A):", self._predict_value_spin)

        transition_row = QHBoxLayout()
        self._from_spin = QSpinBox()
        self._from_spin.setRange(0, 20)
        self._from_spin.setValue(0)
        self._to_spin = QSpinBox()
        self._to_spin.setRange(0, 20)
        self._to_spin.setValue(1)
        transition_row.addWidget(QLabel("from"))
        transition_row.addWidget(self._from_spin)
        transition_row.addWidget(QLabel("to"))
        transition_row.addWidget(self._to_spin)
        predict_form.addRow("Transition:", transition_row)

        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(self._on_predict_clicked)
        predict_form.addRow("", predict_btn)

        self._predict_result_label = QLabel("—")
        self._predict_result_label.setStyleSheet("font-weight: bold;")
        predict_form.addRow("Result:", self._predict_result_label)

        layout.addWidget(predict_group)

        # ── status label ──────────────────────────────────────────────────
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)  # type: ignore[attr-defined]
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # pre-fill with current predictor state
        info = controller.get_predictor_info()
        if info is not None:
            if info["path"] is not None:
                self._path_edit.setText(info["path"])
            self._flux_bias_spin.setValue(info["flux_bias"])
            self._set_status("Currently loaded", error=False)

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
            self._ctrl.set_predictor(predictor, path=path)
            self._set_status("Predictor loaded", error=False)
            logger.info("PredictorDialog: loaded path=%r", path)
        except Exception as exc:
            self._set_status(str(exc), error=True)
            logger.warning("PredictorDialog: load failed: %r", exc)

    def _on_clear(self) -> None:
        self._ctrl.set_predictor(None)
        self._predict_result_label.setText("—")
        self._set_status("Predictor cleared")
        logger.info("PredictorDialog: predictor cleared")

    def _on_predict_clicked(self) -> None:
        predictor = self._ctrl._state.exp_context.predictor
        if predictor is None:
            self._set_status("No predictor loaded — load one first", error=True)
            return
        value = self._predict_value_spin.value()
        from_lvl = self._from_spin.value()
        to_lvl = self._to_spin.value()
        transition = (from_lvl, to_lvl)
        try:
            freq = predictor.predict_freq(value, transition=transition)
            self._predict_result_label.setText(f"{freq:.4f} MHz")
            self._set_status(
                f"Predicted ({from_lvl},{to_lvl}) @ {value:.6g}: {freq:.4f} MHz"
            )
            logger.info(
                "PredictorDialog: predict value=%r transition=%r → %.4f MHz",
                value,
                transition,
                freq,
            )
        except Exception as exc:
            self._set_status(str(exc), error=True)
            self._predict_result_label.setText("error")
            logger.warning("PredictorDialog: predict failed: %r", exc)

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status_label.setText(msg)
        color = "red" if error else "green"
        self._status_label.setStyleSheet(f"color: {color};")
