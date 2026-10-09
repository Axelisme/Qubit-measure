"""Native file or opened-entry ledger selection, without loading data itself."""

from __future__ import annotations

from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.measure.services.save_control import SavedRunInfo


class RunLoadDialog(QDialog):
    """Choose a native file or one loadable ledger row on the Qt owner thread.

    runs contains ordered SavedRunInfo from the opened entry; no files are read.
    initial_point selects a known point, or all points when empty/unknown.
    start_directory is QFileDialog's initial folder, not a reserved destination.
    Accepted implies selected_path() is nonempty. Reject/cancel returns no path.
    Point/experiment filters are exact; time filter matches recorded_at text.
    Rows without data_path remain visible but cannot be loaded. Paths may have
    disappeared; the existing Load owner reports that failure after selection.
    """

    def __init__(
        self,
        runs: tuple[SavedRunInfo, ...],
        *,
        initial_point: str,
        start_directory: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._runs = runs
        self._visible_runs: tuple[SavedRunInfo, ...] = ()
        self._start_directory = start_directory
        self._selected_path: str | None = None
        self.setWindowTitle("Load Data")
        self.resize(850, 440)
        layout = QVBoxLayout(self)
        layout.addLayout(self._build_filters(initial_point))
        self._table = QTableWidget(0, 5)
        self._table.setObjectName("load_runs")
        self._table.setHorizontalHeaderLabels(
            ["Run", "Point", "Experiment", "Recorded at", "Native data"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table)
        buttons = QHBoxLayout()
        file_button = QPushButton("File…")
        file_button.setObjectName("load_file")
        file_button.clicked.connect(self._open_file)
        self._load = QPushButton("Load selected")
        self._load.setObjectName("load_selected")
        self._load.setEnabled(False)
        self._load.clicked.connect(self._load_selected)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("load_cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(file_button)
        buttons.addStretch()
        buttons.addWidget(self._load)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        self._point.currentIndexChanged.connect(self._refresh)
        self._experiment.currentIndexChanged.connect(self._refresh)
        self._time.textChanged.connect(self._refresh)
        self._table.itemSelectionChanged.connect(self._selection_changed)
        self._refresh()

    def _build_filters(self, initial_point: str) -> QHBoxLayout:
        filters = QHBoxLayout()
        self._point = QComboBox()
        self._point.setObjectName("load_point")
        self._point.addItem("All points", None)
        for point in sorted({run.point for run in self._runs if run.point is not None}):
            self._point.addItem(point, point)
        index = self._point.findData(initial_point)
        if index >= 0:
            self._point.setCurrentIndex(index)
        self._experiment = QComboBox()
        self._experiment.setObjectName("load_experiment")
        self._experiment.addItem("All experiments", None)
        for experiment in sorted({run.experiment for run in self._runs}):
            self._experiment.addItem(experiment, experiment)
        self._time = QLineEdit()
        self._time.setObjectName("load_recorded_at")
        self._time.setPlaceholderText("Recorded at contains (UTC)")
        filters.addWidget(QLabel("Point"))
        filters.addWidget(self._point)
        filters.addWidget(QLabel("Experiment"))
        filters.addWidget(self._experiment)
        filters.addWidget(self._time)
        return filters

    def selected_path(self) -> str | None:
        """Return the chosen native path only after this dialog was accepted."""
        return (
            self._selected_path
            if self.result() == QDialog.DialogCode.Accepted
            else None
        )

    def _refresh(self) -> None:
        point = self._point.currentText() if self._point.currentIndex() > 0 else None
        experiment = (
            self._experiment.currentText()
            if self._experiment.currentIndex() > 0
            else None
        )
        recorded_at = self._time.text()
        self._visible_runs = tuple(
            run
            for run in self._runs
            if (point is None or run.point == point)
            and (experiment is None or run.experiment == experiment)
            and recorded_at in run.recorded_at
        )
        self._selected_path = None
        self._table.setRowCount(0)
        self._table.setRowCount(len(self._visible_runs))
        for row, run in enumerate(self._visible_runs):
            values = (
                run.run_id,
                run.point if run.point is not None else "",
                run.experiment,
                run.recorded_at,
                str(run.data_path) if run.data_path is not None else "",
            )
            for column, value in enumerate(values):
                self._table.setItem(row, column, QTableWidgetItem(value))
        self._selection_changed()

    def _selection_changed(self) -> None:
        row = self._table.currentRow()
        self._load.setEnabled(
            0 <= row < len(self._visible_runs)
            and self._visible_runs[row].data_path is not None
        )

    def _load_selected(self) -> None:
        row = self._table.currentRow()
        if not 0 <= row < len(self._visible_runs):
            return
        path = self._visible_runs[row].data_path
        if path is None:
            return
        self._selected_path = str(path)
        self.accept()

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load native data file",
            self._start_directory,
            "Native HDF5 (*.h5);;All files (*)",
        )
        if path:
            self._selected_path = path
            self.accept()
