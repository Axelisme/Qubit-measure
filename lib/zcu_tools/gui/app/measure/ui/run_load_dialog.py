"""Native file selection from opened-entry file metadata, without loading data."""

from __future__ import annotations

from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.measure.artifacts import SavedRunInfo


class RunLoadDialog(QDialog):
    """Choose a native file on the Qt owner thread, even if metadata is unreadable.

    runs contains ordered SavedRunInfo from the opened entry; no files are read.
    initial_point selects a known point, or all points when empty/unknown.
    start_directory is QFileDialog's initial folder, not a reserved destination.
    Accepted implies selected_path() is nonempty. Reject/cancel returns no path.
    Point/experiment filters are exact; time filter matches started_at text.
    Metadata errors remain visible in the native-data column and its tooltip.
    Every row is selectable regardless of metadata validity. Paths may have
    disappeared; the existing Load owner reports that failure after selection.
    This is a single Load interaction; finished discards further selection input.
    The owning window must reject it before window teardown. The caller reads
    selected_path() after exec() and then schedules this dialog's deleteLater().
    A file chooser returning after retirement cannot accept a new selection.
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
        self._interaction_finished = False
        self.finished.connect(self._finish_interaction)
        self._visible_runs: tuple[SavedRunInfo, ...] = ()
        self._start_directory = start_directory
        self._selected_path: str | None = None
        self.setWindowTitle("Load Data")
        self.resize(1100, 440)
        layout = QVBoxLayout(self)
        filters = QWidget()
        filters.setLayout(self._build_filters(initial_point))
        layout.addWidget(filters)
        self._table = QTableWidget(0, 7)
        self._table.setObjectName("load_runs")
        self._table.setHorizontalHeaderLabels(
            [
                "Run",
                "Point",
                "Experiment",
                "Started at",
                "Finished at",
                "Completion",
                "Native data",
            ]
        )
        header = self._table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
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
        commands = QWidget()
        command_layout = QHBoxLayout(commands)
        command_layout.setContentsMargins(0, 0, 0, 0)
        command_layout.addWidget(file_button)
        command_layout.addStretch()
        command_layout.addWidget(self._load)
        buttons.addWidget(commands)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        self._point.currentIndexChanged.connect(self._refresh)
        self._experiment.currentIndexChanged.connect(self._refresh)
        self._time.textChanged.connect(self._refresh)
        self._table.itemSelectionChanged.connect(self._selection_changed)
        self._refresh()

    def _finish_interaction(self) -> None:
        self._interaction_finished = True

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
        self._time.setObjectName("load_started_at")
        self._time.setPlaceholderText("Started at contains (UTC)")
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
        started_at = self._time.text()
        self._visible_runs = tuple(
            run
            for run in self._runs
            if (point is None or run.point == point)
            and (experiment is None or run.experiment == experiment)
            and started_at in (run.started_at or "")
        )
        self._selected_path = None
        self._table.setRowCount(0)
        self._table.setRowCount(len(self._visible_runs))
        for row, run in enumerate(self._visible_runs):
            values = (
                run.run_id,
                run.point if run.point is not None else "",
                run.experiment,
                run.started_at or "",
                run.finished_at or "",
                run.completion or "",
                f"Metadata error: {run.metadata_error}"
                if run.metadata_error
                else str(run.data_path),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(
                    f"{run.data_path}\n{run.metadata_error}"
                    if run.metadata_error
                    else str(run.data_path)
                )
                self._table.setItem(row, column, item)
        self._selection_changed()

    def _selection_changed(self) -> None:
        row = self._table.currentRow()
        self._load.setEnabled(0 <= row < len(self._visible_runs))

    def _load_selected(self) -> None:
        if self._interaction_finished:
            return
        row = self._table.currentRow()
        if not 0 <= row < len(self._visible_runs):
            return
        path = self._visible_runs[row].data_path
        self._selected_path = str(path)
        self.accept()

    def _open_file(self) -> None:
        if self._interaction_finished:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load native data file",
            self._start_directory,
            "Native HDF5 (*.h5);;All files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path and not self._interaction_finished:
            self._selected_path = path
            self.accept()
