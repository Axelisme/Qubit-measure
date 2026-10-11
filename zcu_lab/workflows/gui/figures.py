"""Fixed app-local presentation lifetime, separate from native run ownership."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from qtpy.QtWidgets import QFileDialog, QPushButton, QWidget
from zcu_tools.experiment.workflows import CommittedRecord
from zcu_tools.experiment.workflows.artifacts import read_committed_records
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.plotting.figure_export import render_figure_png
from zcu_tools.plotting.figures import NamedFigures
from zcu_tools.plotting.plots import PlotHost, Plots

logger = logging.getLogger(__name__)


class WorkflowFigureViews:
    """Own native and independent offline presentations on one explicit host.

    App-local implementation, not a package-exported run owner. open_button is
    the sole directory-opening QPushButton; Window owns its enabled policy/slot.
    native is the attached run's actual Plots or None before install/after release.
    _offline is one finished read-only view, never an RPC/native target. _cursor
    is the last attempted native commit count, reset with each new native Plots.
    No Engine, Controller, run identity, operation, session or admission is held.
    Window admits owner/idle/target access before calling; Pause does nothing,
    terminal refresh precedes finish_native, and successful Start/close release
    the offline view. Callers retain returned Figures independently of release.
    This owner is fixed with Window. Science drawing is resolved at each call,
    so bounded science reload does not freeze an old projection callable.
    """

    def __init__(
        self, host: PlotHost, parent: QWidget, open_clicked: Callable[[], None]
    ) -> None:
        """Build on the Qt owner with QApplication, an explicit host and QWidget parent.

        open_clicked is the Window's sole opening slot, not an I/O callback here.
        Caller keeps this owner through producer drain and releases on close.
        """
        self._host = host
        self.open_button = QPushButton("Open overnight trend…", parent)
        self.open_button.setObjectName("workflow.trend.open")
        self.open_button.clicked.connect(open_clicked)
        self.native: Plots | None = None
        self._offline: Plots | None = None
        self._cursor: int | None = None

    def install_native(self, plots: Plots) -> None:
        """Install new active Plots before native admission listeners reenter.

        Window has admitted replacement. Release the previous native presentation
        and reset its cursor; offline stays until Start actually succeeds.
        """
        if self.native is not None:
            self.native.release()
        self.native = plots
        self._cursor = None

    def start_succeeded(self) -> None:
        """Release the finished offline view only after Start submission succeeds."""
        self._release_offline()

    def start_failed(self) -> None:
        """Finish/release the unsuccessful native candidate after submit fails."""
        if self.native is not None:
            self.native.finish(present=False)
            self.native.release()
            self.native = None
        self._cursor = None

    def choose_metadata_root(self) -> Path | None:
        """Choose a metadata directory on Qt owner; cancel returns None without I/O."""
        directory = QFileDialog.getExistingDirectory(
            self.open_button, "Select workflow metadata directory"
        )
        return Path(directory) if directory else None

    def refresh_native(self, records: tuple[CommittedRecord, ...]) -> None:
        """Draw an admitted overnight run's complete committed tuple once per cursor.

        Native Plots must be active; Window supplies authoritative Engine records
        from Engine.records with its default zero cursor. No refit occurs. Cursor
        advances before draw, preventing automatic timer retries after errors or
        attempts to draw finished Plots. Draw errors propagate for view isolation.
        """
        from ..trend import draw_overnight_trend

        if self.native is not None and self._cursor != len(records):
            self._cursor = len(records)
            draw_overnight_trend(self.native, records)

    def finish_native(self) -> None:
        """Finish after producer return and final committed refresh, never on Pause."""
        if self.native is not None:
            self.native.finish()

    def native_names(self) -> tuple[str, ...]:
        """Read retained names after Window target/acceptance admission.

        Released native presentation raises FailedPreconditionError(no_plots);
        an accepted empty collection returns (), without rendering or mutation.
        """
        return tuple(self._require_native())

    def native_png(self, name: str) -> bytes:
        """Render admitted native name at 640 x 480 on Qt owner, without retry.

        Unknown names raise KeyError. Released native Plots raise no_plots,
        matching native_names. Offline views never supply native RPC products.
        """
        return render_figure_png(self._require_native()[name])

    def _require_native(self) -> Plots:
        if self.native is None:
            raise FailedPreconditionError(
                "Run plots have been released", reason_code="no_plots"
            )
        return self.native

    def open(self, metadata_root: Path) -> NamedFigures:
        """Publish a finished independent trend only after candidate success.

        Window must admit Qt owner/idle access before I/O. metadata_root contains
        journal.jsonl. Missing/invalid source and draw errors propagate; failed
        candidates finish without presentation and release, retaining the original
        error if cleanup fails. Draw has no rollback guarantee. Native presentation
        and target stay unchanged. Returned named Figures survive view release.
        """
        from ..trend import draw_overnight_trend

        records = read_committed_records(metadata_root)
        candidate = Plots(self._host)
        try:
            draw_overnight_trend(candidate, records)
            figures = candidate.finish()
        except Exception:
            try:
                candidate.finish(present=False)
                candidate.release()
            except Exception:
                logger.exception("Failed to release an unsuccessful trend view")
            raise
        previous = self._offline
        self._offline = candidate
        if previous is not None:
            previous.release()
        return figures

    def _release_offline(self) -> None:
        if self._offline is not None:
            self._offline.release()
            self._offline = None

    def release(self) -> None:
        """Release presentations only after Window close admission/producer drain."""
        self._release_offline()
        if self.native is not None:
            self.native.release()
            self.native = None
        self._cursor = None
