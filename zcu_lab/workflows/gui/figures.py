"""Fixed app-local presentation lifetime, separate from native run ownership."""

from __future__ import annotations

from qtpy.QtWidgets import QWidget
from zcu_tools.experiment.workflows import CommittedRecord
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.plotting.figure_export import render_figure_png
from zcu_tools.plotting.plots import Plots


class WorkflowFigureViews:
    """Own the attached native presentation and its committed drawing cursor.

    App-local implementation, not a package-exported run owner.
    native is the attached run's actual Plots or None before install/after release.
    _cursor is the last attempted native commit count, reset with new Plots.
    No Engine, Controller, run identity, operation, session or admission is held.
    Window admits owner/target access before calling; Pause does nothing,
    terminal refresh precedes finish_native, and close releases presentation.
    Callers retain Figures independently of release. This owner is fixed with
    Window. Science drawing is resolved at each call, so bounded science reload
    does not freeze an old projection callable.
    """

    def __init__(self) -> None:
        """Create empty presentation state without acquiring any resources.

        Caller keeps this owner through producer drain and releases on close.
        No controls, callbacks or native operation authority are acquired.
        """
        self.native: Plots | None = None
        self._cursor: int | None = None

    def install_native(self, plots: Plots) -> None:
        """Install new active Plots before native admission listeners reenter.

        Window has admitted replacement. Release the previous native presentation
        and reset its cursor.
        """
        if self.native is not None:
            self.native.release()
        self.native = plots
        self._cursor = None

    def start_failed(self) -> None:
        """Finish/release the unsuccessful native candidate after submit fails."""
        if self.native is not None:
            self.native.finish(present=False)
            self.native.release()
            self.native = None
        self._cursor = None

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
            self._size_trend_canvas(self.native)

    def _size_trend_canvas(self, plots: Plots) -> None:
        canvas = plots["overnight_trend"].canvas
        if not isinstance(canvas, QWidget):
            raise RuntimeError("Presented overnight trend has no Qt canvas")
        # Keep both axes readable when the forms consume the default layout.
        canvas.setMinimumSize(640, 480)

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
        matching native_names.
        """
        return render_figure_png(self._require_native()[name])

    def _require_native(self) -> Plots:
        if self.native is None:
            raise FailedPreconditionError(
                "Run plots have been released", reason_code="no_plots"
            )
        return self.native

    def release(self) -> None:
        """Release native presentation after Window close admission/producer drain."""
        if self.native is not None:
            self.native.release()
            self.native = None
        self._cursor = None
