"""Qt-owned experiment canvases, named selectors, interactive views and Data previews."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QLabel, QStackedWidget, QVBoxLayout, QWidget

from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.plotting import FigureContainer, attach_existing_figure_to_container

from .data_figure_preview_gallery import DataFigurePreviewGallery
from .figure_pane_layout import FigurePaneLayout
from .interactive_frontend import InteractiveFrontend

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.plotting.figures import FigureCollection

logger = logging.getLogger(__name__)


class ExperimentFigureArea(QWidget):
    """Own the right-side widgets and their presentation state on the Qt owner thread.

    The containing experiment tab retains commands, subscriptions, progress and
    State projections. This widget owns canvas mounting, selector/cache identity,
    interactive child membership and the raster gallery, not results or saving.
    Clearing presentation retires Qt canvases but keeps caller-retained Figures
    saveable. All public calls are synchronous and require the Qt owner thread.
    """

    def __init__(
        self,
        tab_id: str,
        capabilities: AdapterCapabilities,
        read_figure: Callable[[str], Figure | None],
        *,
        preview_renderer: Callable[[Figure], bytes] | None = None,
    ) -> None:
        """Build panes for the tab's fixed capabilities without reading figures.

        tab_id identifies invariant errors. read_figure borrows the containing
        tab's public pane reader (run/analysis/post_analysis) at each Data refresh;
        it transfers no Figure ownership. Arbitrary synchronous reader failures
        leave only that preview empty. preview_renderer follows the gallery's
        Figure-to-PNG contract; None uses its production renderer. Invalid
        renderers fail in gallery construction, and rendering failures stay local
        to the affected card. The splitter takes Qt ownership of this widget.
        """
        super().__init__()
        self.tab_id = tab_id
        self._has_analysis = capabilities.analysis is not AnalysisMode.NONE
        self._has_post = bool(capabilities.post_analysis)
        self._capabilities = capabilities
        self._read_figure = read_figure
        self._preview_renderer = preview_renderer
        self._data_active = False
        self._build_plot_area()

    def _build_analysis_figure_pane(self) -> None:
        # Analysis figure pane (only when analysis present)
        if self._has_analysis:
            self._analysis_stack = FigurePaneLayout()
            self._analysis_placeholder = QLabel("(no plot yet)")
            # tolerated: QtPy promotes the binding enum members onto Qt at runtime.
            self._analysis_placeholder.setAlignment(Qt.AlignCenter)  # pyright: ignore[reportAttributeAccessIssue]
            self._analysis_stack.addWidget(self._analysis_placeholder)
            self._analysis_container = FigureContainer(
                self._analysis_stack, self._analysis_placeholder
            )
            self._analysis_figures: FigureCollection | None = None
            self._analysis_panel_right = QWidget()
            analysis_right = QVBoxLayout(self._analysis_panel_right)
            analysis_right.setContentsMargins(0, 0, 0, 0)
            self._analysis_selector = QComboBox()
            self._analysis_selector.setAccessibleName("Analysis figure")
            self._analysis_selector.hide()
            self._analysis_selector.currentIndexChanged.connect(
                self._on_analysis_figure_selected
            )
            analysis_right.addWidget(self._analysis_selector)
            analysis_right.addWidget(self._analysis_stack, stretch=1)

    def _build_post_figure_pane(self) -> None:
        # Post figure pane (only when post present)
        if self._has_post:
            self._post_stack = FigurePaneLayout()
            self._post_placeholder = QLabel("(no plot yet)")
            # tolerated: QtPy promotes the binding enum members onto Qt at runtime.
            self._post_placeholder.setAlignment(Qt.AlignCenter)  # pyright: ignore[reportAttributeAccessIssue]
            self._post_stack.addWidget(self._post_placeholder)
            self._post_container = FigureContainer(
                self._post_stack, self._post_placeholder
            )
            self._post_figures: FigureCollection | None = None
            self._post_panel_right = QWidget()
            post_right = QVBoxLayout(self._post_panel_right)
            post_right.setContentsMargins(0, 0, 0, 0)
            self._post_selector = QComboBox()
            self._post_selector.setAccessibleName("Post-analysis figure")
            self._post_selector.hide()
            self._post_selector.currentIndexChanged.connect(
                self._on_post_figure_selected
            )
            post_right.addWidget(self._post_selector)
            post_right.addWidget(self._post_stack, stretch=1)

    def _build_plot_area(self) -> None:
        # ── Right pane: per-pane figure containers ────────────────
        self._plot_layout = QVBoxLayout(self)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._right_stack = QStackedWidget()

        # Run figure pane (always)
        self._run_stack = FigurePaneLayout()
        self._run_placeholder = QLabel("(no plot yet)")
        # tolerated: QtPy promotes the binding enum members onto Qt at runtime.
        self._run_placeholder.setAlignment(Qt.AlignCenter)  # pyright: ignore[reportAttributeAccessIssue]
        self._run_stack.addWidget(self._run_placeholder)
        self._run_container = FigureContainer(self._run_stack, self._run_placeholder)
        self._run_figures: FigureCollection | None = None
        self._run_panel_right = QWidget()
        run_right = QVBoxLayout(self._run_panel_right)
        run_right.setContentsMargins(0, 0, 0, 0)
        self._run_selector = QComboBox()
        self._run_selector.setAccessibleName("Run figure")
        self._run_selector.hide()
        self._run_selector.currentIndexChanged.connect(self._on_run_figure_selected)
        run_right.addWidget(self._run_selector)
        run_right.addWidget(self._run_stack, stretch=1)

        self._build_analysis_figure_pane()
        self._build_post_figure_pane()

        # Data preview gallery — Variant A stacked rail (S1, S3)
        self._data_gallery = DataFigurePreviewGallery(
            self._capabilities,
            renderer=self._preview_renderer,
        )

        # Placeholder for Guide (no figure) — Data now shows gallery (S4)
        self._right_placeholder = QLabel("(no plot yet)")
        # tolerated: QtPy promotes the binding enum members onto Qt at runtime.
        self._right_placeholder.setAlignment(Qt.AlignCenter)  # pyright: ignore[reportAttributeAccessIssue]

        self._right_stack.addWidget(self._run_panel_right)
        if self._has_analysis:
            self._right_stack.addWidget(self._analysis_panel_right)
        if self._has_post:
            self._right_stack.addWidget(self._post_panel_right)
        self._right_stack.addWidget(self._data_gallery)
        self._right_stack.addWidget(self._right_placeholder)

        self._plot_layout.addWidget(self._right_stack, stretch=1)

    def _require_analysis(self) -> None:
        if not self._has_analysis:
            raise RuntimeError(f"tab {self.tab_id!r} does not support analysis")

    def _require_post(self) -> None:
        if not self._has_post:
            raise RuntimeError(f"tab {self.tab_id!r} does not support post-analysis")

    def select_pane(
        self, pane: Literal["run", "analysis", "post_analysis", "data", "guide"]
    ) -> None:
        """Present a source pane, Guide placeholder or freshly rendered Data gallery.

        Data activation reads current figures synchronously before switching the
        page, then reflows the visible gallery. Reader/render failures are isolated
        per preview; reflow failures do not prevent activation. Unsupported source
        panes raise RuntimeError. This changes presentation only, never app state.
        """
        self._data_active = pane == "data"
        if pane == "run":
            self._right_stack.setCurrentWidget(self._run_panel_right)
        elif pane == "analysis":
            self._require_analysis()
            self._right_stack.setCurrentWidget(self._analysis_panel_right)
        elif pane == "post_analysis":
            self._require_post()
            self._right_stack.setCurrentWidget(self._post_panel_right)
        elif pane == "data":
            self._refresh_data_gallery()
            self._right_stack.setCurrentWidget(self._data_gallery)
            # Reflow after switching: hidden pages may not yet have their final width.
            # tolerated: suppress(Exception) splits mixed BaseExceptionGroup; preserve the original group.
            try:  # noqa: SIM105
                self._data_gallery.refresh_layout()
            # tolerated: Data activation must survive arbitrary public reflow/presentation failures.
            except Exception:  # noqa: BLE001
                pass
        elif pane == "guide":
            self._right_stack.setCurrentWidget(self._right_placeholder)
        else:
            raise ValueError(f"unknown pane {pane!r}")

    def set_writable(self, *, writable: bool) -> None:
        """Apply Local availability to mounted frontends, not deferred-deleted children."""
        if self._has_analysis:
            # Removed widgets remain QObject children until DeferredDelete; only
            # stack membership owns mounted presentations.
            for index in range(self._analysis_stack.count()):
                frontend = self._analysis_stack.widget(index)
                if isinstance(frontend, InteractiveFrontend):
                    frontend.set_read_only(read_only=not writable)

    def get_run_container(self) -> FigureContainer:
        """Return the stable Run container borrowed by the explicit plot host."""
        return self._run_container

    def get_analysis_container(self) -> FigureContainer:
        """Borrow Analysis's stable container; raise RuntimeError if unsupported."""
        self._require_analysis()
        return self._analysis_container

    def get_post_container(self) -> FigureContainer:
        """Borrow Post's stable container; raise RuntimeError if unsupported."""
        self._require_post()
        return self._post_container

    def prepare_run_container(self) -> FigureContainer:
        """Clear Run and every downstream presentation for a new run."""
        self.clear_all_figures()
        return self._run_container

    def prepare_analysis_container(self) -> FigureContainer:
        """Clear Analysis canvases and return its stable borrowed container.

        Refresh Data only when active. Unsupported Analysis raises RuntimeError.
        """
        self._require_analysis()
        self._analysis_container.clear_dynamic_canvases()
        self._refresh_data_gallery()
        return self._analysis_container

    def prepare_post_container(self) -> FigureContainer:
        """Clear Post canvases and return its stable borrowed container.

        Refresh Data only when active. Unsupported Post raises RuntimeError.
        """
        self.clear_post_figure()
        return self._post_container

    def clear_post_figure(self) -> None:
        """Clear only Post canvases and refresh active Data; retain cached Figures.

        Unsupported Post raises RuntimeError. Caller-retained Figures stay saveable.
        """
        self._require_post()
        self._post_container.clear_dynamic_canvases()
        self._refresh_data_gallery()

    def mount_interactive_widget(self, widget: QWidget) -> None:
        """Take Qt ownership of widget and present it in Analysis, hiding its selector.

        Unsupported Analysis raises RuntimeError. Other mounted children remain;
        unmount_interactive_widgets retires children matching the supplied type.
        Mounting changes the right page, not the containing tab's left selection.
        """
        self._require_analysis()
        self._analysis_selector.hide()
        self._analysis_stack.set_tiled(tiled=False)
        self._analysis_stack.addWidget(widget)
        self._analysis_stack.setCurrentWidget(widget)
        self._right_stack.setCurrentWidget(self._analysis_panel_right)

    def interactive_frontend(self) -> InteractiveFrontend | None:
        """Return only the currently mounted interactive analysis presentation."""
        current = self._analysis_stack.currentWidget()
        return current if isinstance(current, InteractiveFrontend) else None

    def unmount_interactive_widgets(self, widget_type: type[QWidget]) -> None:
        """Remove matching Analysis children with deleteLater and show its placeholder.

        widget_type is a QWidget subclass, including its subclasses when matching.
        Unsupported Analysis raises RuntimeError. Unrelated children stay mounted
        but inactive. This does not stop frontend work; the caller owns teardown.
        """
        self._require_analysis()
        for index in reversed(range(self._analysis_stack.count())):
            widget = self._analysis_stack.widget(index)
            if isinstance(widget, widget_type):
                self._analysis_stack.removeWidget(widget)
                widget.deleteLater()
        self._analysis_stack.setCurrentWidget(self._analysis_placeholder)

    def get_current_figure_for_pane(self, pane: str) -> Figure | None:
        """Borrow the presented Figure for run, analysis or post_analysis, or None.

        Empty/placeholder panes return None. Unknown names raise ValueError;
        unsupported panes or a mounted canvas without a Figure raise RuntimeError.
        This neither renders nor transfers ownership of the Figure.
        """
        from matplotlib.figure import Figure

        if pane == "run":
            stack = self._run_stack
            placeholder = self._run_placeholder
        elif pane == "analysis":
            self._require_analysis()
            stack = self._analysis_stack
            placeholder = self._analysis_placeholder
        elif pane == "post_analysis":
            self._require_post()
            stack = self._post_stack
            placeholder = self._post_placeholder
        else:
            raise ValueError(f"unknown pane {pane!r}")
        canvas = stack.currentWidget()
        if canvas is None or canvas is placeholder:
            return None
        figure = getattr(canvas, "figure", None)
        if not isinstance(figure, Figure):
            raise RuntimeError(f"tab {self.tab_id!r} canvas has no matplotlib figure")
        return figure

    def clear_all_figures(self) -> None:
        """Clear all figure panes (used for LoadData and Run start invalidation)."""
        self._run_container.clear_dynamic_canvases()
        self._run_figures = None
        self._run_selector.blockSignals(True)
        self._run_selector.clear()
        self._run_selector.blockSignals(False)
        self._run_selector.hide()
        if self._has_analysis:
            self._analysis_container.clear_dynamic_canvases()
            self._analysis_figures = None
            self._analysis_selector.blockSignals(True)
            self._analysis_selector.clear()
            self._analysis_selector.blockSignals(False)
            self._analysis_selector.hide()
        if self._has_post:
            self._post_container.clear_dynamic_canvases()
            self._post_figures = None
            self._post_selector.blockSignals(True)
            self._post_selector.clear()
            self._post_selector.blockSignals(False)
            self._post_selector.hide()
        self._refresh_data_gallery()

    def show_run_figure(self, fig: Figure) -> None:
        """Embed a matplotlib Figure in the Run pane."""
        canvas = attach_existing_figure_to_container(fig, self._run_container)
        draw = getattr(canvas, "draw", None)
        if not callable(draw):
            raise RuntimeError("Attached run canvas does not support draw()")
        draw()
        logger.debug("show_run_figure: tab_id=%r canvas set", self.tab_id)
        self._refresh_data_gallery()

    def show_analysis_figures(self, figures: FigureCollection | None) -> None:
        """Draw borrowed named Figures in Analysis, or clear for None/empty input.

        Preserve the selected name for the same collection identity. Replacing the
        collection rebuilds the selector with signals blocked. Unsupported Analysis,
        a missing selected name or a nondrawable canvas raises RuntimeError; other
        attach/draw failures propagate. Refresh only an active Data gallery.
        """
        self._require_analysis()
        self._show_named_figures("analysis", figures)

    def show_post_analysis_figures(self, figures: FigureCollection | None) -> None:
        """Present Post Figures using the same identity/draw contract as Analysis.

        None/empty clears presentation. Unsupported Post raises RuntimeError.
        """
        self._require_post()
        self._show_named_figures("post_analysis", figures)

    def present_figures(self, pane: str, figures: FigureCollection) -> None:
        """Borrow original names after attachment, synchronously on the Qt owner.

        Only already attached Figures become visible. This never presents an
        undrawn Figure, copies a collection, or changes its ownership. Unknown
        panes raise ValueError. Clear/replacement ends these presentation borrows.
        """
        self._show_named_figures(pane, figures, attach=False)

    def _show_named_figures(
        self,
        pane: str,
        figures: FigureCollection | None,
        *,
        attach: bool = True,
    ) -> None:
        if pane == "run":
            selector, container, stack = (
                self._run_selector,
                self._run_container,
                self._run_stack,
            )
            previous = self._run_figures
            self._run_figures = figures
        elif pane == "analysis":
            self._require_analysis()
            selector, container, stack = (
                self._analysis_selector,
                self._analysis_container,
                self._analysis_stack,
            )
            previous = self._analysis_figures
            self._analysis_figures = figures
        elif pane == "post_analysis":
            self._require_post()
            selector, container, stack = (
                self._post_selector,
                self._post_container,
                self._post_stack,
            )
            previous = self._post_figures
            self._post_figures = figures
        else:
            raise ValueError(f"unknown pane {pane!r}")
        selected = selector.currentText() if figures is previous else ""
        if attach and figures is not previous:
            container.clear_dynamic_canvases()
        names = (
            []
            if figures is None
            else [
                name
                for name, figure in figures.items()
                if attach or container.contains_figure(figure)
            ]
        )
        selector.blockSignals(True)
        try:
            selector.clear()
            selector.addItems(names)
            if selected in names:
                selector.setCurrentText(selected)
        finally:
            selector.blockSignals(False)
        selector.setVisible(len(names) > 4)
        stack.set_tiled(tiled=0 < len(names) <= 4)
        if not names:
            if attach:
                container.clear_dynamic_canvases()
        elif figures is not None:
            if attach:
                self._draw_named_figures(figures, container, names)
            figure = figures[selector.currentText()]
            attach_existing_figure_to_container(figure, container)
        self._refresh_data_gallery()

    @staticmethod
    def _draw_named_figures(
        figures: FigureCollection, container: FigureContainer, names: list[str]
    ) -> None:
        for name in names:
            canvas = attach_existing_figure_to_container(figures[name], container)
            draw = getattr(canvas, "draw", None)
            if not callable(draw):
                raise RuntimeError("Attached figure canvas does not support draw()")
            draw()

    def _on_run_figure_selected(self, index: int) -> None:
        if index >= 0 and self._run_figures is not None:
            self._show_named_figures("run", self._run_figures, attach=False)

    def _on_analysis_figure_selected(self, index: int) -> None:
        if index >= 0 and self._analysis_figures is not None:
            self._show_named_figures("analysis", self._analysis_figures)

    def _on_post_figure_selected(self, index: int) -> None:
        if index >= 0 and self._post_figures is not None:
            self._show_named_figures("post_analysis", self._post_figures)

    def _refresh_data_gallery(self) -> None:
        """Refresh Data gallery from current pane figures when Data is visible (S2)."""
        if not self._data_active:
            return
        try:
            run_fig = self._read_figure("run")
        # tolerated: The pane reader is overridable; a failed run lookup leaves only its preview empty.
        except Exception:  # noqa: BLE001
            run_fig = None
        if self._has_analysis:
            try:
                ana_fig = self._read_figure("analysis")
            # tolerated: An arbitrary analysis pane-reader failure must not block the other previews.
            except Exception:  # noqa: BLE001
                ana_fig = None
        else:
            ana_fig = None
        if self._has_post:
            try:
                post_fig = self._read_figure("post_analysis")
            # tolerated: An arbitrary post pane-reader failure must not block the other previews.
            except Exception:  # noqa: BLE001
                post_fig = None
        else:
            post_fig = None
        try:
            self._data_gallery.update_figures(run_fig, ana_fig, post_fig)
            self._data_gallery.update_collections(
                self._run_figures if run_fig is not None else None,
                self._analysis_figures
                if self._has_analysis and ana_fig is not None
                else None,
                self._post_figures if self._has_post and post_fig is not None else None,
            )
        except Exception:
            logger.exception("failed to refresh data gallery")
