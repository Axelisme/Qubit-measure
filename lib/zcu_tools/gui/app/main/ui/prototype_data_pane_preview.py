# THROWAWAY PROTOTYPE — measure-save-preview-gallery / 001-prototype-data-pane-preview-layouts
# ---------------------------------------------------------------------------
# NOT FOR PRODUCTION. Synthetic in-memory state only — no save/load, hardware,
# network, project-data, or persistence effects. Located next to the Data-pane
# UI it is prototyping (ExpTabWidget / ArtifactSaveCenter are read-only
# density references).
#
# Question: "What should Data-pane preview layout look like?"
# Three variants of the Data subtab, switchable via in-window pill and ←/→
# keys, in realistic experiment-tab split-pane geometry (1280×~780, left
# QTabWidget + right figure pane). See launcher help for the one-command
# launch.
#
#   Variant A — Stacked Right Rail  (save left, previews stacked right)
#   Variant B — Inline Cards         (save+preview paired per artifact)
#   Variant C — Focus Tabs           (save left, single large tabbed preview right)
#
# All actions are inert and show "Prototype — no file written" feedback.
# Arrow keys do NOT intercept when a line-edit / text-edit has focus.
# Availability toggles (Run / Analysis / Post) surface named empty states and
# a live summary so the reviewer can see current prototype state.
#
# Launch:
#   .venv/bin/python script/run_data_pane_preview_prototype.py
#   QT_QPA_PLATFORM=offscreen .venv/bin/python script/run_data_pane_preview_prototype.py --smoke
#
# Cleanup: prototype remains on the task integration branch as throwaway
# source until user feedback (A6) selects a direction; validated decision is
# folded into production in a following ticket.
# ---------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Callable

from qtpy.QtCore import Qt, QTimer, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Synthetic figure factories (in-memory only, no file I/O)
# ---------------------------------------------------------------------------

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False
    Figure = object  # type: ignore[misc, assignment]
    FigureCanvasQTAgg = object  # type: ignore[misc, assignment]


def _make_run_fig() -> Figure:  # type: ignore[override]
    if not _HAS_MPL:
        raise RuntimeError("matplotlib unavailable")
    fig = Figure(figsize=(5.0, 2.8), constrained_layout=True)
    fig.patch.set_facecolor("#fafafa")
    ax = fig.add_subplot(111)
    xs = [4.85 + i * 0.003 for i in range(320)]
    ys = []
    for x in xs:
        # Lorentzian dip around 5.05 GHz with a little texture
        y = 1.0 - 0.62 / (1 + ((x - 5.05) / 0.04) ** 2)
        y += 0.015 * math.sin(40 * x)
        y += 0.006 * math.sin(120 * x)
        ys.append(y)
    ax.plot(xs, ys, color="#286ac7", linewidth=1.7, label="|S21|")
    ax.set_title("Run — OneTone |S21| vs Flux", fontsize=9, pad=8)
    ax.set_xlabel("Frequency (GHz)", fontsize=8)
    ax.set_ylabel("Magnitude (arb.)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.12)
    ax.set_xlim(xs[0], xs[-1])
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    return fig


def _make_analysis_fig() -> Figure:  # type: ignore[override]
    if not _HAS_MPL:
        raise RuntimeError("matplotlib unavailable")
    fig = Figure(figsize=(5.0, 2.8), constrained_layout=True)
    fig.patch.set_facecolor("#fafafa")
    ax = fig.add_subplot(111)
    xs = [i * 0.12 for i in range(50)]
    ys = [
        0.5 + 0.35 * math.sin(x) + 0.05 * math.cos(3 * x) + 0.03 * ((x % 1.7) - 0.85)
        for x in xs
    ]
    fit = [0.5 + 0.35 * math.sin(x) for x in xs]
    ax.scatter(xs, ys, s=14, alpha=0.55, color="#444444", label="data", zorder=2)
    ax.plot(xs, fit, color="#c0392b", linewidth=1.8, label="fit", zorder=3)
    ax.set_title("Analysis — Fitted resonance", fontsize=9, pad=8)
    ax.set_xlabel("Detuning (MHz)", fontsize=8)
    ax.set_ylabel("Response", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.12)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    return fig


def _make_post_fig() -> Figure:  # type: ignore[override]
    if not _HAS_MPL:
        raise RuntimeError("matplotlib unavailable")
    fig = Figure(figsize=(5.0, 2.8), constrained_layout=True)
    fig.patch.set_facecolor("#fafafa")
    ax = fig.add_subplot(111)
    # 2-D heatmap stub (flux × frequency) with synthetic Lorentzian ridge
    import numpy as np

    flux = np.linspace(-0.5, 0.5, 80)
    freq = np.linspace(4.7, 5.3, 80)
    F, Fr = np.meshgrid(flux, freq)
    # ridge center moves with flux
    center = 5.0 + 0.25 * np.sin(3 * flux)
    # broadcast
    data = np.zeros((len(freq), len(flux)))
    for j, c in enumerate(center):
        data[:, j] = (
            1.0
            - 0.7 / (1 + ((freq - c) / 0.06) ** 2)
            + 0.04 * np.random.randn(len(freq))
        )
    im = ax.imshow(
        data,
        origin="lower",
        extent=[flux[0], flux[-1], freq[0], freq[-1]],
        aspect="auto",
        cmap="magma",
        vmin=0.2,
        vmax=1.0,
    )
    ax.set_title("Post-Analysis — Flux-dependent map", fontsize=9, pad=8)
    ax.set_xlabel("Flux (Φ₀)", fontsize=8)
    ax.set_ylabel("Frequency (GHz)", fontsize=8)
    ax.tick_params(labelsize=7)
    # small colorbar-like annotation
    fig.colorbar(im, ax=ax, shrink=0.85, label="arb.")
    return fig


# ---------------------------------------------------------------------------
# Small helpers: pills, empty states, preview cards
# ---------------------------------------------------------------------------


def _pill_style(kind: str) -> str:
    if kind == "available":
        return "background:#d4edda; color:#155724; border:1px solid #c3e6cb; padding:2px 7px; border-radius:9px; font-size:10px; font-weight:600;"
    if kind == "no_result":
        return "background:#f3e0ff; color:#6a1b9a; border:1px solid #d9b6ff; padding:2px 7px; border-radius:9px; font-size:10px; font-weight:600;"
    # no_figure
    return "background:#fff3cd; color:#856404; border:1px solid #ffe69c; padding:2px 7px; border-radius:9px; font-size:10px; font-weight:600;"


class _PreviewCard(QWidget):
    """One named source preview: header pill + figure canvas or named empty."""

    def __init__(
        self,
        source_key: str,
        display_name: str,
        make_fig: Callable[[], Figure],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source_key = source_key
        self._display_name = display_name
        self._make_fig = make_fig
        self._available = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.setObjectName(f"previewCard_{source_key}")

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel(f"<b>{display_name}</b>")
        title.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        header.addWidget(title)
        header.addStretch()
        self._pill = QLabel()
        self._pill.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        header.addWidget(self._pill)
        layout.addLayout(header)

        self._stack = QStackedWidget()
        # figure page (stack holds the page, not the bare canvas)
        fig_page = QWidget()
        fig_layout = QVBoxLayout(fig_page)
        fig_layout.setContentsMargins(0, 0, 0, 0)
        try:
            fig = self._make_fig()
            canvas = FigureCanvasQTAgg(fig)  # type: ignore[abstract]
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # type: ignore[attr-defined]
            # keep figure alive on canvas
            fig_layout.addWidget(canvas)
            self._canvas_page = fig_page
        except Exception as exc:  # pragma: no cover
            fallback = QLabel(f"(figure unavailable: {exc})")
            fallback.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
            fallback.setStyleSheet("color:#6c757d;")
            fig_layout.addWidget(fallback)
            self._canvas_page = fig_page
        self._stack.addWidget(fig_page)

        # empty page
        empty_page = QWidget()
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.setContentsMargins(8, 12, 8, 12)
        empty_layout.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        empty_label = QLabel(
            f"{display_name} — NO FIGURE\n(no result yet)"
            if source_key != "run"
            else f"{display_name} — NO RESULT\n(measurement not run yet)"
        )
        empty_label.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        empty_label.setStyleSheet(
            "color:#6c757d; border:1px dashed #adb5bd; background:#f8f9fa; "
            "padding:18px 14px; border-radius:6px; font-size:11px;"
        )
        empty_label.setWordWrap(True)
        empty_layout.addWidget(empty_label)
        self._stack.addWidget(empty_page)
        self._empty_page = empty_page

        layout.addWidget(self._stack, stretch=1)

        # path hint
        self._path_hint = QLabel(
            f"{display_name} image will save to /tmp/… (prototype)"
        )
        self._path_hint.setStyleSheet("color:#6c757d; font-size:10px;")
        self._path_hint.setWordWrap(True)
        layout.addWidget(self._path_hint)

        self.setStyleSheet(
            "QWidget#previewCard_run, QWidget#previewCard_analysis, QWidget#previewCard_post { border:1px solid #dee2e6; border-radius:6px; background:white; }"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # type: ignore[attr-defined]
        self._apply()

    def set_available(self, available: bool) -> None:
        self._available = bool(available)
        self._apply()

    def _apply(self) -> None:
        if self._available:
            self._pill.setText("● AVAILABLE")
            self._pill.setStyleSheet(_pill_style("available"))
            self._stack.setCurrentWidget(self._canvas_page)
        else:
            if self._source_key == "run":
                self._pill.setText("— NO RESULT")
                self._pill.setStyleSheet(_pill_style("no_result"))
            else:
                self._pill.setText("○ NO FIGURE")
                self._pill.setStyleSheet(_pill_style("no_figure"))
            self._stack.setCurrentWidget(self._empty_page)


class _CompactSaveRow(QWidget):
    """One artifact row: title + status + path + Browse… + Save (all inert)."""

    def __init__(
        self,
        kind: str,
        title: str,
        placeholder: str,
        with_comment: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        self.setStyleSheet(
            "background:white; border:1px solid #e9ecef; border-radius:6px;"
        )

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        t = QLabel(f"<b>{title}</b>")
        t.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        header.addWidget(t)
        header.addStretch()
        self.status = QLabel("○ NOT SAVED")
        self.status.setStyleSheet("color:#c45100; font-weight:700; font-size:11px;")
        header.addWidget(self.status)
        outer.addLayout(header)

        path_row = QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_row.setSpacing(6)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(placeholder)
        self.path_edit.setText(placeholder.replace("/tmp/", "/tmp/demo_"))
        self.path_edit.setToolTip(
            "Prototype path — editing is local only, no persistence"
        )
        path_row.addWidget(self.path_edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.setFixedWidth(78)
        browse.setToolTip("Prototype — no file dialog")
        browse.clicked.connect(lambda: self._inert("Browse — prototype, no dialog"))
        path_row.addWidget(browse)
        save = QPushButton("Save")
        save.setFixedWidth(68)
        save.setToolTip("Prototype — no file will be written")
        save.clicked.connect(lambda: self._inert("Save — prototype, no file written"))
        path_row.addWidget(save)
        outer.addLayout(path_row)

        if with_comment:
            self.comment = QTextEdit()
            self.comment.setPlaceholderText("Optional comment…")
            self.comment.setFixedHeight(56)
            self.comment.setToolTip("Prototype comment — no persistence")
            outer.addWidget(self.comment)
        else:
            self.comment = None  # type: ignore[assignment]

    def _inert(self, msg: str) -> None:
        win = self.window()
        if win is not None and hasattr(win, "statusBar"):
            try:
                sb = win.statusBar()  # type: ignore[call-arg, attr-defined]
                sb.showMessage(msg + " (inert)", 3000)
            except Exception:
                pass

    def set_available(self, available: bool) -> None:
        if available:
            self.status.setText("○ NOT SAVED")
            self.status.setStyleSheet("color:#c45100; font-weight:700; font-size:11px;")
            self.path_edit.setEnabled(True)
        else:
            self.status.setText("— NO RESULT")
            self.status.setStyleSheet("color:#6a1b9a; font-weight:700; font-size:11px;")
            self.path_edit.setEnabled(False)


class _CompactSaveControls(QWidget):
    """ArtifactSaveCenter-like compact controls for Variant A/C left pane."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)
        outer.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]

        heading = QLabel(
            "<b>Save results</b> <span style='color:#6c757d; font-weight:normal;'>(prototype — inert)</span>"
        )
        heading.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        outer.addWidget(heading)
        hint = QLabel(
            "Only outputs supported by this experiment appear here. All Save actions are inert in the prototype."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6c757d; font-size:11px;")
        outer.addWidget(hint)

        self.data_row = _CompactSaveRow(
            "data", "Measurement data", "/tmp/data.hdf5", with_comment=True
        )
        outer.addWidget(self.data_row)
        self.analysis_row = _CompactSaveRow(
            "analysis", "Analysis image", "/tmp/image.png"
        )
        outer.addWidget(self.analysis_row)
        self.post_row = _CompactSaveRow(
            "post", "Post-analysis image", "/tmp/post_image.png"
        )
        outer.addWidget(self.post_row)

        bottom = QWidget()
        bottom_h = QHBoxLayout(bottom)
        bottom_h.setContentsMargins(0, 0, 0, 0)
        bottom_h.setSpacing(8)
        load = QPushButton("Load Data")
        load.setFixedHeight(34)
        load.setToolTip("Prototype — no file will be loaded")
        load.clicked.connect(
            lambda: self._inert("Load Data — prototype, no file loaded")
        )
        bottom_h.addWidget(load, stretch=1)
        save_all = QPushButton("Save All")
        save_all.setFixedHeight(34)
        save_all.setDefault(True)
        save_all.setToolTip("Prototype — no files will be written")
        save_all.clicked.connect(
            lambda: self._inert("Save All — prototype, no files written")
        )
        bottom_h.addWidget(save_all, stretch=1)
        outer.addStretch()
        outer.addWidget(bottom)

    def _inert(self, msg: str) -> None:
        win = self.window()
        if win is not None and hasattr(win, "statusBar"):
            try:
                win.statusBar().showMessage(msg + " (inert)", 3000)  # type: ignore[attr-defined]
            except Exception:
                pass

    def set_availability(self, run: bool, analysis: bool, post: bool) -> None:
        self.data_row.set_available(run)
        self.analysis_row.set_available(analysis)
        self.post_row.set_available(post)


class _InlineSavePreviewCard(QFrame):
    """Grouped save row + preview for one artifact (Variant B inline)."""

    def __init__(
        self,
        source_key: str,
        title: str,
        placeholder: str,
        make_fig: Callable[[], Figure],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)  # type: ignore[attr-defined]
        self.setStyleSheet(
            "QFrame { background:white; border:1px solid #dee2e6; border-radius:8px; }"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # save row header
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        t = QLabel(f"<b>{title}</b>")
        t.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        header.addWidget(t)
        header.addStretch()
        self.status = QLabel("○ NOT SAVED")
        self.status.setStyleSheet("color:#c45100; font-weight:700; font-size:11px;")
        header.addWidget(self.status)
        outer.addLayout(header)

        path_row = QHBoxLayout()
        path_row.setSpacing(6)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(placeholder)
        self.path_edit.setText(placeholder.replace("/tmp/", "/tmp/demo_"))
        path_row.addWidget(self.path_edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.setFixedWidth(72)
        browse.setToolTip("Prototype — no dialog")
        browse.clicked.connect(lambda: self._inert("Browse — prototype, no dialog"))
        path_row.addWidget(browse)
        save = QPushButton("Save")
        save.setFixedWidth(64)
        save.setToolTip("Prototype — no file written")
        save.clicked.connect(lambda: self._inert("Save — prototype, no file written"))
        path_row.addWidget(save)
        outer.addLayout(path_row)

        if source_key == "run":
            self.comment = QTextEdit()
            self.comment.setPlaceholderText("Optional comment…")
            self.comment.setFixedHeight(48)
            outer.addWidget(self.comment)
        else:
            self.comment = None  # type: ignore[assignment]

        # preview
        self.preview = _PreviewCard(source_key, title, make_fig)
        # hide duplicate path hint inside preview for inline (path already above)
        try:
            self.preview._path_hint.hide()  # type: ignore[attr-defined]
        except Exception:
            pass
        self.preview.setStyleSheet("border:none;")
        outer.addWidget(self.preview, stretch=1)

    def _inert(self, msg: str) -> None:
        win = self.window()
        if win is not None and hasattr(win, "statusBar"):
            try:
                win.statusBar().showMessage(msg + " (inert)", 3000)  # type: ignore[attr-defined]
            except Exception:
                pass

    def set_available(self, available: bool) -> None:
        self.preview.set_available(available)
        if available:
            self.status.setText("○ NOT SAVED")
            self.status.setStyleSheet("color:#c45100; font-weight:700; font-size:11px;")
            self.path_edit.setEnabled(True)
        else:
            self.status.setText(
                "— NO RESULT" if self.preview._source_key == "run" else "○ NO FIGURE"
            )  # type: ignore[attr-defined]
            self.status.setStyleSheet(
                "color:#6a1b9a; font-weight:700; font-size:11px;"
                if self.preview._source_key == "run"
                else "color:#856404; font-weight:700; font-size:11px;"
            )
            self.path_edit.setEnabled(False)


class _InlineCardsWidget(QWidget):
    """Variant B left pane: three save+preview cards stacked + bottom Save All."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)
        outer.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]

        heading = QLabel(
            "<b>Save results — Inline previews</b> <span style='color:#6c757d; font-weight:normal;'>(prototype)</span>"
        )
        heading.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        outer.addWidget(heading)
        hint = QLabel(
            "Each card pairs its save destination, status, and Save button directly with its figure preview."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6c757d; font-size:11px;")
        outer.addWidget(hint)

        self.run_card = _InlineSavePreviewCard(
            "run", "Measurement data", "/tmp/data.hdf5", _make_run_fig
        )
        outer.addWidget(self.run_card)
        self.analysis_card = _InlineSavePreviewCard(
            "analysis", "Analysis image", "/tmp/image.png", _make_analysis_fig
        )
        outer.addWidget(self.analysis_card)
        self.post_card = _InlineSavePreviewCard(
            "post", "Post-analysis image", "/tmp/post_image.png", _make_post_fig
        )
        outer.addWidget(self.post_card)

        bottom = QWidget()
        bottom_h = QHBoxLayout(bottom)
        bottom_h.setContentsMargins(0, 0, 0, 0)
        bottom_h.setSpacing(8)
        load = QPushButton("Load Data")
        load.setFixedHeight(34)
        load.setToolTip("Prototype — no file loaded")
        load.clicked.connect(
            lambda: self._inert("Load Data — prototype, no file loaded")
        )
        bottom_h.addWidget(load, stretch=1)
        save_all = QPushButton("Save All")
        save_all.setFixedHeight(34)
        save_all.setDefault(True)
        save_all.setToolTip("Prototype — no files written")
        save_all.clicked.connect(
            lambda: self._inert("Save All — prototype, no files written")
        )
        bottom_h.addWidget(save_all, stretch=1)
        outer.addWidget(bottom)
        outer.addStretch()

    def _inert(self, msg: str) -> None:
        win = self.window()
        if win is not None and hasattr(win, "statusBar"):
            try:
                win.statusBar().showMessage(msg + " (inert)", 3000)  # type: ignore[attr-defined]
            except Exception:
                pass

    def set_availability(self, run: bool, analysis: bool, post: bool) -> None:
        self.run_card.set_available(run)
        self.analysis_card.set_available(analysis)
        self.post_card.set_available(post)


# ---------------------------------------------------------------------------
# Switcher pill
# ---------------------------------------------------------------------------


class _SwitcherBar(QWidget):
    left_clicked = Signal()
    right_clicked = Signal()
    variant_selected = Signal(int)

    def __init__(
        self, variants: list[str], current: int = 0, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._variants = list(variants)
        self._current = int(current)
        self.setFixedHeight(40)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)
        self.setStyleSheet(
            "QWidget { background:#2b2e3a; border-radius:20px; } "
            "QPushButton { background:#3a3e4f; color:white; border:1px solid #4a4e62; border-radius:12px; padding:4px 10px; }"
            "QPushButton:hover { background:#4a4e62; }"
            "QLabel { color:white; font-weight:600; font-size:12px; background:transparent; border:none; }"
        )

        self.left_btn = QPushButton("◀")
        self.left_btn.setFixedSize(32, 28)
        self.left_btn.setToolTip("Previous variant (←)")
        self.left_btn.clicked.connect(self.left_clicked.emit)
        layout.addWidget(self.left_btn)

        self.center = QLabel()
        self.center.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.center.setMinimumWidth(220)
        layout.addWidget(self.center)

        # dot indicators
        self._dots: list[QPushButton] = []
        dots_wrap = QHBoxLayout()
        dots_wrap.setSpacing(4)
        for idx in range(len(self._variants)):
            dot = QPushButton("●" if idx == self._current else "○")
            dot.setFixedSize(22, 22)
            dot.setToolTip(self._variants[idx])
            dot.clicked.connect(lambda _c=False, i=idx: self.variant_selected.emit(i))
            dot.setStyleSheet(
                "QPushButton { background:transparent; border:none; color:#aab0c6; font-size:10px; } QPushButton:hover { color:white; }"
            )
            self._dots.append(dot)
            dots_wrap.addWidget(dot)
        dots_container = QWidget()
        dots_container.setLayout(dots_wrap)
        dots_container.setStyleSheet("background:transparent;")
        layout.addWidget(dots_container)

        self.right_btn = QPushButton("▶")
        self.right_btn.setFixedSize(32, 28)
        self.right_btn.setToolTip("Next variant (→)")
        self.right_btn.clicked.connect(self.right_clicked.emit)
        layout.addWidget(self.right_btn)

        self._refresh()

    def set_current(self, idx: int) -> None:
        self._current = int(idx) % len(self._variants)
        self._refresh()

    def _refresh(self) -> None:
        self.center.setText(self._variants[self._current])
        for i, dot in enumerate(self._dots):
            dot.setText("●" if i == self._current else "○")
            dot.setStyleSheet(
                "QPushButton { background:transparent; border:none; color:white; font-size:11px; font-weight:700; } "
                if i == self._current
                else "QPushButton { background:transparent; border:none; color:#7a80a0; font-size:10px; }"
            )


# ---------------------------------------------------------------------------
# Prototype main window (realistic split-pane chrome + three Data variants)
# ---------------------------------------------------------------------------


class PrototypeMainWindow(QMainWindow):
    """Throwaway main window that hosts three Data-pane preview variants."""

    def __init__(self, initial_variant: int = 0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            "THROWAWAY PROTOTYPE — Data Pane Preview Layouts  A / B / C  ·  measure-save-preview-gallery"
        )
        self.resize(1280, 780)
        self.setFocusPolicy(Qt.StrongFocus)  # type: ignore[attr-defined]

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # banner
        banner = QLabel(
            "⦿ THROWAWAY PROTOTYPE — synthetic in-memory figures · no saves will be written · "
            "switch Data layouts with the pill below or ←/→ keys"
        )
        banner.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        banner.setStyleSheet(
            "background:#fff3cd; border:1px solid #ffe69c; color:#664d03; padding:7px 8px; border-radius:4px; font-size:11px; font-weight:600;"
        )
        banner.setWordWrap(True)
        outer.addWidget(banner)

        # toolbar mimic (realistic density reference)
        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.setSpacing(6)
        new_tab = QPushButton("New Tab ▾")
        new_tab.setEnabled(False)
        new_tab.setToolTip("Prototype — disabled")
        toolbar_row.addWidget(new_tab)
        toolbar_row.addStretch()
        for label in ("Setup…", "Devices…", "Predictor…", "Inspect…"):
            b = QPushButton(label)
            b.setEnabled(False)
            b.setToolTip("Prototype — disabled")
            toolbar_row.addWidget(b)
        outer.addLayout(toolbar_row)

        # context bar mimic
        ctx_row = QHBoxLayout()
        ctx_row.setContentsMargins(0, 0, 0, 0)
        ctx_row.addWidget(QLabel("Context:"))
        ctx_val = QLabel("demo-project — demo-data.hdf5")
        ctx_val.setStyleSheet("color:#495057;")
        ctx_row.addWidget(ctx_val)
        ctx_row.addSpacing(18)
        ctx_row.addWidget(QLabel("Predictor:"))
        pred = QLabel("none")
        pred.setStyleSheet("color:#6c757d;")
        ctx_row.addWidget(pred)
        ctx_row.addStretch()
        ctx_info = QLabel("Prototype — no project persistence")
        ctx_info.setStyleSheet(
            "color:#856404; font-size:11px; border:1px solid #ffe69c; background:#fff8e1; padding:2px 6px; border-radius:4px;"
        )
        ctx_row.addWidget(ctx_info)
        outer.addLayout(ctx_row)

        # availability + variant state row
        state_row = QWidget()
        state_row.setStyleSheet(
            "background:#f1f3f5; border:1px solid #dee2e6; border-radius:6px;"
        )
        sh = QHBoxLayout(state_row)
        sh.setContentsMargins(8, 6, 8, 6)
        sh.setSpacing(10)
        sh.addWidget(QLabel("<b>Preview state:</b>"))
        self.run_chk = QCheckBox("Run AVAILABLE")
        self.run_chk.setChecked(True)
        self.run_chk.setToolTip("Toggle Run figure AVAILABLE / NO RESULT")
        sh.addWidget(self.run_chk)
        self.analysis_chk = QCheckBox("Analysis AVAILABLE")
        self.analysis_chk.setChecked(True)
        self.analysis_chk.setToolTip("Toggle Analysis figure AVAILABLE / NO FIGURE")
        sh.addWidget(self.analysis_chk)
        self.post_chk = QCheckBox("Post AVAILABLE")
        self.post_chk.setChecked(True)
        self.post_chk.setToolTip("Toggle Post-Analysis figure AVAILABLE / NO FIGURE")
        sh.addWidget(self.post_chk)
        sh.addSpacing(12)
        self.variant_label = QLabel()
        self.variant_label.setStyleSheet(
            "font-weight:700; color:#212529; background:transparent; border:none;"
        )
        sh.addWidget(self.variant_label)
        sh.addStretch()
        self.state_summary = QLabel()
        self.state_summary.setStyleSheet(
            "color:#495057; font-size:11px; background:transparent; border:none;"
        )
        self.state_summary.setWordWrap(True)
        sh.addWidget(self.state_summary)
        outer.addWidget(state_row)

        # splitter
        self._splitter = QSplitter(Qt.Horizontal)  # type: ignore[attr-defined]
        outer.addWidget(self._splitter, stretch=1)

        # left tabs
        self.left_tabs = QTabWidget()
        self.left_tabs.setDocumentMode(False)
        # Run
        self.left_tabs.addTab(self._build_run_panel(), "Run")
        # Analysis
        self.left_tabs.addTab(self._build_analysis_panel(), "Analysis")
        # Post
        self.left_tabs.addTab(self._build_post_panel(), "Post-Analysis")
        # Data — stacked left variants
        self.data_left_stack = QStackedWidget()
        self._data_left_a = _CompactSaveControls()
        self._data_left_b = _InlineCardsWidget()
        self._data_left_c = _CompactSaveControls()
        self.data_left_stack.addWidget(self._data_left_a)
        self.data_left_stack.addWidget(self._data_left_b)
        self.data_left_stack.addWidget(self._data_left_c)
        data_left_wrap = QWidget()
        dlw_l = QVBoxLayout(data_left_wrap)
        dlw_l.setContentsMargins(0, 0, 0, 0)
        dlw_l.addWidget(self.data_left_stack)
        self.left_tabs.addTab(data_left_wrap, "Data")
        # Guide
        self.left_tabs.addTab(self._build_guide_panel(), "Guide")

        self._splitter.addWidget(self.left_tabs)

        # right stack
        self.right_stack = QStackedWidget()
        self._run_right = self._build_run_right()
        self._analysis_right = self._build_analysis_right()
        self._post_right = self._build_post_right()

        self.data_right_stack = QStackedWidget()
        self._data_right_a = self._build_data_right_stacked()
        self._data_right_b = self._build_data_right_placeholder()
        self._data_right_c = self._build_data_right_tabbed()
        self.data_right_stack.addWidget(self._data_right_a)
        self.data_right_stack.addWidget(self._data_right_b)
        self.data_right_stack.addWidget(self._data_right_c)

        self._guide_right = QLabel("(no plot — Guide is documentation)")
        self._guide_right.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self._guide_right.setStyleSheet(
            "color:#6c757d; border:1px dashed #ced4da; background:#f8f9fa; padding:24px;"
        )

        self.right_stack.addWidget(self._run_right)  # 0
        self.right_stack.addWidget(self._analysis_right)  # 1
        self.right_stack.addWidget(self._post_right)  # 2
        self.right_stack.addWidget(self.data_right_stack)  # 3
        self.right_stack.addWidget(self._guide_right)  # 4

        self._splitter.addWidget(self.right_stack)
        self._splitter.setSizes([520, 720])
        self._splitter.setCollapsible(0, True)
        self._splitter.setCollapsible(1, False)

        self.left_tabs.currentChanged.connect(self._on_left_tab_changed)

        # switcher pill (in-window)
        switcher_wrap = QWidget()
        switcher_wrap.setStyleSheet("background:transparent;")
        sh2 = QHBoxLayout(switcher_wrap)
        sh2.setContentsMargins(0, 6, 0, 2)
        sh2.addStretch()
        self.switcher = _SwitcherBar(
            variants=["A · Stacked Right Rail", "B · Inline Cards", "C · Focus Tabs"],
            current=int(initial_variant) % 3,
        )
        self.switcher.left_clicked.connect(self._prev_variant)
        self.switcher.right_clicked.connect(self._next_variant)
        self.switcher.variant_selected.connect(self._set_variant)
        sh2.addWidget(self.switcher)
        sh2.addStretch()
        outer.addWidget(switcher_wrap)

        hint = QLabel(
            "Tip: Use ← / → to cycle variants. Arrow keys are ignored while typing in any path or comment field."
        )
        hint.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        hint.setStyleSheet("color:#6c757d; font-size:11px;")
        outer.addWidget(hint)

        self._variant_index = int(initial_variant) % 3
        self._variant_names = [
            "A · Stacked Right Rail",
            "B · Inline Cards",
            "C · Focus Tabs",
        ]

        self.run_chk.toggled.connect(self._on_availability_changed)
        self.analysis_chk.toggled.connect(self._on_availability_changed)
        self.post_chk.toggled.connect(self._on_availability_changed)

        # start on Data tab so previews are immediately visible
        for i in range(self.left_tabs.count()):
            if self.left_tabs.tabText(i) == "Data":
                self.left_tabs.setCurrentIndex(i)
                break

        self._apply_variant()
        self._on_availability_changed()
        self.statusBar().showMessage(
            "Prototype ready — compare A/B/C. All Save/Load actions are inert (no persistence)."
        )

    # -- panel builders ------------------------------------------------

    def _build_run_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)  # type: ignore[attr-defined]
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]
        il.setSpacing(6)
        title = QLabel(
            "<b>Run configuration</b> <span style='color:#6c757d;'>(prototype placeholder)</span>"
        )
        title.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        il.addWidget(title)
        for label in (
            "Qubit frequency (GHz)",
            "Flux bias (Φ₀)",
            "Readout length (ns)",
            "Averaging reps",
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(QLineEdit("5.05"))
            il.addLayout(row)
        note = QLabel(
            "Config form density mirrors real ExpTab Run tab; editing is local only in prototype."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#6c757d; font-size:11px;")
        il.addWidget(note)
        il.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll, stretch=1)
        bar = QFrame()
        bar.setFrameShape(QFrame.StyledPanel)  # type: ignore[attr-defined]
        bar.setStyleSheet(
            "background:#f8f9fa; border:1px solid #dee2e6; border-radius:4px;"
        )
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(8, 6, 8, 6)
        reset = QPushButton("Reset")
        reset.setFlat(True)
        reset.setEnabled(False)
        reset.setToolTip("Prototype — disabled")
        bl.addWidget(reset, stretch=20)
        run = QPushButton("Run")
        run.setEnabled(False)
        run.setFixedHeight(30)
        run.setStyleSheet(
            "background:#286ac7; color:white; font-weight:600; border-radius:4px;"
        )
        bl.addWidget(run, stretch=80)
        layout.addWidget(bar)
        return w

    def _build_analysis_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)  # type: ignore[attr-defined]
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]
        title = QLabel(
            "<b>Analysis parameters</b> <span style='color:#6c757d;'>(prototype)</span>"
        )
        title.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        il.addWidget(title)
        for label in ("Fit model", "Window (MHz)", "Threshold"):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(QLineEdit(" — "))
            il.addLayout(row)
        btn = QPushButton("Analyze")
        btn.setEnabled(False)
        btn.setFixedHeight(30)
        btn.setStyleSheet(
            "background:#286ac7; color:white; font-weight:600; border-radius:4px;"
        )
        il.addWidget(btn)
        wb = QLabel("<b>Writeback preview</b> — no entries (prototype)")
        wb.setStyleSheet(
            "color:#6c757d; font-size:11px; border:1px solid #e9ecef; background:#f8f9fa; padding:6px; border-radius:4px;"
        )
        il.addWidget(wb)
        il.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll, stretch=1)
        return w

    def _build_post_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)  # type: ignore[attr-defined]
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]
        title = QLabel(
            "<b>Post-Analysis</b> <span style='color:#6c757d;'>(prototype)</span>"
        )
        title.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        il.addWidget(title)
        hint = QLabel("Run analyze first to enable post-analysis. (prototype — gated)")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6c757d; font-size:11px;")
        il.addWidget(hint)
        btn = QPushButton("Run Post-Analysis")
        btn.setEnabled(False)
        il.addWidget(btn)
        il.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll, stretch=1)
        return w

    def _build_guide_panel(self) -> QWidget:
        w = QScrollArea()
        w.setWidgetResizable(True)
        inner = QLabel(
            "<h3>Guide</h3>"
            "<p><b>Behavior</b><br>Measure-gui Data subtab will show Run / Analysis / Post-Analysis "
            "figure previews beside their save destinations. Capability-driven absence (no analysis support) "
            "remains distinct from a supported pane that currently has no figure.</p>"
            "<p><b>Prototype scope</b><br>This is a throwaway Qt prototype. All figures are synthetic "
            "and all Save/Load actions are inert. Use the pill or ←/→ to compare the three preview "
            "layouts and choose or mix a direction for production.</p>"
            "<p style='color:#6c757d; font-size:11px;'>Known limitations: no hardware, no persistence, "
            "no figure editing, no real analysis controls — preview is read-only.</p>"
        )
        inner.setWordWrap(True)
        inner.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        inner.setContentsMargins(8, 8, 8, 8)
        inner.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]
        w.setWidget(inner)
        return w

    def _build_run_right(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(
            QLabel(
                "<b>Run figure — large preview</b> <span style='color:#6c757d;'>(prototype)</span>"
            ),
            stretch=0,
        )
        card = _PreviewCard("run", "Run", _make_run_fig)
        self._run_right_card = card  # type: ignore[attr-defined]
        layout.addWidget(card, stretch=1)
        foot = QLabel("Read-only — not interactive (prototype)")
        foot.setStyleSheet("color:#6c757d; font-size:11px;")
        foot.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        layout.addWidget(foot)
        return w

    def _build_analysis_right(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(
            QLabel(
                "<b>Analysis figure</b> <span style='color:#6c757d;'>(prototype)</span>"
            ),
            stretch=0,
        )
        card = _PreviewCard("analysis", "Analysis", _make_analysis_fig)
        self._analysis_right_card = card  # type: ignore[attr-defined]
        layout.addWidget(card, stretch=1)
        return w

    def _build_post_right(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(
            QLabel(
                "<b>Post-Analysis figure</b> <span style='color:#6c757d;'>(prototype)</span>"
            ),
            stretch=0,
        )
        card = _PreviewCard("post", "Post-Analysis", _make_post_fig)
        self._post_right_card = card  # type: ignore[attr-defined]
        layout.addWidget(card, stretch=1)
        return w

    def _build_data_right_stacked(self) -> QWidget:
        """Variant A right rail: three previews stacked vertically."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        title = QLabel(
            "<b>Data — Preview rail</b> &nbsp;<span style='color:#6c757d; font-size:11px;'>Variant A · Stacked Right Rail</span>"
        )
        title.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        layout.addWidget(title)
        hint = QLabel(
            "Three named sources stacked vertically. Run preview is taller (2×) to emphasize measurement; Analysis/Post are compact but readable."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6c757d; font-size:11px;")
        layout.addWidget(hint)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)  # type: ignore[attr-defined]
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setSpacing(10)
        self._card_a_run = _PreviewCard("run", "Run", _make_run_fig)
        self._card_a_run.setMinimumHeight(240)
        il.addWidget(self._card_a_run, stretch=2)
        self._card_a_analysis = _PreviewCard("analysis", "Analysis", _make_analysis_fig)
        self._card_a_analysis.setMinimumHeight(180)
        il.addWidget(self._card_a_analysis, stretch=1)
        self._card_a_post = _PreviewCard("post", "Post-Analysis", _make_post_fig)
        self._card_a_post.setMinimumHeight(180)
        il.addWidget(self._card_a_post, stretch=1)
        il.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll, stretch=1)
        foot = QLabel(
            "Read-only previews — paired to left save rows by color-coded left border (prototype, no interaction)"
        )
        foot.setStyleSheet("color:#6c757d; font-size:10px;")
        foot.setWordWrap(True)
        layout.addWidget(foot)
        return w

    def _build_data_right_placeholder(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        icon = QLabel("◫")
        icon.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        icon.setStyleSheet("font-size:42px; color:#adb5bd;")
        layout.addWidget(icon)
        msg = QLabel(
            "<b>Variant B · Inline Cards</b><br>Previews are embedded directly below each save row in the left Data pane.<br><span style='color:#6c757d; font-size:11px;'>Right pane is idle for Data in this layout — proximity pairing is maximal.</span>"
        )
        msg.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        msg.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        msg.setWordWrap(True)
        msg.setStyleSheet("color:#495057; font-size:12px;")
        layout.addWidget(msg)
        layout.addStretch()
        return w

    def _build_data_right_tabbed(self) -> QWidget:
        """Variant C right focus: single large preview with pill tabs."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        title = QLabel(
            "<b>Data — Focus viewer</b> &nbsp;<span style='color:#6c757d; font-size:11px;'>Variant C · Focus Tabs</span>"
        )
        title.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        layout.addWidget(title)
        hint = QLabel(
            "One large preview at a time for maximum axis/legend readability. Tab between sources."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6c757d; font-size:11px;")
        layout.addWidget(hint)

        self._tabbed_viewer = QTabWidget()
        # each tab holds a large preview card
        self._card_c_run = _PreviewCard("run", "Run", _make_run_fig)
        run_wrap = QWidget()
        rl = QVBoxLayout(run_wrap)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.addWidget(self._card_c_run)
        self._tabbed_viewer.addTab(run_wrap, "Run")

        self._card_c_analysis = _PreviewCard("analysis", "Analysis", _make_analysis_fig)
        ana_wrap = QWidget()
        al = QVBoxLayout(ana_wrap)
        al.setContentsMargins(4, 4, 4, 4)
        al.addWidget(self._card_c_analysis)
        self._tabbed_viewer.addTab(ana_wrap, "Analysis")

        self._card_c_post = _PreviewCard("post", "Post-Analysis", _make_post_fig)
        post_wrap = QWidget()
        pl = QVBoxLayout(post_wrap)
        pl.setContentsMargins(4, 4, 4, 4)
        pl.addWidget(self._card_c_post)
        self._tabbed_viewer.addTab(post_wrap, "Post-Analysis")

        layout.addWidget(self._tabbed_viewer, stretch=1)
        foot = QLabel(
            "Selected tab is the only visible figure — largest axes for this window width (prototype, read-only)"
        )
        foot.setStyleSheet("color:#6c757d; font-size:10px;")
        foot.setWordWrap(True)
        layout.addWidget(foot)
        return w

    # -- variant / availability handling --------------------------------

    def _on_left_tab_changed(self, idx: int) -> None:
        text = self.left_tabs.tabText(idx)
        if text == "Run":
            self.right_stack.setCurrentIndex(0)
        elif text == "Analysis":
            self.right_stack.setCurrentIndex(1)
        elif text == "Post-Analysis":
            self.right_stack.setCurrentIndex(2)
        elif text == "Data":
            self.right_stack.setCurrentIndex(3)
        elif text == "Guide":
            self.right_stack.setCurrentIndex(4)

    def _prev_variant(self) -> None:
        self._variant_index = (self._variant_index - 1) % 3
        self._apply_variant()

    def _next_variant(self) -> None:
        self._variant_index = (self._variant_index + 1) % 3
        self._apply_variant()

    def _set_variant(self, idx: int) -> None:
        self._variant_index = int(idx) % 3
        self._apply_variant()

    def _apply_variant(self) -> None:
        self.data_left_stack.setCurrentIndex(self._variant_index)
        self.data_right_stack.setCurrentIndex(self._variant_index)
        name = self._variant_names[self._variant_index]
        self.variant_label.setText(f"Variant: {name}")
        self.switcher.set_current(self._variant_index)
        self._update_summary()
        self.statusBar().showMessage(
            f"Variant {name} — use ←/→ or pill to compare (all actions inert)", 4000
        )

    def _on_availability_changed(self) -> None:
        run = self.run_chk.isChecked()
        ana = self.analysis_chk.isChecked()
        post = self.post_chk.isChecked()

        # left compact controls (A/C)
        for ctrl in (self._data_left_a, self._data_left_c):
            try:
                ctrl.set_availability(run, ana, post)  # type: ignore[attr-defined]
            except Exception:
                pass
        # inline B
        try:
            self._data_left_b.set_availability(run, ana, post)  # type: ignore[attr-defined]
        except Exception:
            pass

        # right stacked A
        for card, avail in (
            (getattr(self, "_card_a_run", None), run),
            (getattr(self, "_card_a_analysis", None), ana),
            (getattr(self, "_card_a_post", None), post),
        ):
            try:
                if card is not None:
                    card.set_available(avail)
            except Exception:
                pass
        # right tabbed C
        for card, avail in (
            (getattr(self, "_card_c_run", None), run),
            (getattr(self, "_card_c_analysis", None), ana),
            (getattr(self, "_card_c_post", None), post),
        ):
            try:
                if card is not None:
                    card.set_available(avail)
            except Exception:
                pass
        # run/analysis/post right solo previews also reflect availability for realism
        for card, avail in (
            (getattr(self, "_run_right_card", None), run),
            (getattr(self, "_analysis_right_card", None), ana),
            (getattr(self, "_post_right_card", None), post),
        ):
            try:
                if card is not None:
                    card.set_available(avail)
            except Exception:
                pass

        self._update_summary()

    def _update_summary(self) -> None:
        run = "AVAILABLE" if self.run_chk.isChecked() else "NO RESULT"
        ana = "AVAILABLE" if self.analysis_chk.isChecked() else "NO FIGURE"
        post = "AVAILABLE" if self.post_chk.isChecked() else "NO FIGURE"
        name = self._variant_names[self._variant_index]
        self.state_summary.setText(
            f"{name} · Run: {run} · Analysis: {ana} · Post: {post}"
        )

    # -- safe arrow-key handling ----------------------------------------

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QTextEdit)):
            # do not steal arrow keys while typing
            super().keyPressEvent(event)
            return
        # contenteditable check via property
        try:
            if focus is not None and focus.property("isWrapping") is not None:
                # fallback: if focus has text interaction, don't steal
                pass
        except Exception:
            pass
        key = event.key()
        if key == Qt.Key_Left:  # type: ignore[attr-defined]
            self._prev_variant()
            event.accept()
            return
        if key == Qt.Key_Right:  # type: ignore[attr-defined]
            self._next_variant()
            event.accept()
            return
        super().keyPressEvent(event)


def launch(initial_variant: int = 0) -> PrototypeMainWindow:
    """Create and show the prototype window (for embedding in launcher)."""
    win = PrototypeMainWindow(initial_variant=initial_variant)
    win.show()
    return win
