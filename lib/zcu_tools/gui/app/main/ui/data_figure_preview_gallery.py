"""DataFigurePreviewGallery — Variant A stacked raster preview rail.

App-local Qt presentation Module for the measure-gui Data subtab (S1).

It accepts a capability declaration at construction and a complete
``Figure`` snapshot via :meth:`update_figures`, rendering each available
``Figure`` to a fixed-size PNG via an injected ``Figure -> bytes``
callable (S3). Production uses :func:`render_figure_png`; tests inject a
deterministic fake. Each card is rendered independently — a single
card failure produces a named unavailable state without blocking
other cards or the save controls.

Ownership (S2):
- Never retains a matplotlib ``Figure`` or its canvas — only the
  presentation cache (pixmap and text state) is held.
- Never calls ``attach``/``reparent``/plot routing; the owning
  :class:`ExpTabWidget` stays the sole ``FigureContainer`` authority.
- No timer or per-draw subscription — refresh is snapshot-driven by
  Data activation and the Data-visible prepare/clear/show lifecycle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QPixmap  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.main.adapter import AnalysisMode

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.gui.app.main.adapter import AdapterCapabilities

logger = logging.getLogger(__name__)

_EMPTY_TEXT = "— No figure"
_UNAVAILABLE_TEXT = "— Preview unavailable"


class _PreviewCard(QWidget):
    """Single named source preview card (header + raster/empty/error)."""

    def __init__(
        self,
        source_key: str,
        display_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source_key = source_key
        self._display_name = display_name
        self.setObjectName(f"previewCard_{source_key}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        self.setStyleSheet(
            "QWidget#previewCard_run, QWidget#previewCard_analysis, QWidget#previewCard_post "
            "{ border:1px solid #dee2e6; border-radius:6px; background:white; }"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # type: ignore[attr-defined]

        # Header
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel(f"<b>{display_name}</b>")
        title.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        title.setObjectName(f"previewTitle_{source_key}")
        header.addWidget(title)
        header.addStretch()
        outer.addLayout(header)

        # Stack: image / empty / error
        self._stack = QStackedWidget()
        self._stack.setObjectName(f"previewStack_{source_key}")

        # Image page
        image_page = QWidget()
        image_page.setObjectName(f"previewImagePage_{source_key}")
        img_layout = QVBoxLayout(image_page)
        img_layout.setContentsMargins(0, 0, 0, 0)
        self._image_label = QLabel()
        self._image_label.setObjectName(f"previewImage_{source_key}")
        self._image_label.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self._image_label.setScaledContents(False)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # type: ignore[attr-defined]
        self._image_label.setMinimumHeight(120)
        img_layout.addWidget(self._image_label)
        self._stack.addWidget(image_page)
        self._image_page = image_page

        # Empty page — supported but no figure
        empty_page = QWidget()
        empty_page.setObjectName(f"previewEmptyPage_{source_key}")
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.setContentsMargins(8, 12, 8, 12)
        empty_layout.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self._empty_label = QLabel(f"{display_name} — {_EMPTY_TEXT}\n(no figure yet)")
        self._empty_label.setObjectName(f"previewEmpty_{source_key}")
        self._empty_label.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet(
            "color:#6c757d; border:1px dashed #adb5bd; background:#f8f9fa; "
            "padding:12px 8px; border-radius:6px; font-size:11px;"
        )
        empty_layout.addWidget(self._empty_label)
        self._stack.addWidget(empty_page)
        self._empty_page = empty_page

        # Error page — render failure
        error_page = QWidget()
        error_page.setObjectName(f"previewErrorPage_{source_key}")
        error_layout = QVBoxLayout(error_page)
        error_layout.setContentsMargins(8, 12, 8, 12)
        error_layout.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self._error_label = QLabel(f"{display_name} — {_UNAVAILABLE_TEXT}")
        self._error_label.setObjectName(f"previewError_{source_key}")
        self._error_label.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet(
            "color:#842029; border:1px solid #f5c2c7; background:#f8d7da; "
            "padding:12px 8px; border-radius:6px; font-size:11px;"
        )
        error_layout.addWidget(self._error_label)
        self._stack.addWidget(error_page)
        self._error_page = error_page

        outer.addWidget(self._stack, stretch=1)

        # Initial state: empty (supported but no figure yet)
        self.show_empty()

    def show_empty(self) -> None:
        self._stack.setCurrentWidget(self._empty_page)

    def show_unavailable(self, detail: str | None = None) -> None:
        if detail:
            self._error_label.setText(
                f"{self._display_name} — {_UNAVAILABLE_TEXT}\n{detail}"
            )
        else:
            self._error_label.setText(f"{self._display_name} — {_UNAVAILABLE_TEXT}")
        self._stack.setCurrentWidget(self._error_page)

    def show_image(self, pixmap: QPixmap) -> None:
        self._image_label.setPixmap(pixmap)
        self._stack.setCurrentWidget(self._image_page)

    def current_state(self) -> str:
        """Return 'available' | 'empty' | 'unavailable' for test inspection."""
        cur = self._stack.currentWidget()
        if cur is self._image_page:
            return "available"
        if cur is self._empty_page:
            return "empty"
        if cur is self._error_page:
            return "unavailable"
        return "unknown"

    def current_text(self) -> str:
        """Return visible label text for the current state."""
        cur = self._stack.currentWidget()
        if cur is self._image_page:
            return ""
        if cur is self._empty_page:
            return self._empty_label.text()
        if cur is self._error_page:
            return self._error_label.text()
        return ""


class DataFigurePreviewGallery(QWidget):
    """Variant A stacked raster preview rail (S1, S3).

    Construction declares Analysis/Post-Analysis capability; presentation
    is a scrollable vertical rail of named cards. :meth:`update_figures`
    accepts a complete snapshot; each card renders independently via the
    injected ``Figure -> PNG bytes`` callable. A single card failure
    produces a named unavailable state and is logged, without blocking
    other cards.

    The gallery holds only pixmap/text cache — never a ``Figure``,
    canvas, result, or save state (S2).
    """

    def __init__(
        self,
        capabilities: AdapterCapabilities,
        parent: QWidget | None = None,
        *,
        renderer: Callable[[Figure], bytes] | None = None,
    ) -> None:
        super().__init__(parent)
        if capabilities is None:
            raise TypeError("DataFigurePreviewGallery requires AdapterCapabilities")
        # Import here to avoid circular import at module load.
        from zcu_tools.gui.app.main.adapter import AdapterCapabilities as _Caps

        if not isinstance(capabilities, _Caps):
            raise TypeError(
                f"DataFigurePreviewGallery requires AdapterCapabilities, got {type(capabilities).__name__!r}"
            )
        self._capabilities = capabilities
        self._has_analysis = capabilities.analysis is not AnalysisMode.NONE
        self._has_post = bool(capabilities.post_analysis)
        self._renderer: Callable[[Figure], bytes]
        if renderer is not None:
            if not callable(renderer):
                raise TypeError("renderer must be callable")
            self._renderer = renderer
        else:
            # Lazy import of production fixed-size renderer.
            from zcu_tools.gui.app.main.figure_export import render_figure_png

            self._renderer = render_figure_png

        self.setObjectName("dataFigurePreviewGallery")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        header = QLabel(
            "<b>Figure previews</b> <span style='color:#6c757d; font-weight:normal;'>(read-only)</span>"
        )
        header.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        header.setObjectName("previewGalleryHeader")
        outer.addWidget(header)
        hint = QLabel(
            "Run is always shown; Analysis/Post appear when supported by this experiment."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6c757d; font-size:11px;")
        hint.setObjectName("previewGalleryHint")
        outer.addWidget(hint)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("previewGalleryScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)  # type: ignore[attr-defined]
        inner = QWidget()
        inner.setObjectName("previewGalleryInner")
        self._inner_layout = QVBoxLayout(inner)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setSpacing(10)
        self._inner_layout.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]

        # Cards — Run always, Analysis/Post per capability
        self._cards: dict[str, _PreviewCard] = {}

        run_card = _PreviewCard("run", "Run")
        self._cards["run"] = run_card
        self._inner_layout.addWidget(run_card, stretch=2)

        if self._has_analysis:
            ana_card = _PreviewCard("analysis", "Analysis")
            self._cards["analysis"] = ana_card
            self._inner_layout.addWidget(ana_card, stretch=1)

        if self._has_post:
            post_card = _PreviewCard("post", "Post-Analysis")
            # Normalize key: use "post_analysis" externally but "post" internally? Use "post_analysis" for update_figures.
            self._cards["post_analysis"] = post_card
            # Also expose alias "post" for convenience
            self._cards["post"] = post_card
            self._inner_layout.addWidget(post_card, stretch=1)

        self._inner_layout.addStretch()
        self._scroll.setWidget(inner)
        outer.addWidget(self._scroll, stretch=1)

        # Initial empty state for all present cards
        for card in self._cards.values():
            # Avoid double-setting for alias
            if card.current_state() != "empty":
                card.show_empty()

    # -- capability queries -----------------------------------------

    @property
    def has_analysis(self) -> bool:
        return self._has_analysis

    @property
    def has_post(self) -> bool:
        return self._has_post

    def has_card(self, key: str) -> bool:
        """Return True if a card for ``key`` exists (run|analysis|post_analysis|post)."""
        norm = "post_analysis" if key == "post" else key
        if norm == "post_analysis" and "post_analysis" in self._cards:
            return True
        return key in self._cards

    def card_state(self, key: str) -> str:
        """Return card state: 'available' | 'empty' | 'unavailable'."""
        card = self._cards.get(key)
        if card is None and key == "post":
            card = self._cards.get("post_analysis")
        if card is None and key == "post_analysis":
            card = self._cards.get("post")
        if card is None:
            raise KeyError(f"no preview card for {key!r}")
        return card.current_state()

    def card_text(self, key: str) -> str:
        """Return visible text for the card's current state (empty/error)."""
        card = self._cards.get(key)
        if card is None and key == "post":
            card = self._cards.get("post_analysis")
        if card is None and key == "post_analysis":
            card = self._cards.get("post")
        if card is None:
            raise KeyError(f"no preview card for {key!r}")
        return card.current_text()

    def card_count(self) -> int:
        """Return number of distinct preview cards (Run + optional)."""
        # Deduplicate alias
        keys = {k for k in self._cards if k != "post"}
        return len(keys)

    # -- snapshot update ---------------------------------------------

    def update_figures(
        self,
        run: Figure | None,
        analysis: Figure | None = None,
        post_analysis: Figure | None = None,
        **kwargs: Figure | None,
    ) -> None:
        """Refresh all cards from a complete figure snapshot (S2).

        ``run`` is always considered; ``analysis`` and ``post_analysis``
        are only rendered when their capability is present. ``None``
        shows the named empty state; a renderer exception shows the
        named unavailable state for that card only.

        The gallery never retains the ``Figure`` objects — only the
        resulting pixmap/text cache is held.
        """
        # Support alias kwargs for flexibility (post/post_analysis)
        if "post" in kwargs and post_analysis is None:
            post_analysis = kwargs["post"]
        # Normalize: do not hold refs beyond this call.
        self._update_card("run", run)
        if self._has_analysis:
            self._update_card("analysis", analysis)
        if self._has_post:
            self._update_card("post_analysis", post_analysis)

    def _update_card(self, key: str, fig: Figure | None) -> None:
        card = self._cards.get(key)
        if card is None:
            # Try alias
            if key == "post_analysis":
                card = self._cards.get("post")
            elif key == "post":
                card = self._cards.get("post_analysis")
        if card is None:
            # Unsupported card — do nothing (S4).
            return
        if fig is None:
            card.show_empty()
            return
        # Render via injected adapter (S3) — per-card isolated.
        try:
            png_bytes = self._renderer(fig)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - exercised via injected failure
            logger.warning("preview render failed for %r: %s", key, exc, exc_info=True)
            card.show_unavailable(type(exc).__name__)
            return
        # Convert PNG bytes to pixmap
        try:
            pixmap = QPixmap()
            if not isinstance(png_bytes, (bytes, bytearray)):
                raise TypeError(
                    f"renderer returned {type(png_bytes).__name__!r}, expected bytes"
                )
            if not pixmap.loadFromData(png_bytes, "PNG"):
                raise RuntimeError("pixmap loadFromData failed")
            # Optional scaling to gallery width is handled by the outer scroll
            # and label's size policy; keep the pixmap at its rendered size.
            card.show_image(pixmap)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("preview pixmap failed for %r: %s", key, exc, exc_info=True)
            card.show_unavailable(type(exc).__name__)

    # -- test helpers ------------------------------------------------

    def find_card(self, key: str) -> _PreviewCard | None:
        """Return the card widget for ``key`` (or None if unsupported)."""
        card = self._cards.get(key)
        if card is None and key in ("post", "post_analysis"):
            card = self._cards.get("post_analysis") or self._cards.get("post")
        return card
