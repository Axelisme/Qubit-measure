"""DataFigurePreviewGallery — Variant A stacked raster preview rail.

App-local Qt presentation Module for the measure-gui Data subtab (S1).

It accepts a capability declaration at construction and a complete
``Figure`` snapshot via :meth:`update_figures`, rendering each available
``Figure`` to a fixed-size PNG via an injected ``Figure -> bytes``
callable (S3). Production uses :func:`render_figure_preview_png`; tests inject a
deterministic fake. Each card is rendered independently — a single
card failure produces a named unavailable state without blocking
other cards or the save controls.

Responsive mosaic (S1/S4):
- Breakpoint is viewport-driven: the gallery's own width (scroll
  viewport if available) is checked against two minimum-width cards
  plus spacing. Wide when available >= 2*MIN_CARD_WIDTH + spacing,
  otherwise narrow.
- Narrow: vertical single column (Run, Analysis, Post-Analysis).
- Wide + 3 cards: Run left spanning two rows, Analysis top-right,
  Post-Analysis bottom-right.
- Wide + 2 cards: Run and Analysis side-by-side.
- Single card: full width.

Ownership (S2):
- Never retains a matplotlib ``Figure`` or its canvas — only the
  presentation cache (pixmap and text state) is held.
- Never calls ``attach``/``reparent``/plot routing; the owning
  :class:`ExpTabWidget` stays the sole ``FigureContainer`` authority.
- No timer or per-draw subscription — refresh is snapshot-driven by
  Data activation and the Data-visible prepare/clear/show lifecycle.

Render cache (S3):
- Each available card caches the original pixmap; on image-viewport
  resize the displayed pixmap is recomputed with KeepAspectRatio and
  SmoothTransformation so it never exceeds the viewport or crops.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from qtpy.QtCore import QEvent, QObject, QSize, Qt
from qtpy.QtGui import QPixmap, QResizeEvent, QShowEvent
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.measure.adapter import AnalysisMode
from zcu_tools.gui.widgets import secondary_colour, set_foreground
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from zcu_tools.gui.app.measure.adapter import AdapterCapabilities

logger = logging.getLogger(__name__)

_EMPTY_TEXT = "No figure yet"
_UNAVAILABLE_TEXT = "— Preview unavailable"

# Responsive mosaic — viewport-driven breakpoint (S1).
MIN_CARD_WIDTH = 220
LAYOUT_SPACING = 10
OUTER_MARGIN = 6
WIDE_THRESHOLD = 2 * MIN_CARD_WIDTH + LAYOUT_SPACING  # 450 for inner viewport


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
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(MIN_CARD_WIDTH)

        # Header
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel(f"<b>{display_name}</b>")
        title.setTextFormat(Qt.TextFormat.RichText)
        title.setObjectName(f"previewTitle_{source_key}")
        header.addWidget(title)
        header.addStretch()
        outer.addLayout(header)

        # Stack the image, empty, and error pages in that order.
        self._stack = QStackedWidget()
        self._stack.setObjectName(f"previewStack_{source_key}")

        self._image_page = self._build_image_page()
        # The card header already names the source, so the placeholder does not.
        self._empty_page, self._empty_label = self._build_message_page(
            "Empty",
            _EMPTY_TEXT,
            "color:#6c757d; border:1px dashed #adb5bd; background:#f8f9fa; "
            "padding:12px 8px; border-radius:6px; font-size:11px;",
        )
        self._error_page, self._error_label = self._build_message_page(
            "Error",
            f"{display_name} — {_UNAVAILABLE_TEXT}",
            "color:#842029; border:1px solid #f5c2c7; background:#f8d7da; "
            "padding:12px 8px; border-radius:6px; font-size:11px;",
        )
        # The error text names what failed; the empty text is a placeholder.
        make_label_selectable(self._error_label)

        outer.addWidget(self._stack, stretch=1)

        # Presentation cache — original pixmap, never a Figure (S3).
        self._orig_pixmap: QPixmap | None = None

        # Initial state: empty (supported but no figure yet)
        self.show_empty()

    def _build_image_page(self) -> QWidget:
        """Add the aspect-fit raster viewport as the first stack page."""
        image_page = QWidget()
        image_page.setObjectName(f"previewImagePage_{self._source_key}")
        img_layout = QVBoxLayout(image_page)
        img_layout.setContentsMargins(0, 0, 0, 0)
        self._image_label = QLabel()
        self._image_label.setObjectName(f"previewImage_{self._source_key}")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setScaledContents(False)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._image_label.setMinimumHeight(120)
        self._image_label.setMinimumWidth(120)
        img_layout.addWidget(self._image_label)
        self._stack.addWidget(image_page)
        # Watch image viewport resize so aspect-fit stays within bounds (S3).
        self._image_label.installEventFilter(self)

        return image_page

    def _build_message_page(
        self, name: str, text: str, stylesheet: str
    ) -> tuple[QWidget, QLabel]:
        """Add a centered, wrapping empty/error page and return its text label."""
        page = QWidget()
        page.setObjectName(f"preview{name}Page_{self._source_key}")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel(text)
        label.setObjectName(f"preview{name}_{self._source_key}")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet(stylesheet)
        layout.addWidget(label)
        self._stack.addWidget(page)
        return page, label

    def show_empty(self) -> None:
        self._orig_pixmap = None
        self._stack.setCurrentWidget(self._empty_page)

    def show_unavailable(self, detail: str | None = None) -> None:
        self._orig_pixmap = None
        if detail:
            self._error_label.setText(
                f"{self._display_name} — {_UNAVAILABLE_TEXT}\n{detail}"
            )
        else:
            self._error_label.setText(f"{self._display_name} — {_UNAVAILABLE_TEXT}")
        self._stack.setCurrentWidget(self._error_page)

    def show_image(self, pixmap: QPixmap) -> None:
        # Cache original pixmap (presentation cache, S3).
        self._orig_pixmap = pixmap
        self.rescale_image_to_viewport()
        self._stack.setCurrentWidget(self._image_page)

    def rescale_image_to_viewport(self) -> None:
        """Fit the cached pixmap to the image viewport without cropping.

        Call after a gallery reflow or card resize. Empty/null images are a no-op;
        a not-yet-laid-out label uses the card width as a provisional viewport.
        """
        if self._orig_pixmap is None or self._orig_pixmap.isNull():
            return
        # Viewport is the image label's available size.
        vp = self._image_label.size()
        # If label not yet laid out (0), fallback to card width or defer.
        if vp.width() <= 0 or vp.height() <= 0:
            # Try card size minus margins, or assume minimal viewport.
            cw = self.width()
            self.height()
            # Estimate inner label size: card width minus outer margins (12) and
            # card inner margins, label min 120.
            est_w = max(120, cw - 12) if cw > 0 else 396
            est_h = max(120, vp.height() if vp.height() > 0 else 120)
            vp = QSize(est_w, est_h)
        # KeepAspectRatio + SmoothTransformation per spec; ensure does not exceed viewport.
        scaled = self._orig_pixmap.scaled(
            vp,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a0 is None or a1 is None:
            return super().eventFilter(a0, a1)
        if (
            a0 is self._image_label
            and a1.type() == QEvent.Type.Resize
            and self._orig_pixmap is not None
            and self._stack.currentWidget() is self._image_page
        ):
            self.rescale_image_to_viewport()
        return super().eventFilter(a0, a1)

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        if (
            self._orig_pixmap is not None
            and self._stack.currentWidget() is self._image_page
        ):
            self.rescale_image_to_viewport()

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

    # -- aspect-fit inspection helpers (for tests) -------------------

    def original_pixmap(self) -> QPixmap | None:
        """Return cached original pixmap (presentation cache)."""
        return self._orig_pixmap

    def displayed_pixmap(self) -> QPixmap | None:
        """Return currently displayed (scaled) pixmap, or None."""
        pm = self._image_label.pixmap()
        if pm.isNull():
            return None
        return pm

    def image_viewport_size(self) -> QSize:
        """Return current image label viewport size."""
        return self._image_label.size()


class DataFigurePreviewGallery(QWidget):
    """Variant A stacked raster preview rail (S1, S3).

    Construction declares Analysis/Post-Analysis capability; presentation
    is a scrollable viewport-responsive card mosaic. :meth:`update_figures`
    accepts a complete snapshot; each card renders independently via the
    injected ``Figure -> PNG bytes`` callable. A single card failure
    produces a named unavailable state and is logged, without blocking
    other cards.

    The gallery holds only pixmap/text cache — never a ``Figure``,
    canvas, result, or save state (S2).
    Responsive reflow (S1/S4) is viewport-driven: the gallery's own width
    (scroll viewport when available) decides narrow vs wide at threshold
    2*MIN_CARD_WIDTH + spacing. Aspect-fit scaling (S3) keeps the displayed
    pixmap within the image viewport with KeepAspectRatio.

    Viewport setup tolerates Qt RuntimeError lifetime failures; other setup
    failures propagate from construction.
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
        from zcu_tools.gui.app.measure.adapter import AdapterCapabilities as _Caps

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
            from zcu_tools.gui.app.measure.figure_export import (
                render_figure_preview_png,
            )

            self._renderer = render_figure_preview_png

        self.setObjectName("dataFigurePreviewGallery")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(OUTER_MARGIN, OUTER_MARGIN, OUTER_MARGIN, OUTER_MARGIN)
        outer.setSpacing(6)

        header = QLabel(
            f"<b>Figure previews</b> <span style='color:{secondary_colour(self).name()}; font-weight:normal;'>(read-only)</span>"
        )
        header.setTextFormat(Qt.TextFormat.RichText)
        header.setObjectName("previewGalleryHeader")
        outer.addWidget(header)
        hint = QLabel(
            "Run is always shown; Analysis/Post appear when supported by this experiment."
        )
        hint.setWordWrap(True)
        font = hint.font()
        font.setPixelSize(11)
        hint.setFont(font)
        set_foreground(hint, secondary_colour(self))
        hint.setObjectName("previewGalleryHint")
        outer.addWidget(hint)

        self._build_viewport()
        outer.addWidget(self._scroll, stretch=1)
        # Viewport-driven reflow must track scroll viewport size, not only
        # gallery widget size; QScrollArea viewport may resize after gallery
        # layout, so arrange on viewport resize as well (S1).
        try:
            vp = self._scroll.viewport()
            if vp is not None:
                vp.installEventFilter(self)
        except RuntimeError:
            # A deleted Qt viewport must not prevent gallery construction.
            pass

        # Initial layout arrangement
        self.refresh_layout()

        # Initial empty state for all present cards
        for card in self._cards.values():
            # Avoid double-setting for alias
            if card.current_state() != "empty":
                card.show_empty()

    def _build_viewport(self) -> None:
        """Build the scrollable card grid without installing resize reactions yet."""
        self._scroll = QScrollArea()
        self._scroll.setObjectName("previewGalleryScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._inner = QWidget()
        self._inner.setObjectName("previewGalleryInner")
        # Use grid for viewport-driven mosaic (narrow single column vs wide mosaic).
        self._inner_layout = QGridLayout(self._inner)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setSpacing(LAYOUT_SPACING)
        self._inner_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Track ordered cards for layout.
        self._ordered_cards: list[_PreviewCard] = []

        self._build_cards()

        self._scroll.setWidget(self._inner)

    def _build_cards(self) -> None:
        """Create capability-selected cards in display order, retaining the post alias."""
        # Cards — Run always, Analysis/Post per capability
        self._cards: dict[str, _PreviewCard] = {}

        run_card = _PreviewCard("run", "Run")
        self._cards["run"] = run_card
        self._ordered_cards.append(run_card)

        if self._has_analysis:
            ana_card = _PreviewCard("analysis", "Analysis")
            self._cards["analysis"] = ana_card
            self._ordered_cards.append(ana_card)

        if self._has_post:
            post_card = _PreviewCard("post", "Post-Analysis")
            # Normalize key: use "post_analysis" externally but "post" internally.
            self._cards["post_analysis"] = post_card
            # Also expose alias "post" for convenience
            self._cards["post"] = post_card
            self._ordered_cards.append(post_card)

    # -- viewport-driven responsive helpers -------------------------

    def _viewport_available_width(self) -> int:
        """Return available width for cards (gallery's own viewport)."""
        # Prefer scroll viewport when laid out; it reflects the inner preview
        # area's actual width excluding outer margins and scroll chrome.
        try:
            vp = self._scroll.viewport()
            if vp is not None and vp.width() > 0:
                return int(vp.width())
        except RuntimeError:
            # A deleted Qt viewport falls back to the gallery's own width.
            pass
        # Fallback to gallery width minus outer margins.
        w = int(self.width())
        if w > 0:
            return max(0, w - 2 * OUTER_MARGIN)
        return 0

    def _is_wide(self) -> bool:
        """Return True when viewport can fit two minimum-width cards (S1)."""
        avail = self._viewport_available_width()
        if avail <= 0:
            # Not yet laid out — treat as narrow until resize/show.
            return False
        return avail >= WIDE_THRESHOLD

    def is_wide_mode(self) -> bool:
        """Return the responsive mode, tolerating only Qt viewport lifetime errors.

        A RuntimeError from viewport access uses the gallery width instead;
        other viewport failures propagate to the caller.
        """
        return self._is_wide()

    def refresh_layout(self) -> None:
        """Reflow existing cards for the current viewport width and card count.

        Call on resize/show or after activating the containing Data page, when
        hidden-page resize events may have been deferred. Available images are
        rescaled after wide-mode reflow. An empty gallery is a no-op.
        """
        # Clear grid without deleting widgets.
        while self._inner_layout.count():
            item = self._inner_layout.takeAt(0)
            # widget stays owned; just removed from layout
            del item
        # Reset stretches
        for i in range(max(2, len(self._ordered_cards))):
            self._inner_layout.setColumnStretch(i, 0)
            self._inner_layout.setRowStretch(i, 0)

        is_wide = self._is_wide()
        n = len(self._ordered_cards)
        if n == 0:
            return
        if not is_wide:
            # Narrow: single column vertical.
            for r, card in enumerate(self._ordered_cards):
                self._inner_layout.addWidget(card, r, 0, 1, 1)
            self._inner_layout.setColumnStretch(0, 1)
            return
        # Wide mode — shape by card count.
        if n == 1:
            self._inner_layout.addWidget(self._ordered_cards[0], 0, 0, 1, 1)
            self._inner_layout.setColumnStretch(0, 1)
        elif n == 2:
            # Side-by-side
            self._inner_layout.addWidget(self._ordered_cards[0], 0, 0, 1, 1)
            self._inner_layout.addWidget(self._ordered_cards[1], 0, 1, 1, 1)
            self._inner_layout.setColumnStretch(0, 1)
            self._inner_layout.setColumnStretch(1, 1)
            self._inner_layout.setRowStretch(0, 1)
        else:  # n >=3, use first three for mosaic (Run left spanning two rows)
            # Run left spanning two rows
            self._inner_layout.addWidget(self._ordered_cards[0], 0, 0, 2, 1)
            self._inner_layout.addWidget(self._ordered_cards[1], 0, 1, 1, 1)
            if n >= 3:
                self._inner_layout.addWidget(self._ordered_cards[2], 1, 1, 1, 1)
            # Extra cards beyond 3 (should not happen per capability, but handle vertically)
            for idx in range(3, n):
                self._inner_layout.addWidget(
                    self._ordered_cards[idx], 2 + (idx - 3), 0, 1, 2
                )
            # Make left column larger (Run is "large" per spec).
            self._inner_layout.setColumnStretch(0, 2)
            self._inner_layout.setColumnStretch(1, 1)
            self._inner_layout.setRowStretch(0, 1)
            self._inner_layout.setRowStretch(1, 1)

        # Ensure each card rescales after reflow (viewport size changed).
        for card in self._ordered_cards:
            if (
                card.original_pixmap() is not None
                and card.current_state() == "available"
            ):
                card.rescale_image_to_viewport()

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self.refresh_layout()

    def showEvent(self, a0: QShowEvent | None) -> None:
        super().showEvent(a0)
        self.refresh_layout()

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a0 is None or a1 is None:
            return super().eventFilter(a0, a1)
        try:
            if a0 is self._scroll.viewport() and a1.type() == QEvent.Type.Resize:
                self.refresh_layout()
        # tolerated: Public reflow overrides can raise arbitrary errors; isolate Qt resize delivery.
        except Exception:  # noqa: BLE001
            pass
        return super().eventFilter(a0, a1)

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
        # Deduplicate alias — count ordered cards.
        return len(self._ordered_cards)

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
            png_bytes = self._renderer(fig)
        # tolerated: The injected renderer is an open callback; preserve per-card failure isolation.
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - exercised via injected failure
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
            card.show_image(pixmap)
        # tolerated: PNG decode/allocation and Qt presentation failures must stay local to this card.
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
            logger.warning("preview pixmap failed for %r: %s", key, exc, exc_info=True)
            card.show_unavailable(type(exc).__name__)

    # -- test helpers ------------------------------------------------

    def find_card(self, key: str) -> _PreviewCard | None:
        """Return the card widget for ``key`` (or None if unsupported)."""
        card = self._cards.get(key)
        if card is None and key in ("post", "post_analysis"):
            card = self._cards.get("post_analysis") or self._cards.get("post")
        return card

    def card_viewport_size(self, key: str) -> QSize | None:
        """Return image viewport size for card ``key`` (test helper)."""
        card = self.find_card(key)
        if card is None:
            return None
        return card.image_viewport_size()

    def displayed_pixmap_size(self, key: str) -> QSize | None:
        """Return displayed (scaled) pixmap size for card ``key``."""
        card = self.find_card(key)
        if card is None:
            return None
        pm = card.displayed_pixmap()
        if pm is None:
            return None
        return pm.size()

    def original_pixmap_size(self, key: str) -> QSize | None:
        """Return original pixmap size for card ``key``."""
        card = self.find_card(key)
        if card is None:
            return None
        pm = card.original_pixmap()
        if pm is None:
            return None
        return pm.size()
