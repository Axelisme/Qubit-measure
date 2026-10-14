"""Read-only Guide subtab view that renders an adapter's Markdown guide."""

from __future__ import annotations

import html
import io
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.renderer import RendererProtocol
from markdown_it.token import Token
from markdown_it.utils import EnvType, OptionsDict
from matplotlib import mathtext, rc_context
from matplotlib.font_manager import FontProperties
from mdit_py_plugins.dollarmath import dollarmath_plugin
from qtpy.QtCore import QUrl
from qtpy.QtGui import (
    QDesktopServices,
    QFontInfo,
    QImage,
    QPalette,
    QResizeEvent,
    QTextCursor,
    QTextDocument,
    QTextImageFormat,
)
from qtpy.QtWidgets import QTextBrowser, QWidget

from zcu_tools.gui.app.measure.adapter import AdapterGuide

_IMAGE_RESOURCE = QTextDocument.ResourceType.ImageResource.value
# Rendered formulas are served from memory under this URL scheme.
_MATH_SCHEME = "mathtext"
_LOCAL_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
# Qt's HTML importer draws tables without borders; this only styles the HTML
# that markdown-it-py produced.
_STYLESHEET = """
table { border-collapse: collapse; border-width: 1px; border-style: solid;
        border-color: #888888; margin: 6px 0; }
th, td { border-width: 1px; border-style: solid; border-color: #888888;
         padding: 4px 8px; }
pre { background-color: rgba(127, 127, 127, 40); padding: 6px; }
blockquote { margin-left: 16px; }
"""


class GuideView(QTextBrowser):
    """Read-only Markdown view of one adapter's ``AdapterGuide``.

    The view renders ``guide.markdown`` once at construction with markdown-it-py
    (CommonMark plus tables, raw HTML shown as literal text) and shows the
    resulting HTML. Code spans, fenced and indented code blocks, link
    destinations and autolinks keep their text as written. A clicked link goes
    to ``QDesktopServices.openUrl`` unchanged and never navigates this view; a
    bare URL is plain text, not a link.

    TeX between ``$...$`` inside a line, and a ``$$...$$`` block on its own
    lines, renders through matplotlib mathtext as an image in the view's font
    size and text color. An opening ``$`` must be followed by a non-space and
    must not follow a digit; a closing ``$`` must follow a non-space and must
    not be followed by a digit. ``\\$`` is a literal dollar. Expressions
    mathtext rejects and ``$$...$$`` inside a line stay as written.

    A relative image path that names a PNG or JPG file directly inside
    ``guide.resource_dir`` is shown, with or without alt text, scaled down to
    the view width when wider and kept at its own size otherwise. Every other
    image stays unloaded and the rest of the guide still renders: a missing
    file, another directory, SVG or any other suffix, a URL with a scheme, or
    any image when ``resource_dir`` is None. Guide content never raises; Qt and
    matplotlib errors other than a mathtext rejection propagate from
    construction.
    """

    def __init__(self, guide: AdapterGuide, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._resource_dir = guide.resource_dir
        self._math_images: dict[str, QImage] = {}
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._open_link)
        document = self._document()
        document.setUndoRedoEnabled(False)
        document.setDefaultStyleSheet(_STYLESHEET)
        parser = MarkdownIt("commonmark", {"html": False}).enable("table")
        parser.use(
            dollarmath_plugin,
            allow_space=False,
            allow_digits=False,
            double_inline=True,
            renderer=self._math_html,
        )
        parser.add_render_rule("math_inline_double", _inline_double_source)
        self.setHtml(parser.render(guide.markdown))
        self._fit_images()

    def loadResource(self, type: int, name: QUrl) -> Any:
        """Serve rendered formulas and allowed local images; refuse the rest."""
        if type != _IMAGE_RESOURCE:
            return None
        if name.scheme() == _MATH_SCHEME:
            return self._math_images.get(name.path())
        path = self._local_image_path(name)
        if path is None:
            return None
        image = QImage(str(path))
        return None if image.isNull() else image

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self._fit_images()

    def _document(self) -> QTextDocument:
        document = self.document()
        if document is None:
            raise RuntimeError("QTextBrowser has no document")
        return document

    def _open_link(self, url: QUrl) -> None:
        QDesktopServices.openUrl(url)

    def _local_image_path(self, name: QUrl) -> Path | None:
        if self._resource_dir is None or name.scheme():
            return None
        relative = PurePosixPath(name.path())
        if relative.parent != PurePosixPath("."):
            return None
        if relative.suffix.lower() not in _LOCAL_IMAGE_SUFFIXES:
            return None
        path = self._resource_dir / relative.name
        return path if path.is_file() else None

    def _math_html(self, content: str, options: dict[str, Any]) -> str:
        """Render one dollarmath token as an image tag, or its source text."""
        display = bool(options["display_mode"])
        image = self._render_math(content)
        if image is None:
            delimiter = "$$" if display else "$"
            return html.escape(f"{delimiter}{content}{delimiter}")
        key = str(len(self._math_images))
        self._math_images[key] = image
        align = "" if display else ' style="vertical-align: middle"'
        return f'<img src="{_MATH_SCHEME}:{key}" alt="math"{align}>'

    def _render_math(self, expression: str) -> QImage | None:
        ratio = self.devicePixelRatioF()
        buffer = io.BytesIO()
        try:
            # math_to_image saves an opaque white figure by default, which
            # shows as a box on non-white palettes.
            with rc_context({"savefig.transparent": True}):
                mathtext.math_to_image(
                    f"${expression}$",
                    buffer,
                    prop=FontProperties(size=QFontInfo(self.font()).pointSizeF()),
                    dpi=self.logicalDpiY() * ratio,
                    format="png",
                    color=self.palette().color(QPalette.ColorRole.Text).name(),
                )
        except ValueError:
            # mathtext rejects unsupported or malformed TeX with ValueError;
            # the caller shows the source text instead.
            return None
        image = QImage.fromData(buffer.getvalue(), "PNG")
        image.setDevicePixelRatio(ratio)
        return image

    def _fit_images(self) -> None:
        document = self._document()
        viewport = self.viewport()
        if viewport is None:
            return
        available = viewport.width() - 2 * document.documentMargin()
        if available <= 0:
            return
        placements: list[tuple[int, int, QTextImageFormat]] = []
        block = document.begin()
        while block.isValid():
            fragments = block.begin()
            while not fragments.atEnd():
                fragment = fragments.fragment()
                char_format = fragment.charFormat()
                if char_format.isImageFormat():
                    placements.append(
                        (
                            fragment.position(),
                            fragment.length(),
                            char_format.toImageFormat(),
                        )
                    )
                fragments += 1
            block = block.next()
        cursor = QTextCursor(document)
        for position, length, image_format in placements:
            image = document.resource(_IMAGE_RESOURCE, QUrl(image_format.name()))
            if not isinstance(image, QImage) or image.isNull():
                continue
            natural = image.deviceIndependentSize()
            width = min(natural.width(), available)
            height = natural.height() * width / natural.width()
            if image_format.width() == width and image_format.height() == height:
                continue
            image_format.setWidth(width)
            image_format.setHeight(height)
            cursor.setPosition(position)
            cursor.setPosition(position + length, QTextCursor.MoveMode.KeepAnchor)
            cursor.setCharFormat(image_format)


def _inline_double_source(
    _renderer: RendererProtocol,
    tokens: Sequence[Token],
    idx: int,
    _options: OptionsDict,
    _env: EnvType,
) -> str:
    """markdown-it render rule that shows ``$$...$$`` inside a line as written.

    dollarmath tokenizes inline ``$$`` only so its dollars cannot pair as
    ``$...$`` math; display math is a ``$$`` block on its own lines.
    """
    token = tokens[idx]
    return html.escape(f"{token.markup}{token.content}{token.markup}")
