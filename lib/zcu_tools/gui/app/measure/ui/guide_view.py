"""Read-only Guide subtab view that renders an adapter's Markdown guide."""

from __future__ import annotations

import io
import re
import string
from pathlib import Path, PurePosixPath
from typing import Any

from matplotlib import mathtext
from matplotlib.font_manager import FontProperties
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
_MARKDOWN_FEATURES = (
    QTextDocument.MarkdownFeature.MarkdownDialectGitHub
    | QTextDocument.MarkdownFeature.MarkdownNoHTML
)
# Rendered formulas are served from memory under this URL scheme.
_MATH_SCHEME = "mathtext"
_LOCAL_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})

_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_DISPLAY_MATH = re.compile(r"^(\s*)\$\$([^$]*\S[^$]*)\$\$\s*$")
# Leftmost-first alternation: code spans and backslash escapes are consumed
# before a dollar can open math, so `$x$` in code and \$ stay literal. An
# opening $ is followed by a non-space, a closing $ follows a non-space that is
# not a backslash and is not followed by a digit.
_INLINE_TOKEN = re.compile(
    r"(?P<code>(?P<ticks>`+)(?!`).*?(?<!`)(?P=ticks)(?!`))"
    r"|(?P<escape>\\.)"
    r"|(?P<double>\$\$)"
    r"|\$(?P<math>[^\s$\\]|[^\s$][^$]*[^\s$\\])\$(?!\d)"
)


class GuideView(QTextBrowser):
    """Read-only Markdown view of one adapter's ``AdapterGuide``.

    The view renders ``guide.markdown`` once at construction with the GitHub
    Markdown dialect; raw HTML tags show as literal text. A clicked link goes
    to ``QDesktopServices.openUrl`` unchanged and never navigates this view.

    TeX between ``$...$`` inside a line, and a line holding only ``$$...$$``,
    renders through matplotlib mathtext as an image in the view's font size and
    text color. An opening ``$`` must be followed by a non-space; a closing
    ``$`` must follow a non-space and must not be followed by a digit. ``\\$``
    is a literal dollar. Expressions mathtext rejects, ``$$`` inside a line,
    and dollars in code spans or fenced code blocks stay as written. Indented
    code blocks are not recognised, because list continuations use the same
    indentation.

    A relative image path that names a PNG or JPG file directly inside
    ``guide.resource_dir`` is shown, scaled down to the view width when wider
    and kept at its own size otherwise. Every other image stays unloaded and
    the rest of the guide still renders: a missing file, another directory,
    SVG or any other suffix, a URL with a scheme, or any image when
    ``resource_dir`` is None. Guide content never raises; Qt and matplotlib
    errors other than a mathtext rejection propagate from construction.
    """

    def __init__(self, guide: AdapterGuide, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._resource_dir = guide.resource_dir
        self._math_images: dict[str, QImage] = {}
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._open_link)
        document = self._document()
        document.setUndoRedoEnabled(False)
        document.setMarkdown(self._replace_math(guide.markdown), _MARKDOWN_FEATURES)
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

    def _replace_math(self, markdown: str) -> str:
        """Return ``markdown`` with renderable formulas replaced by image links."""
        lines: list[str] = []
        fence: str | None = None
        for line in markdown.split("\n"):
            if fence is not None:
                if _closes_fence(line, fence):
                    fence = None
                lines.append(line)
                continue
            opened = _FENCE_OPEN.match(line)
            if opened is not None:
                fence = opened.group(1)
                lines.append(line)
                continue
            display = _DISPLAY_MATH.match(line)
            if display is not None:
                indent, expression = display.groups()
                lines.append(indent + self._math_or_literal(expression, line.strip()))
                continue
            lines.append(_INLINE_TOKEN.sub(self._inline_token, line))
        return "\n".join(lines)

    def _inline_token(self, match: re.Match[str]) -> str:
        expression = match.group("math")
        if expression is None:
            return match.group(0)
        return self._math_or_literal(expression, match.group(0))

    def _math_or_literal(self, expression: str, source: str) -> str:
        image = self._render_math(expression)
        if image is None:
            return _literal_markdown(source)
        key = str(len(self._math_images))
        self._math_images[key] = image
        # Qt's Markdown importer drops images with empty alt text.
        return f"![math]({_MATH_SCHEME}:{key})"

    def _render_math(self, expression: str) -> QImage | None:
        ratio = self.devicePixelRatioF()
        buffer = io.BytesIO()
        try:
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


def _closes_fence(line: str, fence: str) -> bool:
    """Return whether ``line`` closes a code fence opened by ``fence``."""
    closing = re.match(rf"^ {{0,3}}({re.escape(fence[0])}+)\s*$", line)
    return closing is not None and len(closing.group(1)) >= len(fence)


def _literal_markdown(text: str) -> str:
    """Escape every ASCII punctuation character so Markdown shows ``text`` as is."""
    return "".join(f"\\{char}" if char in string.punctuation else char for char in text)
