"""Width-bound captions for Measure's Data figure views."""

from qtpy.QtCore import QEvent
from qtpy.QtGui import QResizeEvent, QTextLayout, QTextOption
from qtpy.QtWidgets import QLabel, QSizePolicy


class FigureCaption(QLabel):
    """Fit a figure heading to its assigned width, including unbroken identifiers."""

    def __init__(self, caption: str = "") -> None:
        """Create a label for `caption`, with line breaks added only as needed."""
        super().__init__(caption)
        self._caption = caption
        # The pane sets the width; even an unbroken heading cannot widen it.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def set_caption(self, caption: str) -> None:
        """Replace the label text and fit it to the current width."""
        self._caption = caption
        self._wrap_caption()

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self._wrap_caption()

    def changeEvent(self, a0: QEvent | None) -> None:
        super().changeEvent(a0)
        if a0 is not None and a0.type() == QEvent.Type.FontChange:
            self._wrap_caption()

    def _wrap_caption(self) -> None:
        width = self.contentsRect().width()
        caption = self._caption
        if 0 < width < self.fontMetrics().horizontalAdvance(caption):
            breakable = caption.replace("_", "_\u200b")
            layout = QTextLayout(breakable, self.font())
            option = QTextOption()
            option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            layout.setTextOption(option)
            # Qt line offsets count UTF-16 code units, not Python characters.
            encoded = breakable.encode("utf-16-le")
            parts: list[str] = []
            layout.beginLayout()
            while (line := layout.createLine()).isValid():
                line.setLineWidth(width)
                start = 2 * line.textStart()
                end = start + 2 * line.textLength()
                parts.append(encoded[start:end].decode("utf-16-le"))
            layout.endLayout()
            caption = "\n".join(parts).replace("\u200b", "")
        if self.text() != caption:
            self.setText(caption)
