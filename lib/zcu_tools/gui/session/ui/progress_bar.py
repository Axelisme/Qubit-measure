"""Lightweight progress bar widget for high-frequency progress updates."""

from __future__ import annotations

from qtpy.QtCore import QRectF, QSize, Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QColor, QPainter, QPen  # type: ignore[attr-defined]
from qtpy.QtWidgets import QSizePolicy, QWidget  # type: ignore[attr-defined]


class LightweightProgressBar(QWidget):
    """Small progress bar with a QProgressBar-compatible subset.

    Native ``QProgressBar.setValue`` can synchronously spend hundreds of
    milliseconds in style/update paths during autofluxdep runs. This widget keeps
    the same narrow API the app uses, but ``setValue`` only stores state and
    schedules a repaint so Qt can coalesce high-frequency updates.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._minimum = 0
        self._maximum = 100
        self._value = 0
        self._format = "%v/%m"
        self._text_visible = True
        self.setMinimumHeight(18)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def minimum(self) -> int:
        return self._minimum

    def maximum(self) -> int:
        return self._maximum

    def value(self) -> int:
        return self._value

    def format(self) -> str:
        return self._format

    def setRange(self, minimum: int, maximum: int) -> None:
        minimum = int(minimum)
        maximum = int(maximum)
        if maximum < minimum:
            maximum = minimum
        if self._minimum == minimum and self._maximum == maximum:
            return
        self._minimum = minimum
        self._maximum = maximum
        self._value = self._clamp_value(self._value)
        self.update()

    def setMaximum(self, maximum: int) -> None:
        self.setRange(self._minimum, int(maximum))

    def setValue(self, value: int) -> None:
        value = self._clamp_value(int(value))
        if self._value == value:
            return
        self._value = value
        self.update()

    def setFormat(self, fmt: str) -> None:
        if self._format == fmt:
            return
        self._format = fmt
        self.update()

    def setTextVisible(self, visible: bool) -> None:
        visible = bool(visible)
        if self._text_visible == visible:
            return
        self._text_visible = visible
        self.update()

    def isTextVisible(self) -> bool:
        return self._text_visible

    def sizeHint(self) -> QSize:
        return QSize(240, 20)

    def _clamp_value(self, value: int) -> int:
        if self._maximum <= self._minimum:
            return self._minimum
        return max(self._minimum, min(self._maximum, value))

    def _fraction(self) -> float:
        span = self._maximum - self._minimum
        if span <= 0:
            return 0.0
        return (self._value - self._minimum) / span

    def _text(self) -> str:
        span = self._maximum - self._minimum
        percent = 0 if span <= 0 else round(self._fraction() * 100)
        return (
            self._format.replace("%v", str(self._value))
            .replace("%m", str(self._maximum))
            .replace("%p", str(percent))
        )

    def paintEvent(self, event) -> None:  # noqa: ANN001
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = 3.0
        painter.setPen(QPen(QColor(150, 156, 168), 1.0))
        painter.setBrush(QColor(242, 244, 247))
        painter.drawRoundedRect(rect, radius, radius)

        fraction = self._fraction()
        if fraction > 0.0:
            fill = rect.adjusted(1.0, 1.0, -1.0, -1.0)
            fill.setWidth(max(1.0, fill.width() * min(1.0, fraction)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(71, 128, 214))
            painter.drawRoundedRect(fill, radius - 1.0, radius - 1.0)

        if self._text_visible:
            text_rect = self.rect().adjusted(6, 0, -6, 0)
            text = self.fontMetrics().elidedText(
                self._text(),
                Qt.TextElideMode.ElideRight,
                max(0, text_rect.width()),
            )
            painter.setPen(QColor(28, 33, 42))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)
