"""FrequencyField — a nullable GHz field with a needed-but-blank hint.

The Search form has two of them, r_f and sample_f. Each is optional (blank
means unset); some transition categories need it, and the form then marks the
field until it holds a value.
"""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtGui import QDoubleValidator, QPalette
from qtpy.QtWidgets import QLabel, QLineEdit, QVBoxLayout, QWidget

from zcu_tools.gui.widgets.control_indicator import CAUTION_COLOR


class FrequencyField(QWidget):
    """A line edit for an optional frequency, with its hint directly below.

    The line edit's objectName is object_name; the hint label's is
    ``f"{object_name}.required"`` and shows hint_text in the caution colour.
    A blank field shows a faded ``(unset)`` so it never reads as a value.
    """

    def __init__(
        self, object_name: str, hint_text: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._edit = QLineEdit()
        self._edit.setObjectName(object_name)
        self._edit.setPlaceholderText("(unset)")
        # Some styles draw placeholder text like entered text; fade it.
        palette = self._edit.palette()
        faded = palette.color(QPalette.ColorRole.Text)
        faded.setAlphaF(0.35)
        palette.setColor(QPalette.ColorRole.PlaceholderText, faded)
        self._edit.setPalette(palette)
        self._edit.setValidator(QDoubleValidator())
        self._hint = QLabel(hint_text)
        self._hint.setObjectName(f"{object_name}.required")
        self._hint.setStyleSheet(f"color: {CAUTION_COLOR};")
        self._hint.setWordWrap(True)
        self._hint.setHidden(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._edit)
        layout.addWidget(self._hint)

    def on_change(self, callback: Callable[[], None]) -> None:
        """Call callback after every text change, including set_value."""
        self._edit.textChanged.connect(lambda _text: callback())

    def value(self) -> float | None:
        """The entered frequency, or None when blank or not a number."""
        text = self._edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def set_value(self, value: float | None) -> None:
        """Show value in the field; None leaves it blank."""
        self._edit.setText("" if value is None else f"{value:g}")

    def mark_needed(self, *, needed: bool) -> bool:
        """Show the hint when needed and the field is blank; return whether it shows."""
        missing = needed and self.value() is None
        self._hint.setHidden(not missing)
        return missing
