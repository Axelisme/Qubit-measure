"""Shared custom Qt widget helpers."""

from __future__ import annotations

from qtpy.QtWidgets import QDoubleSpinBox  # type: ignore[attr-defined]


class TrimDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that strips trailing zeros, keeping at least one decimal place.

    Example: decimals=6, value=1.5 → displays "1.5" instead of "1.500000".
    """

    def textFromValue(self, v: float) -> str:
        text = f"{v:.{self.decimals()}f}"
        # strip trailing zeros after decimal point, but keep at least one digit
        if "." in text:
            text = text.rstrip("0")
            if text.endswith("."):
                text += "0"
        return text
