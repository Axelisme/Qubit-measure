"""Shared custom Qt widget helpers."""

from __future__ import annotations

from math import isfinite
from sys import float_info

from PyQt6.QtCore import pyqtSignal
from qtpy.QtGui import QValidator
from qtpy.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QWidget


class FiniteFloatSpinBox(QAbstractSpinBox):
    """Native spinbox preserving every finite Python float without quantization.

    Decimal/scientific text uses Python float parsing and repr round-tripping,
    including subnormals. None belongs to the caller's nullable control, not this
    editor. setValue rejects nonfinite values with ValueError. Invalid text is
    rejected; incomplete text stays uncommitted while value() retains the last
    accepted float. Leaving incomplete text restores the last accepted value.
    clear() empties the display without changing that accepted value, so callers
    can present unknown fields through their separate nullable control.
    Arrows change the value by 1 per step. Overflow saturates at the largest
    finite magnitude; rounding below a float's spacing may leave it unchanged.
    valueChanged emits the new finite float only when its numeric value changes.
    """

    valueChanged = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create a zero-valued editor with optional Qt parent ownership."""
        super().__init__(parent)
        self._value = 0.0
        editor = self.lineEdit()
        if editor is None:
            raise RuntimeError("Qt spinbox did not create its line editor")
        self._editor = editor
        self._editor.setText(repr(self._value))
        self._editor.textEdited.connect(self._on_text_edited)

    def value(self) -> float:
        """Return the last accepted finite float, independent of uncommitted text."""
        return self._value

    def setValue(self, value: float) -> None:
        """Display a finite float exactly; ValueError rejects NaN/infinity."""
        if not isfinite(value):
            raise ValueError("Value must be finite")
        changed = value != self._value
        self._value = value
        self._editor.setText(repr(value))
        if changed:
            self.valueChanged.emit(value)

    def validate(
        self, input: str | None, pos: int
    ) -> tuple[QValidator.State, str, int]:
        """Return (state, input text, cursor pos) for Qt's numeric validation.

        Finite numeric text is Acceptable, numeric prefixes are Intermediate,
        and nonnumeric/nonfinite text is Invalid. None is treated as empty text.
        """
        text = input or ""
        try:
            value = float(text)
        except ValueError:
            # Only numeric prefixes are intermediate, not arbitrary invalid text.
            state = (
                QValidator.State.Intermediate
                if all(char in "0123456789.eE+-" for char in text)
                else QValidator.State.Invalid
            )
        else:
            state = (
                QValidator.State.Acceptable
                if isfinite(value)
                else QValidator.State.Invalid
            )
        return state, text, pos

    def fixup(self, input: str | None) -> str:
        """Keep valid text; restore the last accepted float for incomplete input."""
        text = input or ""
        if self.validate(text, len(text))[0] == QValidator.State.Acceptable:
            return text
        return repr(self._value)

    def stepBy(self, steps: int) -> None:
        """Add steps in units of 1; overflow saturates at ±sys.float_info.max."""
        try:
            candidate = self._value + steps
        except OverflowError:
            candidate = float_info.max if steps > 0 else -float_info.max
        if not isfinite(candidate):
            candidate = float_info.max if steps > 0 else -float_info.max
        self.setValue(candidate)

    def stepEnabled(self) -> QAbstractSpinBox.StepEnabledFlag:
        """Enable both native arrows; stepping at finite extremes is bounded."""
        return self.StepEnabledFlag.StepUpEnabled | self.StepEnabledFlag.StepDownEnabled

    def _on_text_edited(self, text: str) -> None:
        state, _, _ = self.validate(text, len(text))
        if state == QValidator.State.Acceptable:
            value = float(text)
            if value != self._value:
                self._value = value
                self.valueChanged.emit(value)


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
