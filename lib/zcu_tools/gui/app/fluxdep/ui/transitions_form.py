"""Transition-set presets + a small structured editor for the fit panel.

The notebook hand-writes a ``TransitionDict`` per fit (categories like
``transitions`` / ``mirror`` / ``red side`` / ``blue side``, each a list of
``(i, j)`` level pairs, plus optional ``r_f`` / ``sample_f`` scalars). The GUI
offers a preset dropdown that fills the per-category fields, which the user can
then fine-tune.

``TransitionsForm`` is a QWidget: a preset combo + one line edit per category
holding its ``(i, j)`` pairs as text (e.g. ``(0,1),(0,2)``). ``r_f`` / ``sample_f``
are NOT edited here (they are separate spin boxes on the fit panel) — this widget
owns only the transition lists. ``get_transitions`` parses the fields back to a
``TransitionDict``; a malformed field raises ``ValueError`` (fast fail).
"""

from __future__ import annotations

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFormLayout,
    QLineEdit,
    QWidget,
)

from zcu_tools.notebook.persistance import TransitionDict

# The editable categories (wire-name → label). r_f/sample_f are scalars handled
# elsewhere, so they are not in this list.
CATEGORIES: tuple[str, ...] = (
    "transitions",
    "mirror",
    "red side",
    "blue side",
)

# Preset name → {category: list[(i, j)]}. The notebook's common choices.
PRESETS: dict[str, dict[str, list[tuple[int, int]]]] = {
    "basic": {
        "transitions": [(0, 1), (0, 2), (1, 2), (1, 3)],
        "mirror": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)],
    },
    "integer": {
        "transitions": [(0, 1), (0, 2), (1, 2)],
        "mirror": [(0, 1), (0, 2)],
    },
    "all": {
        "transitions": [(i, j) for i in (0, 1) for j in range(5) if i < j],
        "mirror": [(i, j) for i in (0, 1) for j in range(5) if i < j],
        "red side": [(0, 1)],
        "blue side": [(0, 1)],
    },
}

_DEFAULT_PRESET = "basic"


def format_pairs(pairs: list[tuple[int, int]]) -> str:
    """Render ``[(0,1),(0,2)]`` as ``(0,1),(0,2)`` for a line edit."""
    return ",".join(f"({i},{j})" for i, j in pairs)


def parse_pairs(text: str) -> list[tuple[int, int]]:
    """Parse ``(0,1),(0,2)`` back to ``[(0,1),(0,2)]`` (fast-fails on garbage).

    Empty / whitespace text yields an empty list. Accepts pairs with or without
    surrounding parentheses and tolerates spaces; anything that is not an
    int-pair raises ``ValueError``.
    """
    text = text.strip()
    if not text:
        return []
    # Split on '),(' boundaries after normalising: strip outer parens per token.
    tokens = [t for t in text.replace(" ", "").split("),(") if t]
    pairs: list[tuple[int, int]] = []
    for tok in tokens:
        tok = tok.strip("()")
        parts = tok.split(",")
        if len(parts) != 2:
            raise ValueError(f"transition pair must be 'i,j', got {tok!r}")
        try:
            i, j = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(f"transition pair must be integers, got {tok!r}") from exc
        pairs.append((i, j))
    return pairs


class TransitionsForm(QWidget):
    """Per-category ``(i, j)`` line edits → ``TransitionDict`` (hand-entered).

    Transitions are independent of the bounds preset (which the fit panel owns):
    each category is a free-text list of ``(i, j)`` level pairs, seeded with the
    default basic set. There is no preset dropdown here — bounds presets live on
    the fit panel.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)

        self._fields: dict[str, QLineEdit] = {}
        for cat in CATEGORIES:
            edit = QLineEdit()
            self._fields[cat] = edit
            layout.addRow(cat, edit)

        # Seed with the default basic transition set so the form is not empty.
        default = PRESETS[_DEFAULT_PRESET]
        for cat, edit in self._fields.items():
            edit.setText(format_pairs(default.get(cat, [])))

    def set_transitions(self, transitions: TransitionDict) -> None:
        """Fill the fields from an existing ``TransitionDict`` (categories only)."""
        for cat, edit in self._fields.items():
            pairs = transitions.get(cat, [])
            edit.setText(format_pairs(list(pairs)))

    def get_transitions(self) -> TransitionDict:
        """Parse the fields back to a ``TransitionDict`` (empty categories dropped)."""
        result: dict[str, list[tuple[int, int]]] = {}
        for cat, edit in self._fields.items():
            pairs = parse_pairs(edit.text())
            if pairs:
                result[cat] = pairs
        return TransitionDict(result)  # type: ignore[arg-type]
