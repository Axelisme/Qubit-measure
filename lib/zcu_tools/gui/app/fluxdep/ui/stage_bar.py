"""StageBar — a horizontal, numbered row of pipeline stages: ``1 A → 2 B → …``.

The bar only draws what its host tells it and reports clicks. The host derives
each stage's status from published State and live interactive facts, decides
what a click does (holder gate, discard confirmation, view only), and marks
the stage whose content it shows as current. ``stage_status`` gives every
host the same priority when facts overlap. The Spectra page uses it for
``Align → Sample`` and the Fit page for ``Filter → Search → Show``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

StageStatus = Literal["open", "done", "draft", "locked"]
"""What a stage shows.

open: can be entered and has no published result yet. done: its result is
published (shown with ✓). draft: an unfinished draft of it is live (shown
with "draft"). locked: a previous stage is not done yet, so the button is
disabled and cannot be clicked.
"""

_SUFFIX: dict[StageStatus, str] = {
    "open": "",
    "done": " ✓",
    "draft": " · draft",
    "locked": "",
}

_TOOLTIP: dict[StageStatus, str] = {
    "open": "Not done yet",
    "done": "Done; click to view or redo",
    "draft": "Unfinished draft",
    "locked": "Finish the previous stage first",
}


def stage_status(*, done: bool, draft: bool, locked: bool) -> StageStatus:
    """Pick one status from a stage's facts: draft, then locked, then done.

    done means the stage's result is published; draft means an unfinished
    draft of it is live; locked means a previous stage is not done. A live
    draft wins over a lock, and a lock hides a published result. Returns
    "open" when none of them holds.
    """
    if draft:
        return "draft"
    if locked:
        return "locked"
    return "done" if done else "open"


class StageBar[K: str](QWidget):
    """A numbered stage row whose buttons report clicks to the host.

    Each stage is a QPushButton with objectName ``<object_prefix>.<key>`` and
    text ``<n> <label>`` plus the status mark. The button carries the dynamic
    Qt property ``stage_status`` (a StageStatus value) and is checked while it
    is the current stage. Every stage starts as "open" and no stage is current.
    """

    def __init__(
        self,
        object_prefix: str,
        stages: Sequence[tuple[K, str]],
        *,
        on_click: Callable[[K], None],
        parent: QWidget | None = None,
    ) -> None:
        """Build one button per (key, label) in stages, left to right.

        object_prefix names the buttons, e.g. "fluxdep.stage" gives
        "fluxdep.stage.align". on_click(key) runs on every click of an enabled
        stage; the bar itself changes no status or current mark on a click.
        Raises ValueError for no stages or a repeated key.
        """
        super().__init__(parent)
        keys = [key for key, _label in stages]
        if not keys:
            raise ValueError("a stage bar needs at least one stage")
        if len(set(keys)) != len(keys):
            raise ValueError(f"stage keys must be unique, got {keys}")
        self.setObjectName(object_prefix)
        self._on_click = on_click
        self._labels: dict[K, str] = {}
        self._buttons: dict[K, QPushButton] = {}
        self._current: K | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for number, (key, label) in enumerate(stages, start=1):
            if number > 1:
                layout.addWidget(QLabel("→"))
            button = QPushButton()
            button.setObjectName(f"{object_prefix}.{key}")
            button.setCheckable(True)
            # key=key binds this stage's key; clicked passes the checked flag.
            button.clicked.connect(lambda _checked=False, key=key: self._clicked(key))
            layout.addWidget(button)
            self._labels[key] = f"{number} {label}"
            self._buttons[key] = button
            self._show(key, "open")
        layout.addStretch(1)

    def set_status(self, key: K, status: StageStatus) -> None:
        """Show status on stage key; "locked" disables its button.

        Raises KeyError for a key the bar was not built with.
        """
        self._show(key, status)

    def set_current(self, key: K | None) -> None:
        """Mark key as the stage whose content the host shows; None marks none.

        Raises KeyError for a key the bar was not built with.
        """
        if key is not None and key not in self._buttons:
            raise KeyError(key)
        self._current = key
        for stage, button in self._buttons.items():
            button.setChecked(stage == key)

    def _show(self, key: K, status: StageStatus) -> None:
        button = self._buttons[key]
        button.setText(self._labels[key] + _SUFFIX[status])
        button.setToolTip(_TOOLTIP[status])
        button.setProperty("stage_status", status)
        button.setEnabled(status != "locked")

    def _clicked(self, key: K) -> None:
        # Qt toggles a checkable button before clicked; only the host moves current.
        self._buttons[key].setChecked(key == self._current)
        self._on_click(key)
