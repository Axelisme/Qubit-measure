"""Lifecycle helpers for non-modal Qt dialogs opened with ``open()``."""

from __future__ import annotations

from collections.abc import Callable, Hashable

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import QDialog  # type: ignore[attr-defined]


class DialogRefStore:
    """Retain non-modal dialogs until Qt signals that their lifetime ended."""

    def __init__(self) -> None:
        self._refs: dict[Hashable, QDialog] = {}
        self._next_transient_key = 0

    def open_transient(
        self,
        dialog: QDialog,
        *,
        delete_on_close: bool = True,
        on_finished: Callable[[int], None] | None = None,
    ) -> QDialog:
        """Retain and open a dialog under a private one-shot key."""
        key = ("transient", self._next_transient_key)
        self._next_transient_key += 1
        return self.open_named(
            key,
            dialog,
            delete_on_close=delete_on_close,
            on_finished=on_finished,
        )

    def open_named(
        self,
        key: Hashable,
        dialog: QDialog,
        *,
        delete_on_close: bool = True,
        on_finished: Callable[[int], None] | None = None,
    ) -> QDialog:
        """Retain and open a dialog under a caller-owned key.

        Existing-key policy stays with the caller: a duplicate key is a bug, not
        an implicit raise/show decision.
        """
        if key in self._refs:
            raise RuntimeError(f"dialog key is already retained: {key!r}")
        self._refs[key] = dialog
        if delete_on_close:
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        def _cleanup() -> None:
            self._refs.pop(key, None)

        def _on_finished(status: int) -> None:
            _cleanup()
            if on_finished is not None:
                on_finished(status)

        dialog.finished.connect(_on_finished)
        dialog.destroyed.connect(lambda *_args: _cleanup())
        dialog.open()
        return dialog

    def get(self, key: Hashable) -> QDialog | None:
        return self._refs.get(key)

    def __len__(self) -> int:
        return len(self._refs)
