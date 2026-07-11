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
        on_released: Callable[[], None] | None = None,
    ) -> QDialog:
        """Retain and open a dialog under a caller-owned key.

        Existing-key policy stays with the caller: a duplicate key is a bug, not
        an implicit raise/show decision.
        """
        if delete_on_close:
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.retain_named(
            key,
            dialog,
            on_finished=on_finished,
            on_released=on_released,
        )
        dialog.open()
        return dialog

    def retain_named(
        self,
        key: Hashable,
        dialog: QDialog,
        *,
        on_finished: Callable[[int], None] | None = None,
        on_released: Callable[[], None] | None = None,
    ) -> QDialog:
        """Retain an already-managed dialog without changing or opening it."""
        if key in self._refs:
            raise RuntimeError(f"dialog key is already retained: {key!r}")
        self._refs[key] = dialog

        def _cleanup() -> None:
            if self._refs.get(key) is not dialog:
                return
            self._refs.pop(key, None)
            if on_released is not None:
                on_released()

        def _on_finished(status: int) -> None:
            _cleanup()
            if on_finished is not None:
                on_finished(status)

        dialog.finished.connect(_on_finished)
        dialog.destroyed.connect(lambda *_args: _cleanup())
        return dialog

    def discard(self, key: Hashable) -> QDialog | None:
        """Stop retaining ``key`` without changing the dialog lifetime."""
        return self._refs.pop(key, None)

    def get(self, key: Hashable) -> QDialog | None:
        return self._refs.get(key)

    def __len__(self) -> int:
        return len(self._refs)
