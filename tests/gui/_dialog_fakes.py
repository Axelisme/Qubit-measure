"""DialogPresenter test adapters."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

from qtpy.QtWidgets import QWidget
from zcu_tools.gui.widgets.dialog_presenter import DestructiveConfirmation

DialogKind = Literal[
    "information",
    "warning",
    "critical",
    "confirm",
    "destructive_confirm",
    "save_discard_cancel",
]
# Mirrors the production SaveDiscardCancel answer type structurally.
SaveDiscardCancel = Literal["save", "discard", "cancel"]


@dataclass(frozen=True)
class DialogCall:
    kind: DialogKind
    title: str
    message: str
    action_text: str | None = None
    default: bool | None = None
    cancel_text: str | None = None


class RecordingDialogPresenter:
    """Recording DialogPresenter with scripted confirmation answers."""

    def __init__(
        self,
        *,
        confirm_answers: Iterable[bool] | None = None,
        destructive_answers: Iterable[bool] | None = None,
        save_discard_cancel_answers: Iterable[SaveDiscardCancel] | None = None,
        defer_confirm: bool = False,
    ) -> None:
        self.calls: list[DialogCall] = []
        self._confirm_answers = deque(confirm_answers or ())
        self._destructive_answers = deque(destructive_answers or ())
        self._save_discard_cancel_answers = deque(save_discard_cancel_answers or ())
        self._defer_confirm = defer_confirm
        self._pending_confirm: deque[Callable[[bool], None]] = deque()

    def queue_confirm(self, *answers: bool) -> None:
        self._confirm_answers.extend(answers)

    def queue_destructive_confirm(self, *answers: bool) -> None:
        self._destructive_answers.extend(answers)

    def queue_save_discard_cancel(self, *answers: SaveDiscardCancel) -> None:
        self._save_discard_cancel_answers.extend(answers)

    def information(self, parent: QWidget, title: str, message: str) -> None:
        del parent
        self.calls.append(DialogCall("information", title, message))

    def warning(self, parent: QWidget, title: str, message: str) -> None:
        del parent
        self.calls.append(DialogCall("warning", title, message))

    def critical(self, parent: QWidget, title: str, message: str) -> None:
        del parent
        self.calls.append(DialogCall("critical", title, message))

    def confirm(
        self,
        parent: QWidget,
        title: str,
        message: str,
        *,
        default: bool = False,
    ) -> bool:
        del parent
        self.calls.append(DialogCall("confirm", title, message, default=default))
        if self._confirm_answers:
            return self._confirm_answers.popleft()
        raise AssertionError(
            "Unexpected confirmation dialog without a queued answer: "
            f"title={title!r}, message={message!r}"
        )

    def confirm_async(
        self,
        parent: QWidget,
        title: str,
        message: str,
        *,
        on_decision: Callable[[bool], None],
        default: bool = False,
    ) -> None:
        if self._defer_confirm:
            self.calls.append(DialogCall("confirm", title, message, default=default))
            self._pending_confirm.append(on_decision)
            return
        on_decision(self.confirm(parent, title, message, default=default))

    def decide_confirm(self, accepted: bool) -> None:
        """Deliver one pending asynchronous confirmation decision."""
        if not self._pending_confirm:
            raise AssertionError("No pending asynchronous confirmation")
        self._pending_confirm.popleft()(accepted)

    def save_discard_cancel(
        self, parent: QWidget, title: str, message: str
    ) -> SaveDiscardCancel:
        del parent
        self.calls.append(DialogCall("save_discard_cancel", title, message))
        if self._save_discard_cancel_answers:
            return self._save_discard_cancel_answers.popleft()
        raise AssertionError(
            "Unexpected Save/Discard/Cancel dialog without a queued answer: "
            f"title={title!r}, message={message!r}"
        )

    def destructive_confirm(
        self,
        parent: QWidget,
        confirmation: DestructiveConfirmation,
        *,
        on_decision: Callable[[bool], None],
    ) -> None:
        """Record labels; only an explicit positive decision accepts the action."""
        del parent
        self.calls.append(
            DialogCall(
                "destructive_confirm",
                confirmation.title,
                confirmation.message,
                action_text=confirmation.action_text,
                default=confirmation.default,
                cancel_text=confirmation.cancel_text,
            )
        )
        if self._destructive_answers:
            on_decision(self._destructive_answers.popleft())
            return
        raise AssertionError(
            "Unexpected destructive confirmation dialog without a queued answer: "
            f"title={confirmation.title!r}, message={confirmation.message!r}, action_text={confirmation.action_text!r}"
        )

    def messages(self, kind: DialogKind | None = None) -> list[str]:
        if kind is None:
            return [call.message for call in self.calls]
        return [call.message for call in self.calls if call.kind == kind]

    def consume_message_containing(self, kind: DialogKind, text: str) -> str:
        for index, call in enumerate(self.calls):
            if call.kind == kind and text in call.message:
                self.calls.pop(index)
                return call.message
        raise AssertionError(
            f"Expected {kind} dialog containing {text!r}; got {self.calls!r}"
        )

    def assert_no_unexpected_messages(self) -> None:
        unexpected = [
            call
            for call in self.calls
            if call.kind in {"information", "warning", "critical"}
        ]
        if not unexpected:
            return
        formatted = "; ".join(
            f"{call.kind} {call.title!r}: {call.message!r}" for call in unexpected
        )
        raise AssertionError(f"Unexpected dialog message(s): {formatted}")
