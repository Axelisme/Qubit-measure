"""DialogPresenter test adapters."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from qtpy.QtWidgets import QWidget  # type: ignore[attr-defined]

DialogKind = Literal[
    "information", "warning", "critical", "confirm", "destructive_confirm"
]


@dataclass(frozen=True)
class DialogCall:
    kind: DialogKind
    title: str
    message: str
    action_text: str | None = None
    default: bool | None = None


class RecordingDialogPresenter:
    """Recording DialogPresenter with scripted confirmation answers."""

    def __init__(
        self,
        *,
        confirm_answers: Iterable[bool] | None = None,
        destructive_answers: Iterable[bool] | None = None,
    ) -> None:
        self.calls: list[DialogCall] = []
        self._confirm_answers = deque(confirm_answers or ())
        self._destructive_answers = deque(destructive_answers or ())

    def queue_confirm(self, *answers: bool) -> None:
        self._confirm_answers.extend(answers)

    def queue_destructive_confirm(self, *answers: bool) -> None:
        self._destructive_answers.extend(answers)

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

    def destructive_confirm(
        self,
        parent: QWidget,
        title: str,
        message: str,
        *,
        action_text: str,
        default: bool = False,
    ) -> bool:
        del parent
        self.calls.append(
            DialogCall(
                "destructive_confirm",
                title,
                message,
                action_text=action_text,
                default=default,
            )
        )
        if self._destructive_answers:
            return self._destructive_answers.popleft()
        raise AssertionError(
            "Unexpected destructive confirmation dialog without a queued answer: "
            f"title={title!r}, message={message!r}, action_text={action_text!r}"
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
