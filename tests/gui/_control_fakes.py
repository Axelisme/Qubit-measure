"""Small recording helpers for control-facet contract tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RecordedCall:
    target: str
    method: str
    args: tuple[object, ...] = ()
    kwargs: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class SameObject:
    value: object

    def __eq__(self, other: object) -> bool:
        return other is self.value


def same(value: object) -> SameObject:
    return SameObject(value)


def call(
    target: str,
    method: str,
    *args: object,
    **kwargs: object,
) -> RecordedCall:
    return RecordedCall(
        target=target,
        method=method,
        args=args,
        kwargs=tuple(kwargs.items()),
    )


@dataclass
class CallLog:
    calls: list[RecordedCall] = field(default_factory=list)

    def add(
        self,
        target: str,
        method: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.calls.append(call(target, method, *args, **kwargs))


class RecordingSignal:
    def __init__(
        self,
        log: CallLog,
        name: str,
        *,
        disconnect_error: Exception | None = None,
    ) -> None:
        self._log = log
        self._name = name
        self._disconnect_error = disconnect_error
        self.handlers: list[Callable[..., None]] = []

    def connect(self, handler: Callable[..., None]) -> None:
        self._log.add(self._name, "connect", handler)
        self.handlers.append(handler)

    def disconnect(self) -> None:
        self._log.add(self._name, "disconnect")
        self.handlers.clear()
        if self._disconnect_error is not None:
            raise self._disconnect_error
