"""Event-controlled public loop scheduling for cross-thread acknowledgement scenarios."""

import asyncio
from collections.abc import Callable
from contextvars import Context
from threading import Event, get_ident
from typing import Literal

import pytest


class LoopHold:
    """Deschedule one caller's first posted callback before or after delivery."""

    def __init__(
        self, *, phase: Literal["before", "after"], release_on_second: bool = False
    ) -> None:
        self.phase = phase
        self.release_on_second = release_on_second
        self.thread: int | None = None
        self.entered = Event()
        self.release = Event()
        self.posts = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = asyncio.BaseEventLoop.call_soon_threadsafe

        def schedule(
            loop: asyncio.BaseEventLoop,
            callback: Callable[..., object],
            *args: object,
            context: Context | None = None,
        ) -> asyncio.Handle:
            if get_ident() != self.thread:
                return original(loop, callback, *args, context=context)
            self.posts += 1
            if self.posts != 1:
                handle = original(loop, callback, *args, context=context)
                if self.release_on_second:
                    self.release.set()
                return handle

            def held() -> None:
                if self.phase == "after":
                    callback(*args)
                self.entered.set()
                assert self.release.wait(10), "Scenario teardown releases loop delivery"
                if self.phase == "before":
                    callback(*args)

            return original(loop, held, context=context)

        monkeypatch.setattr(asyncio.BaseEventLoop, "call_soon_threadsafe", schedule)
