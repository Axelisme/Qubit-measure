"""Notify remote handlers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter

from ._common import Handler

logger = logging.getLogger(__name__)


def _h_notify_open(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    message = str(params["message"])
    timeout = float(params["timeout"])  # type: ignore[arg-type]
    token = adapter.ctrl.open_notify_prompt(message, timeout)
    return {"token": token}


def _h_notify_await(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # off_main_thread handler: blocks the IO worker on the thread-safe
    # NotifyChannel.consume(). Never touches main-thread-owned state.
    token = int(params["token"])  # type: ignore[arg-type]
    timeout = float(params["timeout"])  # type: ignore[arg-type]
    result = adapter.ctrl.await_notify(token, timeout)
    wire: dict[str, object] = {"reason": result.reason}
    if result.reply is not None:
        wire["reply"] = result.reply
    return wire


HANDLERS: dict[str, Handler] = {
    "notify.open": _h_notify_open,
    "notify.await": _h_notify_await,
}
