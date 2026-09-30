"""Notify remote handlers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


logger = logging.getLogger(__name__)


def h_notify_open(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    message = str(params["message"])
    timeout = float(params["timeout"])  # type: ignore[arg-type]
    token = adapter.ctrl.open_notify_prompt(message, timeout)
    return {"token": token}


def h_notify_await(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    # off_main_thread handler: blocks the IO worker on the thread-safe
    # NotifyChannel.consume(). Never touches main-thread-owned state.
    token = int(params["token"])  # type: ignore[arg-type]
    timeout = float(params["timeout"])  # type: ignore[arg-type]
    if not 0 <= timeout <= 600:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "timeout must be between 0 and 600 seconds",
            reason="invalid_timeout",
        )
    result = adapter.ctrl.await_notify(token, timeout)
    wire: dict[str, object] = {"reason": result.reason}
    if result.reply is not None:
        wire["reply"] = result.reply
    return wire
