"""Notify remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    default_number,
    required_integer,
    required_string,
)
from ._registry import RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "notify.open",
        "notify:h_notify_open",
        MethodSpec(
            30.0,
            "Open a non-modal agent-prompt dialog on the main thread. Returns {token}.",
            (
                required_string("message", "Message to display to the user"),
                default_number(
                    "timeout", 600.0, "Prompt auto-close timeout in seconds"
                ),
            ),
        ),
    ),
    method_entry(
        "notify.await",
        "notify:h_notify_await",
        MethodSpec(
            # Off-main handlers bypass the owner-thread watchdog. The handler
            # caps the caller's wait at 600 seconds; MCP adds transport slack
            # above this catalog deadline.
            615.0,
            "Block the IO worker until the notify prompt settles. Returns "
            "{reason, reply?}. reason in {'reply', 'dismiss', 'timeout'}.",
            (
                required_integer("token", "Token returned by notify.open"),
                default_number(
                    "timeout", 600.0, "Consumer backstop in seconds (0–600)"
                ),
            ),
            off_main_thread=True,
        ),
    ),
)
