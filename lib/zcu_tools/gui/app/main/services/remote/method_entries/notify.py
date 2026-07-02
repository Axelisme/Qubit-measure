"""Notify remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    _int,
    _num_default,
    _str,
)
from ._registry import RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "notify.open",
        "notify:_h_notify_open",
        MethodSpec(
            30.0,
            "Open a non-modal agent-prompt dialog on the main thread. Returns {token}.",
            (
                _str("message", "Message to display to the user"),
                _num_default("timeout", 600.0, "Prompt auto-close timeout in seconds"),
            ),
        ),
    ),
    method_entry(
        "notify.await",
        "notify:_h_notify_await",
        MethodSpec(
            # Nominal only — off_main_thread handlers bypass the main-thread budget
            # watchdog (control_service), so the real bound is the caller's `timeout`
            # param. Kept >= the default consumer backstop (600 + slack) for clarity.
            615.0,
            "Block the IO worker until the notify prompt settles. Returns "
            "{reason, reply?}. reason in {'reply', 'dismiss', 'timeout'}.",
            (
                _int("token", "Token returned by notify.open"),
                _num_default("timeout", 600.0, "Consumer backstop timeout in seconds"),
            ),
            off_main_thread=True,
        ),
    ),
)
