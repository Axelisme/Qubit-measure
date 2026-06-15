"""Tests for notify.open / notify.await dispatch handlers (Stage 4b).

Covers:
  - _h_notify_open: calls ctrl.open_notify_prompt and returns {token}
  - _h_notify_await: calls ctrl.await_notify and folds result → {reason, reply?}
  - METHOD_REGISTRY["notify.open"].off_main_thread is False (main-thread handler)
  - METHOD_REGISTRY["notify.await"].off_main_thread is True (IO-worker handler)
  - reply-absent when reason != 'reply'
"""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.app.main.services.remote.dispatch import (
    METHOD_REGISTRY,
    _h_notify_await,
    _h_notify_open,
)
from zcu_tools.gui.session.notify_handles import NotifyResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(
    open_return: int = 42, await_return: NotifyResult | None = None
) -> MagicMock:
    adapter = MagicMock()
    adapter.ctrl.open_notify_prompt.return_value = open_return
    if await_return is not None:
        adapter.ctrl.await_notify.return_value = await_return
    return adapter


# ---------------------------------------------------------------------------
# _h_notify_open
# ---------------------------------------------------------------------------


def test_h_notify_open_calls_ctrl_and_returns_token() -> None:
    adapter = _make_adapter(open_return=7)
    result = _h_notify_open(adapter, {"message": "hello", "timeout": 30.0})
    adapter.ctrl.open_notify_prompt.assert_called_once_with("hello", 30.0)
    assert result == {"token": 7}


def test_h_notify_open_coerces_message_to_str() -> None:
    adapter = _make_adapter(open_return=1)
    # message given as a non-str (wire may deliver any type)
    _h_notify_open(adapter, {"message": 123, "timeout": 10.0})
    call_args = adapter.ctrl.open_notify_prompt.call_args
    assert isinstance(call_args.args[0], str)
    assert call_args.args[0] == "123"


def test_h_notify_open_coerces_timeout_to_float() -> None:
    adapter = _make_adapter(open_return=1)
    _h_notify_open(adapter, {"message": "hi", "timeout": "600"})
    call_args = adapter.ctrl.open_notify_prompt.call_args
    assert isinstance(call_args.args[1], float)


# ---------------------------------------------------------------------------
# _h_notify_await
# ---------------------------------------------------------------------------


def test_h_notify_await_reply_includes_reply_key() -> None:
    adapter = _make_adapter(await_return=NotifyResult("reply", "yes"))
    result = _h_notify_await(adapter, {"token": 7, "timeout": 600.0})
    adapter.ctrl.await_notify.assert_called_once_with(7, 600.0)
    assert result["reason"] == "reply"
    assert result["reply"] == "yes"


def test_h_notify_await_dismiss_omits_reply_key() -> None:
    adapter = _make_adapter(await_return=NotifyResult("dismiss"))
    result = _h_notify_await(adapter, {"token": 3, "timeout": 600.0})
    assert result["reason"] == "dismiss"
    assert "reply" not in result


def test_h_notify_await_timeout_omits_reply_key() -> None:
    adapter = _make_adapter(await_return=NotifyResult("timeout"))
    result = _h_notify_await(adapter, {"token": 5, "timeout": 600.0})
    assert result["reason"] == "timeout"
    assert "reply" not in result


def test_h_notify_await_coerces_token_to_int() -> None:
    adapter = _make_adapter(await_return=NotifyResult("dismiss"))
    _h_notify_await(adapter, {"token": "9", "timeout": 600.0})
    call_args = adapter.ctrl.await_notify.call_args
    assert isinstance(call_args.args[0], int)
    assert call_args.args[0] == 9


# ---------------------------------------------------------------------------
# METHOD_REGISTRY off_main_thread flags
# ---------------------------------------------------------------------------


def test_notify_open_is_not_off_main_thread() -> None:
    spec = METHOD_REGISTRY["notify.open"]
    assert spec.off_main_thread is False


def test_notify_await_is_off_main_thread() -> None:
    spec = METHOD_REGISTRY["notify.await"]
    assert spec.off_main_thread is True
