from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.event_bus import BaseEventBus, EventOrigin
from zcu_tools.gui.expected_error import (
    ExpectedError,
    ExpectedErrorCategory,
    FailedPreconditionError,
    InvalidInputError,
)
from zcu_tools.gui.remote.control_service import (
    RemoteControlServiceBase,
    SubscriptionCtx,
)
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.method_spec import BoundMethod, MethodSpec
from zcu_tools.gui.remote.rpc_endpoint import ClientLink, MainThreadDispatcher
from zcu_tools.gui.session.value_lookup import ProviderError


class _ImmediateSignal:
    def emit(self, callback: Callable[[], None]) -> None:
        callback()


class _InvalidCategoryExpectedError(ExpectedError):
    category = cast(ExpectedErrorCategory, "invalid_category")
    reason_code = ""


def _service() -> RemoteControlServiceBase:
    service = object.__new__(RemoteControlServiceBase)
    service.ctrl = SimpleNamespace(bus=BaseEventBus())
    service._dispatcher = cast(
        MainThreadDispatcher, SimpleNamespace(invoke=_ImmediateSignal())
    )
    service._endpoint = MagicMock()
    return service


@pytest.mark.parametrize("off_main_thread", [False, True])
def test_dispatch_scopes_handler_to_stable_per_connection_agent_origin(
    off_main_thread: bool,
) -> None:
    service = _service()
    first = _link()
    second = _link()
    observed: list[EventOrigin] = []

    def _capture(
        _adapter: RemoteControlServiceBase, _params: Mapping[str, object]
    ) -> Mapping[str, object]:
        observed.append(service.ctrl.bus.current_origin)
        return {}

    spec = BoundMethod(
        handler=_capture,
        spec=MethodSpec(
            timeout_seconds=1.0,
            description="origin capture",
            off_main_thread=off_main_thread,
        ),
    )
    service._dispatch_on_main(first, "request-1", "test.origin", spec, {})
    service._dispatch_on_main(first, "request-2", "test.origin", spec, {})
    service._dispatch_on_main(second, "request-3", "test.origin", spec, {})

    first_ctx = cast(SubscriptionCtx, first.app_ctx)
    second_ctx = cast(SubscriptionCtx, second.app_ctx)
    assert first_ctx.client_id != second_ctx.client_id
    assert observed == [
        EventOrigin(kind="agent", client_id=first_ctx.client_id),
        EventOrigin(kind="agent", client_id=first_ctx.client_id),
        EventOrigin(kind="agent", client_id=second_ctx.client_id),
    ]
    assert service.ctrl.bus.current_origin == EventOrigin(kind="user")


def _link() -> ClientLink:
    link = ClientLink("test", token_required=False)
    link.app_ctx = SubscriptionCtx()
    return link


def _spec(exc: BaseException, *, off_main_thread: bool = False) -> BoundMethod:
    def _raise(
        adapter: RemoteControlServiceBase, params: Mapping[str, object]
    ) -> Mapping[str, object]:
        del adapter, params
        raise exc

    return BoundMethod(
        handler=_raise,
        spec=MethodSpec(
            timeout_seconds=1.0,
            description="synthetic dispatch test",
            off_main_thread=off_main_thread,
        ),
    )


def _dispatch(exc: BaseException, *, off_main_thread: bool = False) -> MagicMock:
    service = _service()
    service._dispatch_on_main(
        _link(),
        "request-1",
        "test.raise",
        _spec(exc, off_main_thread=off_main_thread),
        {},
    )
    return cast(MagicMock, service._endpoint)


@pytest.mark.parametrize("off_main_thread", [False, True])
@pytest.mark.parametrize(
    ("exc", "code", "reason"),
    [
        (
            InvalidInputError("bad input", reason_code="bad_field"),
            ErrorCode.INVALID_PARAMS,
            "bad_field",
        ),
        (
            FailedPreconditionError("not ready", reason_code="no_context"),
            ErrorCode.PRECONDITION_FAILED,
            "no_context",
        ),
    ],
)
def test_dispatch_projects_expected_errors_identically_on_both_thread_paths(
    exc: ExpectedError,
    code: ErrorCode,
    reason: str,
    off_main_thread: bool,
) -> None:
    endpoint = _dispatch(exc, off_main_thread=off_main_thread)

    endpoint.reply_error.assert_called_once()
    assert endpoint.reply_error.call_args.kwargs == {
        "rid": "request-1",
        "code": code,
        "message": str(exc),
        "reason": reason,
        "data": None,
    }


@pytest.mark.parametrize("off_main_thread", [False, True])
def test_dispatch_preserves_direct_remote_error_with_structured_data(
    off_main_thread: bool,
) -> None:
    error = RemoteError(
        ErrorCode.PRECONDITION_FAILED,
        "structured",
        reason="stale",
        data={"stale": ["context"]},
    )

    endpoint = _dispatch(error, off_main_thread=off_main_thread)

    assert endpoint.reply_error.call_args.kwargs == {
        "rid": "request-1",
        "code": ErrorCode.PRECONDITION_FAILED,
        "message": "structured",
        "reason": "stale",
        "data": {"stale": ["context"]},
    }


@pytest.mark.parametrize("off_main_thread", [False, True])
@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("programmer bug"),
        ProviderError("ctx.value", "provider", RuntimeError("provider bug")),
        OSError("disk failed"),
    ],
)
def test_dispatch_keeps_unexpected_errors_as_controller_errors_with_traceback(
    exc: BaseException,
    off_main_thread: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR, logger="zcu_tools.gui.remote.control_service"):
        endpoint = _dispatch(exc, off_main_thread=off_main_thread)

    assert endpoint.reply_error.call_args.kwargs == {
        "rid": "request-1",
        "code": ErrorCode.CONTROLLER_ERROR,
        "message": str(exc),
    }
    assert any(record.exc_info is not None for record in caplog.records)


@pytest.mark.parametrize("off_main_thread", [False, True])
def test_dispatch_contains_expected_error_projection_failure(
    off_main_thread: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = _InvalidCategoryExpectedError("invalid expected-error category")
    with caplog.at_level(logging.ERROR, logger="zcu_tools.gui.remote.control_service"):
        endpoint = _dispatch(error, off_main_thread=off_main_thread)

    kwargs = endpoint.reply_error.call_args.kwargs
    assert kwargs["rid"] == "request-1"
    assert kwargs["code"] is ErrorCode.CONTROLLER_ERROR
    assert "invalid_category" in cast(str, kwargs["message"])
    assert any(record.exc_info is not None for record in caplog.records)
