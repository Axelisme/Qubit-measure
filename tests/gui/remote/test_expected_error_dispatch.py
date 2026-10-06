from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import TypedDict, TypeVar, cast

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
from zcu_tools.gui.remote.rpc_endpoint import ClientLink, ControlOptions
from zcu_tools.gui.remote.wire import Request
from zcu_tools.gui.session.value_lookup import ProviderError

_T = TypeVar("_T")


class _ImmediateOwnerScheduler:
    def is_owner_thread(self) -> bool:
        return True

    def post(self, callback: Callable[[], None]) -> None:
        callback()

    def call(self, callback: Callable[[], _T]) -> _T:
        del callback
        raise AssertionError("call is not used by dispatch")


class _InvalidCategoryExpectedError(ExpectedError):
    category = cast(ExpectedErrorCategory, "invalid_category")
    reason_code = ""


class ErrorResponse(TypedDict):
    id: str
    ok: bool
    error: dict[str, object]


def _service(
    registry: Mapping[str, BoundMethod], bus: BaseEventBus | None = None
) -> RemoteControlServiceBase:
    return RemoteControlServiceBase(
        SimpleNamespace(bus=bus if bus is not None else BaseEventBus()),
        ControlOptions(port=0),
        owner_scheduler=_ImmediateOwnerScheduler(),
        wire_version=1,
        gui_version=1,
        server_name="DispatchTest",
        method_registry=registry,
        event_serializers={},
        wire_event_name=str,
    )


@pytest.mark.parametrize("off_main_thread", [False, True])
def test_dispatch_scopes_handler_to_stable_per_connection_agent_origin(
    off_main_thread: bool,
) -> None:
    bus = BaseEventBus()
    observed: list[EventOrigin] = []

    def _capture(
        _adapter: RemoteControlServiceBase, _params: Mapping[str, object]
    ) -> Mapping[str, object]:
        observed.append(bus.current_origin)
        return {}

    spec = BoundMethod(
        handler=_capture,
        spec=MethodSpec(
            timeout_seconds=1.0,
            description="origin capture",
            off_main_thread=off_main_thread,
        ),
    )
    service = _service({"test.origin": spec}, bus)
    first, second = _link(service), _link(service)
    service.route(first, Request("request-1", "test.origin", {}))
    service.route(first, Request("request-2", "test.origin", {}))
    service.route(second, Request("request-3", "test.origin", {}))

    first_ctx, second_ctx = first.app_ctx, second.app_ctx
    assert isinstance(first_ctx, SubscriptionCtx)
    assert isinstance(second_ctx, SubscriptionCtx)
    assert first_ctx.client_id != second_ctx.client_id
    assert observed == [
        EventOrigin(kind="agent", client_id=first_ctx.client_id),
        EventOrigin(kind="agent", client_id=first_ctx.client_id),
        EventOrigin(kind="agent", client_id=second_ctx.client_id),
    ]
    assert bus.current_origin == EventOrigin(kind="user")


def _link(service: RemoteControlServiceBase) -> ClientLink:
    link = ClientLink("test", token_required=False)
    service.on_client_open(link)
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


def _dispatch(exc: BaseException, *, off_main_thread: bool = False) -> ErrorResponse:
    service = _service({"test.raise": _spec(exc, off_main_thread=off_main_thread)})
    link = _link(service)
    service.route(link, Request("request-1", "test.raise", {}))
    response: ErrorResponse = json.loads(link.outbound.get_nowait())
    assert response["id"] == "request-1"
    assert not response["ok"]
    return response


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
    response = _dispatch(exc, off_main_thread=off_main_thread)
    assert response["error"] == {
        "code": code.value,
        "message": str(exc),
        "reason": reason,
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
    response = _dispatch(error, off_main_thread=off_main_thread)
    assert response["error"] == {
        "code": ErrorCode.PRECONDITION_FAILED.value,
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
        response = _dispatch(exc, off_main_thread=off_main_thread)
    assert response["error"] == {
        "code": ErrorCode.CONTROLLER_ERROR.value,
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
        response = _dispatch(error, off_main_thread=off_main_thread)
    assert response["error"]["code"] == ErrorCode.CONTROLLER_ERROR.value
    assert "invalid_category" in str(response["error"]["message"])
    assert any(record.exc_info is not None for record in caplog.records)
