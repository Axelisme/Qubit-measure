"""Owner handler warnings through the public request/reply and logging seams."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TypeVar

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.remote.control_service import RemoteAppSpec, RemoteControlServiceBase
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.method_spec import BoundMethod, MethodSpec
from zcu_tools.gui.remote.rpc_endpoint import ClientLink, ControlOptions
from zcu_tools.gui.remote.wire import Request

_T = TypeVar("_T")
_LOGGER = "zcu_tools.gui.remote.control_service"


@dataclass
class Clock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


@dataclass
class OwnerScheduler:
    clock: Clock

    def is_owner_thread(self) -> bool:
        return True

    def post(self, callback: Callable[[], None]) -> None:
        # Simulate queue delay separately from the handler's own execution time.
        self.clock.now += 2.0
        callback()

    def call(self, callback: Callable[[], _T]) -> _T:
        return callback()


@pytest.mark.parametrize("outcome", ["success", "remote", "expected", "unexpected"])
def test_owner_handler_latency_warns_without_changing_replies(
    outcome: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = Clock()
    monkeypatch.setattr(time, "perf_counter", clock)
    duration = 0.0

    def handler(
        _adapter: RemoteControlServiceBase, _params: Mapping[str, object]
    ) -> dict[str, object]:
        clock.now += duration
        if outcome == "remote":
            raise RemoteError(
                ErrorCode.PRECONDITION_FAILED,
                "not ready",
                reason="stale",
                data={"target": "point"},
            )
        if outcome == "expected":
            raise InvalidInputError("bad input", reason_code="bad_field")
        if outcome == "unexpected":
            raise OSError("disk failed")
        return {"value": "committed"}

    registry = {
        name: BoundMethod(handler, MethodSpec(10.0, name, off_main_thread=off_main))
        for name, off_main in [("test.owner", False), ("test.off_main", True)]
    }
    service = RemoteControlServiceBase(
        SimpleNamespace(bus=BaseEventBus()),
        ControlOptions(port=0),
        app=RemoteAppSpec(
            wire_version=1,
            gui_version=1,
            server_name="HandlerLatencyTest",
            method_registry=registry,
            event_serializers={},
            wire_event_name=str,
        ),
        owner_scheduler=OwnerScheduler(clock),
    )
    link = ClientLink("test", token_required=False)
    service.on_client_open(link)
    expected_reply = {
        "success": {"ok": True, "result": {"value": "committed"}},
        "remote": {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "message": "not ready",
                "reason": "stale",
                "data": {"target": "point"},
            },
        },
        "expected": {
            "ok": False,
            "error": {
                "code": "invalid_params",
                "message": "bad input",
                "reason": "bad_field",
            },
        },
        "unexpected": {
            "ok": False,
            "error": {"code": "controller_error", "message": "disk failed"},
        },
    }[outcome]

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        for method, duration, warned in [
            ("test.owner", 0.75, True),
            ("test.owner", 0.5, False),
            ("test.owner", 0.25, False),
            ("test.off_main", 0.75, False),
            ("test.owner", 0.501, True),
        ]:
            caplog.clear()
            service.route(link, Request("request", method, {}))
            assert json.loads(link.outbound.get_nowait()) == {
                "id": "request",
                **expected_reply,
            }
            warnings = [
                record
                for record in caplog.records
                if record.name == _LOGGER and record.levelno == logging.WARNING
            ]
            assert len(warnings) == int(warned)
            if warned:
                message = warnings[0].getMessage()
                assert method in message
                assert str(round(duration * 1000)) in message
                assert "ms" in message
                assert "\n" not in message
