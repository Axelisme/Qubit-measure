"""Nullable parameter declarations through schema and in-process dispatch."""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.remote.control_service import RemoteAppSpec, RemoteControlServiceBase
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.method_spec import MethodSpec, build_method_registry
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec, build_input_schema
from zcu_tools.gui.remote.rpc_endpoint import ClientLink, ControlOptions
from zcu_tools.gui.remote.wire import Request


@pytest.mark.parametrize(
    "kind,schema_type",
    [
        (JsonType.STRING, "string"),
        (JsonType.INTEGER, "integer"),
        (JsonType.NUMBER, "number"),
        (JsonType.BOOLEAN, "boolean"),
        (JsonType.OBJECT, "object"),
        (JsonType.ARRAY, "array"),
        (JsonType.NUMBER_PAIRS, "array"),
    ],
)
def test_nullable_schema_keeps_required_key_and_adds_null_value(
    kind: JsonType, schema_type: str
) -> None:
    spec = ParamSpec("base", kind, required=True, nullable=True)
    schema = build_input_schema((spec,))
    assert schema.get("required") == ["base"]
    assert schema["properties"]["base"].get("type") == [schema_type, "null"]


def test_nullable_schema_preserves_enum_and_untyped_json() -> None:
    enum_schema = build_input_schema(
        (
            ParamSpec(
                "base",
                JsonType.STRING,
                required=False,
                nullable=True,
                enum=("fast", "careful"),
            ),
        )
    )
    assert "required" not in enum_schema
    assert enum_schema["properties"]["base"] == {
        "type": ["string", "null"],
        "enum": ["fast", "careful", None],
    }
    json_schema = build_input_schema((ParamSpec("base", JsonType.JSON, nullable=True),))
    assert json_schema.get("required") == ["base"]
    assert json_schema["properties"]["base"] == {}


class ImmediateOwnerScheduler:
    """Deliver synthetic dispatch on the calling owner thread."""

    def is_owner_thread(self) -> bool:
        return True

    def post(self, callback: Callable[[], None]) -> None:
        callback()

    def call[T](self, callback: Callable[[], T]) -> T:
        return callback()


@dataclass
class HandlerState:
    bus: BaseEventBus = field(default_factory=BaseEventBus)
    admitted: list[dict[str, object]] = field(default_factory=list)


def test_dispatch_nullable_base_distinguishes_absence_null_and_wrong_type() -> None:
    base = ParamSpec("base", JsonType.STRING, nullable=True)
    state = HandlerState()

    def echo(_adapter: object, params: Mapping[str, object]) -> dict[str, object]:
        result = dict(params)
        state.admitted.append(result)
        return result

    service = RemoteControlServiceBase(
        state,
        ControlOptions(port=0),
        app=RemoteAppSpec(
            wire_version=1,
            gui_version=1,
            server_name="NullableContract",
            method_registry=build_method_registry(
                {"test.reload": echo},
                {
                    "test.reload": MethodSpec(
                        1.0,
                        "Synthetic nullable base",
                        (
                            base,
                            ParamSpec(
                                "legacy", JsonType.STRING, required=False, default="old"
                            ),
                            ParamSpec(
                                "optional",
                                JsonType.STRING,
                                required=False,
                                default="fallback",
                                nullable=True,
                            ),
                        ),
                    )
                },
            ),
            event_serializers={},
            wire_event_name=str,
        ),
        owner_scheduler=ImmediateOwnerScheduler(),
    )
    link = ClientLink("nullable-test", token_required=False)
    service.on_client_open(link)

    def call(params: Mapping[str, object]) -> dict[str, object]:
        service.route(link, Request("nullable", "test.reload", dict(params)))
        return json.loads(link.outbound.get_nowait())

    try:
        assert call({"base": None}) == {
            "id": "nullable",
            "ok": True,
            "result": {"base": None, "legacy": "old", "optional": "fallback"},
        }
        assert call({"base": "hash", "legacy": None, "optional": None})["result"] == {
            "base": "hash",
            "legacy": "old",
            "optional": None,
        }
        for params in ({}, {"base": 1}, {"base": False}, {"base": ""}):
            with pytest.raises(RemoteError) as caught:
                call(params)
            assert caught.value.code == ErrorCode.INVALID_PARAMS
        assert state.admitted == [
            {"base": None, "legacy": "old", "optional": "fallback"},
            {"base": "hash", "legacy": "old", "optional": None},
        ]
    finally:
        service.on_client_close(link, on_owner_thread=True)
