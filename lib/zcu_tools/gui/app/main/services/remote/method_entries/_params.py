"""ParamSpec factory shorthands for measure-gui remote specs."""

from __future__ import annotations

from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec


def _str(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.STRING, required=True, description=desc)


def _str_opt(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.STRING, required=False, description=desc)


def _obj(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.OBJECT, required=True, description=desc)


def _obj_default(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(
        name, JsonType.OBJECT, required=False, default={}, description=desc
    )


def _json(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.JSON, required=True, description=desc)


def _expected_versions() -> ParamSpec:
    """Wire-only optimistic-concurrency guard param (mcp-filled, MCP-hidden).

    The mcp layer attaches the resource->version map this op depends on; the
    server compares it atomically. Hidden from the agent-facing MCP schema.
    """
    return ParamSpec(
        "expected_versions",
        JsonType.OBJECT,
        required=False,
        default={},
        description="Resource versions the caller depends on (mcp bookkeeping)",
        mcp_hidden=True,
    )


def _int(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.INTEGER, required=True, description=desc)


def _num(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.NUMBER, required=True, description=desc)


def _num_default(name: str, default: float, desc: str = "") -> ParamSpec:
    return ParamSpec(
        name, JsonType.NUMBER, required=False, default=default, description=desc
    )


def _int_default(name: str, default: int, desc: str = "") -> ParamSpec:
    return ParamSpec(
        name, JsonType.INTEGER, required=False, default=default, description=desc
    )


def _int_opt(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.INTEGER, required=False, description=desc)


def _bool_default(name: str, default: bool, desc: str = "") -> ParamSpec:
    return ParamSpec(
        name, JsonType.BOOLEAN, required=False, default=default, description=desc
    )


def _comment() -> ParamSpec:
    return ParamSpec(
        "comment", JsonType.STRING, required=False, default="", description="Comment"
    )
