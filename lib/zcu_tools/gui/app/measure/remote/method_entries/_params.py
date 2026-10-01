"""ParamSpec factory shorthands for measure-gui remote specs."""

from __future__ import annotations

from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec


def required_string(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.STRING, required=True, description=desc)


def optional_string(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.STRING, required=False, description=desc)


def required_object(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.OBJECT, required=True, description=desc)


def default_object(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(
        name, JsonType.OBJECT, required=False, default={}, description=desc
    )


def required_json(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.JSON, required=True, description=desc)


def required_integer(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.INTEGER, required=True, description=desc)


def required_number(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.NUMBER, required=True, description=desc)


def default_number(name: str, default: float, desc: str = "") -> ParamSpec:
    return ParamSpec(
        name, JsonType.NUMBER, required=False, default=default, description=desc
    )


def default_integer(name: str, default: int, desc: str = "") -> ParamSpec:
    return ParamSpec(
        name, JsonType.INTEGER, required=False, default=default, description=desc
    )


def optional_integer(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.INTEGER, required=False, description=desc)


def default_boolean(name: str, *, default: bool, desc: str = "") -> ParamSpec:
    return ParamSpec(
        name, JsonType.BOOLEAN, required=False, default=default, description=desc
    )


def save_comment() -> ParamSpec:
    return ParamSpec(
        "comment",
        JsonType.STRING,
        required=False,
        description="Omit to use the current GUI Save comment draft",
    )
