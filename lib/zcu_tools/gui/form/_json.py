"""JSON boundary conversion, including ordinary Pydantic default values."""

import math
from collections.abc import Mapping

from pydantic import BaseModel, JsonValue

from zcu_tools.gui.cfg.codec import encode_complex


def copy_json(value: object) -> JsonValue:
    """Detach finite JSON data; non-JSON values or non-string keys raise ValueError."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, list):
        return [copy_json(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            result[key] = copy_json(item)
        return result
    raise ValueError(f"Not JSON data: {type(value).__name__}")


def copy_object(value: object) -> dict[str, JsonValue]:
    """Detach a finite JSON object; other envelopes raise ValueError."""
    copied = copy_json(value)
    if not isinstance(copied, dict):
        raise ValueError("Form JSON input must be an object")
    return copied


def default_json(value: object) -> JsonValue:
    """Encode ordinary defaults, including nested models, tuples and complex values."""
    if isinstance(value, BaseModel):
        return default_json(value.model_dump(mode="python"))
    if isinstance(value, complex):
        return copy_json(encode_complex(value))
    if isinstance(value, (list, tuple)):
        return [default_json(item) for item in value]
    if isinstance(value, Mapping):
        return copy_object({key: default_json(item) for key, item in value.items()})
    return copy_json(value)
