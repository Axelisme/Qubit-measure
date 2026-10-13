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


def expr_text(value: JsonValue) -> str | None:
    """Return the text of an Expr envelope, exactly {"expr": <str>}, else None.

    Any other value, including an object with extra keys or non-string
    text, is not an Expr envelope and returns None.
    """
    if isinstance(value, dict) and set(value) == {"expr"}:
        text = value["expr"]
        if isinstance(text, str):
            return text
    return None


def default_json(value: object) -> JsonValue:
    """Encode an ordinary value as finite JSON, including models, tuples and complex."""
    if isinstance(value, BaseModel):
        return default_json(value.model_dump(mode="python"))
    if isinstance(value, complex):
        return copy_json(encode_complex(value))
    if isinstance(value, (list, tuple)):
        return [default_json(item) for item in value]
    if isinstance(value, Mapping):
        return copy_object({key: default_json(item) for key, item in value.items()})
    return copy_json(value)
