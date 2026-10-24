"""The cfg/comment envelope stored in Labber's root comment."""

import json
import time

from pydantic import TypeAdapter

from .container_models import JsonObject

_CFG_JSON = TypeAdapter(JsonObject)
_TEXT = TypeAdapter(str | None)


def encode_labber_comment(values: JsonObject, comment: str | None = None) -> str:
    """Encode cfg values and optional user text with a local creation timestamp.

    Preserve the cfg/comment/timestamp JSON envelope and formatted local time
    YYYY-MM-DD HH:MM:SS. Invalid JSON values or non-string comment raise
    ValueError. This timestamp is formatting time, not run start/completion.
    """
    cfg = _CFG_JSON.validate_python(values, strict=True)
    comment = _TEXT.validate_python(comment, strict=True)
    envelope: JsonObject = {"cfg": cfg}
    if comment is not None:
        envelope["comment"] = comment
    envelope["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return json.dumps(envelope, indent=2)
