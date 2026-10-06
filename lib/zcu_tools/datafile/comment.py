"""The cfg/comment envelope stored in Labber's root comment."""

import json
import time
from dataclasses import dataclass

from pydantic import TypeAdapter

from .native_models import JsonObject


@dataclass(frozen=True, kw_only=True)
class LabberComment:
    """Decoded Labber comment without an experiment cfg model.

    cfg is complete JSON configuration or None when absent. comment is the
    envelope's optional user text, or the original text for non-envelopes.
    timestamp is the optional local wall-clock string in a recognized envelope;
    free text never supplies a timestamp or historical run evidence.
    """

    cfg: JsonObject | None = None
    comment: str | None = None
    timestamp: str | None = None


_COMMENT = TypeAdapter(LabberComment)
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


def decode_labber_comment(text: str) -> LabberComment:
    """Decode the cfg/comment envelope, preserving non-envelope text verbatim.

    Non-JSON, JSON scalar/array and objects with no cfg/comment/timestamp keys
    return cfg/timestamp=None and comment=text. An object containing any of
    those keys is an envelope: wrong field types raise ValueError. Unknown
    object keys are ignored. No experiment model is imported or validated.
    """
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return LabberComment(comment=text)
    if not isinstance(raw, dict) or not {"cfg", "comment", "timestamp"} & raw.keys():
        return LabberComment(comment=text)
    return _COMMENT.validate_json(text, strict=True)
