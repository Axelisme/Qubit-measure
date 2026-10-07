"""Experiment cfg mapping to the datafile-owned Labber comment codec."""

from __future__ import annotations

from pydantic import TypeAdapter

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import (
    CfgSnapshot,
    JsonObject,
    decode_labber_comment,
    encode_labber_comment,
)
from zcu_tools.utils import format_obj


def make_labber_cfg_snapshot(cfg: ConfigBase, *, schema_version: str) -> CfgSnapshot:
    """Map this cfg to a JSON snapshot for the Labber writer.

    cfg supplies the experiment's complete configuration; schema_version comes
    from its declared spec. cfg_type records the cfg class name, not an import
    path. Convert array/NumPy/QickParam values with the existing plain-value
    mapping. Unsupported JSON values raise ValueError. No live state is read.
    """
    return CfgSnapshot(
        values=TypeAdapter(JsonObject).validate_python(format_obj(cfg.to_dict())),
        cfg_type=type(cfg).__name__,
        schema_version=schema_version,
    )


def make_comment(cfg: ConfigBase, comment: str | None = None) -> str:
    """Map cfg to JSON and encode it with optional text and local format time.

    cfg is the experiment's ConfigBase snapshot. format_obj converts its array
    and NumPy values into JSON values. Invalid cfg JSON or text raises ValueError.
    The envelope encoding belongs to datafile, not this adapter.
    """
    return encode_labber_comment(format_obj(cfg.to_dict()), comment)


def parse_comment(
    comment: str,
) -> tuple[JsonObject | None, str | None, str | None]:
    """Return cfg JSON, user text and local envelope timestamp, in that order.

    Non-envelope text is preserved in the second result with cfg/time absent.
    Recognized envelopes with invalid field types raise ValueError. This adapter
    does not validate cfg as an experiment model or infer historical run time.
    """
    decoded = decode_labber_comment(comment)
    return decoded.cfg, decoded.comment, decoded.timestamp
