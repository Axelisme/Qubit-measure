"""Typed ledger envelopes and payloads; producers supply identities and evidence."""

import math
import re
from datetime import datetime, timedelta
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)


def _json_value(value: object) -> object:
    """Reject non-JSON Python values before Pydantic can coerce them."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, list):
        for item in value:
            _json_value(item)
        return value
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item)
        return value
    raise ValueError("value must be strict JSON with finite numbers and string keys")


type JsonValue = Annotated[
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue],
    BeforeValidator(_json_value),
]
"""A JSON null, boolean, integer, finite float, string, list or string-key object."""
type JsonObject = Annotated[
    dict[str, JsonValue], Field(strict=True), BeforeValidator(_json_value)
]
"""A string-key JSON object; no bytes, tuples or non-finite numbers."""
type Origin = Literal["gui", "notebook", "mcp"]
"""Operation source; offline CLI operations use notebook."""
type OutputFormat = Literal["data_h5", "labber", "png", "json"]
"""Serialization format of one saved output."""


def _uuid(value: str) -> str:
    UUID(value)
    return value


def _nonempty(value: str) -> str:
    if not value.strip():
        raise ValueError("expected a nonempty string")
    return value


type _Identity = Annotated[StrictStr, AfterValidator(_uuid)]
type _Nonempty = Annotated[StrictStr, AfterValidator(_nonempty)]


def validate_origin(origin: Origin, call_id: str | None) -> None:
    """Reject unknown origins, missing MCP IDs and IDs on non-MCP operations."""
    if origin not in ("gui", "notebook", "mcp"):
        raise ValueError("origin must be gui, notebook or mcp")
    if origin == "mcp":
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("origin=mcp requires a nonempty call_id")
    elif call_id is not None:
        raise ValueError("call_id must be None for non-MCP origins")


class _LedgerModel(BaseModel):
    # All envelope and nested payload models share immutability and extra policy.
    model_config = ConfigDict(frozen=True, extra="forbid")


class SavedOutput(_LedgerModel):
    """One output attempt; path is evidence, not resolved or inspected by ledger.

    artifact names the result artifact. member is data, figure or analysis.
    format selects serialization. status is saved or failed; error must be a
    nonempty failure reason for failed and None for saved.
    """

    artifact: StrictStr
    member: Literal["data", "figure", "analysis"]
    format: OutputFormat
    path: StrictStr
    status: Literal["saved", "failed"]
    error: StrictStr | None = None

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        """Require error only for a failed output."""
        if self.status == "failed":
            if self.error is None or not self.error.strip():
                raise ValueError("failed status requires a nonempty error")
        elif self.error is not None:
            raise ValueError("saved status requires error=None")
        return self


class AcceptedWrite(_LedgerModel):
    """Accepted working-unit value at a nonempty dot-separated container path."""

    path: _Nonempty
    value: JsonValue

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        """Reject empty path segments; no escaping or indexing is supported."""
        if any(not segment.strip() for segment in value.split(".")):
            raise ValueError("path must contain nonempty dot-separated segments")
        return value


class SourceReference(_LedgerModel):
    """Optional accepted evidence: kind run names a run ID, event names a UUID.

    id must be nonempty. This model does not resolve either kind of identity.
    """

    kind: Literal["run", "event"]
    id: _Nonempty

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        """Require UUID syntax for event references, without ledger lookup."""
        if self.kind == "event":
            _uuid(self.id)
        return self


class AcquiredPayload(_LedgerModel):
    """Acquired run evidence supplied by the acquisition producer.

    run_id identifies the run, tab its display name, tab_id its optional stable
    ID, experiment the spec tag, cfg_summary the JSON configuration summary,
    point the optional point label, and roles maps role names to component names.
    """

    run_id: StrictStr
    tab: StrictStr
    tab_id: StrictStr | None = None
    experiment: StrictStr
    cfg_summary: JsonObject
    point: StrictStr | None = None
    roles: dict[StrictStr, StrictStr]


class SavedPayload(_LedgerModel):
    """Run output attempts; labber_path is an optional successful Labber path.

    run_id identifies the run; outputs retains each SavedOutput in caller order.
    """

    run_id: StrictStr
    outputs: tuple[SavedOutput, ...]
    labber_path: StrictStr | None = None


class AnalyzedPayload(_LedgerModel):
    """Analysis evidence: analysis_kind names the method, run_ids its input runs.

    summary is a JSON object; large details may be attached as a ledger record.
    """

    analysis_kind: StrictStr
    run_ids: tuple[StrictStr, ...]
    summary: JsonObject


class AcceptedPayload(_LedgerModel):
    """One or more accepted writes and their JSON analysis_summary.

    run_id and source optionally identify existing evidence; both may be None.
    No prior acquisition, analysis event or database run is required.
    """

    writes: Annotated[tuple[AcceptedWrite, ...], Field(min_length=1)]
    analysis_summary: JsonObject
    run_id: StrictStr | None = None
    source: SourceReference | None = None


class ImportPayload(_LedgerModel):
    """Historical evidence copied from another entry into a local import event.

    source_entry_id and source_event_id are UUIDs. source_event_json is the exact
    original JSON line without LF. source_record is the original optional record
    reference, never a query path. Only the outer event.record is queried.
    """

    source_entry_id: _Identity
    source_event_id: _Identity
    source_event_json: StrictStr
    source_record: StrictStr | None = None


class LedgerEvent(_LedgerModel):
    """Immutable event envelope, validated with its corresponding typed payload.

    format is zcu.ledger; format_version is 1.x (append only accepts 1.0).
    id and entry_id are UUIDs; kind selects acquired, saved, analyzed, accepted
    or import. at is an ISO 8601 UTC timestamp with Z or +00:00. origin identifies
    the operation; call_id is required only for MCP. record is None on append,
    or records/<id>.json on disk. payload carries the kind-specific evidence.
    New 1.0 models reject unknown fields; readers can project future-minor fields
    while preserving the complete raw line separately in LedgerEntry.
    """

    format: Literal["zcu.ledger"] = "zcu.ledger"
    format_version: StrictStr = "1.0"
    id: _Identity
    kind: Literal["acquired", "saved", "analyzed", "accepted", "import"]
    at: StrictStr
    entry_id: _Identity
    origin: Origin
    call_id: StrictStr | None = None
    record: StrictStr | None = None
    payload: (
        AcquiredPayload
        | SavedPayload
        | AnalyzedPayload
        | AcceptedPayload
        | ImportPayload
    )

    @field_validator("format_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        """Accept only a major 1 ledger version, without changing its spelling."""
        if re.fullmatch(r"1\.(0|[1-9][0-9]*)", value) is None:
            raise ValueError("format_version must be a supported 1.x version")
        return value

    @field_validator("at")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        """Require an explicit UTC offset, preserving caller timestamp text."""
        if not value.endswith(("Z", "+00:00")) or datetime.fromisoformat(
            value
        ).utcoffset() != timedelta(0):
            raise ValueError("at must be an ISO 8601 UTC timestamp")
        return value

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        """Check operation identity, canonical record reference and payload kind."""
        validate_origin(self.origin, self.call_id)
        if self.record is not None and self.record != f"records/{self.id}.json":
            raise ValueError("record must be records/<id>.json")
        expected: dict[str, type[BaseModel]] = {
            "acquired": AcquiredPayload,
            "saved": SavedPayload,
            "analyzed": AnalyzedPayload,
            "accepted": AcceptedPayload,
            "import": ImportPayload,
        }
        if not isinstance(self.payload, expected[self.kind]):
            raise ValueError("kind does not match payload type")
        return self
