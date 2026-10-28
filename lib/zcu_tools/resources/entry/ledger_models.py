"""Typed ledger envelopes and payloads; producers supply identities and evidence."""

import math
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated, Literal, Self, get_args, get_origin
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictStr,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import (
    PydanticUndefined,
    SchemaSerializer,
    SchemaValidator,
    core_schema,
)

from zcu_tools.core.complex_codec import encode_complex
from zcu_tools.core.numeric_scalar import normalize_numpy_scalar
from zcu_tools.resources.parameter_fields import model_field_schema

from .aliases import AliasValue


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


def validate_origin(origin: Origin) -> None:
    """Reject origins other than gui, notebook or mcp before resource writes."""
    if origin not in ("gui", "notebook", "mcp"):
        raise ValueError("origin must be gui, notebook or mcp")


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
    """One accepted value for a nonempty dot-separated container path.

    path is a concrete destination with nonempty segments; no escaping or
    indexing is supported. value stores strict JSON in working units.
    At the field root, numpy floating scalars become Python float with normal
    rounding, and numpy integer scalars become Python int without loss.
    Non-finite results, including finite extended floats that overflow Python
    float, raise ValueError. Native complex input at this field alone becomes
    the shared __complex__ tag before JSON validation. Nested numpy scalars,
    nested complex, non-finite components and other non-JSON values raise
    ValueError; strings are never decoded.
    """

    path: _Nonempty
    value: JsonValue

    @field_validator("value", mode="before")
    @classmethod
    def _normalize_scalar_value(cls, value: object) -> object:
        value = normalize_numpy_scalar(value)
        if isinstance(value, complex):
            return encode_complex(value)
        return value

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

    run_id identifies the run, experiment the spec tag, point the optional
    point label, and roles maps
    aliases to scalar names or ordered name lists. Configuration lives with run data.
    """

    run_id: StrictStr
    experiment: StrictStr
    point: StrictStr | None = None
    roles: dict[StrictStr, AliasValue]


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
    """One or more concrete accepted writes linked to their source evidence.

    run_id and source optionally identify existing evidence; both may be None.
    No prior acquisition, analysis event or database run is required.
    """

    writes: Annotated[tuple[AcceptedWrite, ...], Field(min_length=1)]
    run_id: StrictStr | None = None
    source: SourceReference | None = None


class ImportPayload(_LedgerModel):
    """Historical evidence copied from another entry into a local import event.

    source_event_id is the historical event UUID. source_event_json is the exact
    original JSON line without LF. source_record is the original optional record
    reference, never a query path. Only the outer event.record is queried.
    """

    source_event_id: _Identity
    source_event_json: StrictStr
    source_record: StrictStr | None = None


_PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    "acquired": AcquiredPayload,
    "saved": SavedPayload,
    "analyzed": AnalyzedPayload,
    "accepted": AcceptedPayload,
    "import": ImportPayload,
}


class LedgerEvent(_LedgerModel):
    """Immutable event envelope, validated with its corresponding typed payload.

    format is zcu.ledger; format_version is 1.x (append only accepts 1.0).
    id is an event UUID; kind selects acquired, saved, analyzed, accepted
    or import. at is an ISO 8601 UTC timestamp with Z or +00:00. origin identifies
    the frontend as gui, notebook or mcp. record is None on append,
    or records/<id>.json on disk. payload carries the kind-specific evidence.
    New 1.0 models reject unknown fields. RecordsLedger readers ignore unknown
    envelope/payload fields at every supported minor, preserving the complete raw
    line separately in LedgerEntry.
    """

    format: Literal["zcu.ledger"] = "zcu.ledger"
    format_version: StrictStr = "1.0"
    id: _Identity
    kind: Literal["acquired", "saved", "analyzed", "accepted", "import"]
    at: StrictStr
    origin: Origin
    record: StrictStr | None = None
    payload: (
        AcquiredPayload
        | SavedPayload
        | AnalyzedPayload
        | AcceptedPayload
        | ImportPayload
    )

    @field_validator("payload", mode="before")
    @classmethod
    def validate_payload(cls, value: object, info: ValidationInfo) -> object:
        """Validate a raw mapping or payload model against the envelope kind.

        info supplies Pydantic's previously validated kind/format_version fields
        and the ledger_read context flag used by RecordsLedger. Known-field/type
        errors always raise ValueError. Unknown keys are ignored when reading or
        projecting future 1.x, not used to select a branch; new 1.0 inputs reject
        them.
        """
        kind = info.data.get("kind")
        if kind not in _PAYLOAD_MODELS:
            # Invalid/missing kind is reported by its own required field.
            return value
        future = info.data.get("format_version", "1.0") != "1.0"
        reading = info.context is not None and info.context.get("ledger_read", False)
        return _PAYLOAD_MODELS[kind].model_validate(
            value, extra="ignore" if reading or future else "forbid"
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
        if self.record is not None and self.record != f"records/{self.id}.json":
            raise ValueError("record must be records/<id>.json")
        if not isinstance(self.payload, _PAYLOAD_MODELS[self.kind]):
            raise ValueError("kind does not match payload type")
        return self


@dataclass(frozen=True)
class LedgerAcquisition:
    """Run-index evidence: at is validated UTC text; payload holds run identity.

    The payload contains only run_id, experiment, optional point and roles.
    Envelope origin and attachment references are not consumed by this projection.
    """

    at: str
    payload: AcquiredPayload


@dataclass(frozen=True)
class LedgerSavedPaths:
    """Run-index evidence from successful outputs, without opening any path.

    run_id identifies the saved run. paths contains recorded exact path strings
    in output order, only for status=saved. Failed output paths and other output
    metadata remain unread; these strings do not authorize filesystem access.
    """

    run_id: str
    paths: tuple[str, ...]


class LedgerEventContent:
    """Readonly historical JSON with field-level reads and explicit full projection.

    data is one decoded JSON object; source identifies its ledger and line for
    diagnostics. Construction never validates unused payload fields. Returned
    values and projections are detached. Field defaults stay stable on this
    content. Leaf reads skip parent model checks; requested complete submodels
    and project() run their full validation. No file is read.
    """

    def __init__(self, data: JsonObject, *, source: str) -> None:
        self._data = deepcopy(data)
        self._source = source
        self._defaults: dict[str, JsonValue] = {}

    @property
    def id(self) -> str:
        """Validate the UUID event identity needed for indexing and duplicates."""
        return TypeAdapter(_Identity).validate_python(self._data.get("id"))

    def validate_index(self) -> None:
        """Validate only ledger header, event identity and declared run references.

        Missing headers, unsupported versions, invalid event/run IDs raise.
        Payload fields unrelated to those references and attachments stay opaque.
        """
        for field in ("format", "format_version", "id"):
            self.value(field)
        kind = self._data.get("kind")
        payload = self._data.get("payload")
        if kind in ("acquired", "saved"):
            self.value("payload.run_id")
        elif isinstance(payload, dict):
            for field in ("run_id", "run_ids"):
                if field in payload:
                    self.value(f"payload.{field}")

    def acquisition(self) -> LedgerAcquisition | None:
        """Read acquired identity/time, or None for another validated event kind.

        Validate only at and the acquired payload; unused envelope fields and
        attachment references stay opaque. Invalid consumed fields raise
        ValueError naming the source line. No files or attachments are opened.
        """
        if self.value("kind") != "acquired":
            return None
        try:
            payload = AcquiredPayload.model_validate(
                deepcopy(self._data.get("payload")), extra="ignore"
            )
        except ValueError as exc:
            raise ValueError(f"{self._source}: payload: {exc}") from exc
        return LedgerAcquisition(self._string_value("at"), payload)

    def saved_paths(self) -> LedgerSavedPaths | None:
        """Read saved run ID and successful paths, or None for another kind.

        Validate output sequence structure, each status and only saved paths.
        Other output fields, labber_path and attachment references stay opaque.
        Invalid consumed fields raise ValueError with source/path. Paths are
        evidence strings, not resolved/opened paths or an access capability.
        """
        if self.value("kind") != "saved":
            return None
        try:
            outputs, _ = self._field("payload.outputs")
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{self._source}: payload.outputs: {exc}") from exc
        if not isinstance(outputs, list):
            raise ValueError(f"{self._source}: payload.outputs: expected sequence")
        paths = tuple(
            self._string_value(f"payload.outputs.{index}.path")
            for index in range(len(outputs))
            if self.value(f"payload.outputs.{index}.status") == "saved"
        )
        return LedgerSavedPaths(self._string_value("payload.run_id"), paths)

    def _string_value(self, path: str) -> str:
        value = self.value(path)
        if not isinstance(value, str):
            raise ValueError(f"{self._source}: {path}: expected string")
        return value

    def project(self) -> LedgerEvent:
        """Validate the whole event/payload; unknown reader fields are ignored.

        Invalid values raise ValueError naming the original ledger line and field.
        This does not follow an attachment reference or touch the filesystem.
        """
        try:
            return LedgerEvent.model_validate(
                deepcopy(self._data), extra="ignore", context={"ledger_read": True}
            )
        except ValueError as exc:
            raise ValueError(f"{self._source}: {exc}") from exc

    def value(self, path: str) -> JsonValue:
        """Read one dotted field or canonical sequence index with field validators.

        Payload kind selects its declared schema, not a guessed union branch.
        Ancestors must have the requested mapping/sequence structure. Missing
        required fields and malformed requested values raise with source/path.
        A whole model-valued field runs that requested model's complete checks.
        """
        try:
            node, schema = self._field(path)
            value = SchemaValidator(schema).validate_python(
                deepcopy(node), extra="ignore"
            )
            return TypeAdapter(JsonValue).validate_python(
                SchemaSerializer(schema).to_python(value, mode="json")
            )
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{self._source}: {path}: {exc}") from exc

    def _payload_model(self) -> type[BaseModel]:
        kind = self.value("kind")
        if not isinstance(kind, str) or kind not in _PAYLOAD_MODELS:
            raise ValueError("unknown payload kind")
        return _PAYLOAD_MODELS[kind]

    def _field(self, path: str) -> tuple[object, core_schema.CoreSchema]:
        parts = path.split(".")
        if not all(parts):
            raise ValueError("expected a dotted field path")
        node: object = self._data
        model: type[BaseModel] | None = LedgerEvent
        annotation: object = JsonValue
        schema = TypeAdapter(JsonValue).core_schema
        for index, part in enumerate(parts):
            if index == 0 and part == "payload":
                annotation = self._payload_model()
                schema = TypeAdapter(annotation).core_schema
            else:
                annotation, schema = self._child_schema(model, annotation, part)
            node = self._child_value(node, part, schema, ".".join(parts[: index + 1]))
            model = (
                None
                if get_origin(annotation) in (dict, list, tuple)
                else next(
                    (
                        item
                        for item in (annotation, *get_args(annotation))
                        if isinstance(item, type) and issubclass(item, BaseModel)
                    ),
                    None,
                )
            )
        return node, schema

    @staticmethod
    def _child_schema(
        model: type[BaseModel] | None,
        annotation: object,
        part: str,
    ) -> tuple[object, core_schema.CoreSchema]:
        if model is not None:
            field = model.model_fields.get(part)
            if field is None:
                raise ValueError("unknown field")
            return field.rebuild_annotation(), model_field_schema(model, part)
        args = get_args(annotation)
        origin = get_origin(annotation)
        child = (
            args[1]
            if origin is dict and len(args) == 2
            else args[0]
            if origin in (list, tuple) and args
            else JsonValue
        )
        return child, TypeAdapter(child).core_schema

    def _child_value(
        self,
        node: object,
        part: str,
        schema: core_schema.CoreSchema,
        path: str,
    ) -> object:
        if isinstance(node, dict):
            if part in node:
                return node[part]
            if path not in self._defaults:
                default = SchemaValidator(schema).validate_python(PydanticUndefined)
                self._defaults[path] = TypeAdapter(JsonValue).validate_python(
                    SchemaSerializer(schema).to_python(default, mode="json")
                )
            return self._defaults[path]
        if isinstance(node, (tuple, list)) and (
            part == "0"
            or (part.isascii() and part.isdecimal() and not part.startswith("0"))
        ):
            if int(part) >= len(node):
                raise ValueError("index out of range")
            return node[int(part)]
        raise ValueError("required field or ancestor is missing")
