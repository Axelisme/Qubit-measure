"""Experiment-owned historical run documents and configuration envelopes."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    TypeAdapter,
    ValidationError,
    field_validator,
)
from pydantic_core import PydanticUndefined, SchemaValidator

from zcu_tools.datafile import JsonObject
from zcu_tools.format_version import YamlValue
from zcu_tools.resources.context.library import LibraryContent
from zcu_tools.resources.entry import ParameterContent
from zcu_tools.resources.entry.aliases import AliasValue
from zcu_tools.resources.parameter_fields import model_field_schema

_WORKFLOW_RUN_ID_PATTERN = r"[0-9a-f]{32}"
_RUN_ID_PATTERN = (
    rf"^(?:[0-9]{{8}}T[0-9]{{6}}Z-[a-z0-9]{{6}}|"
    rf"{_WORKFLOW_RUN_ID_PATTERN}-[0-9]{{6}}-[1-9][0-9]*)$"
)


@dataclass(frozen=True, kw_only=True)
class CfgSnapshot:
    """Complete JSON cfg and declaration identity, without a typed projection.

    values retains unknown nested keys. cfg_type is the declaring class name,
    not an import path. schema_version is that declaration's ASCII major.minor.
    validate_cfg_snapshot rejects malformed envelopes without changing values.
    """

    values: JsonObject
    cfg_type: str
    schema_version: str


class ParameterSnapshot(BaseModel):
    """Captured analysis environment, independent of live files and write targets.

    point is resource-owned component/general/opaque metadata content. library
    retains persistent Specs, not materialized cfg. aliases preserves authored
    row order and softlinks; None denotes fixed names without authored rows.
    names contains resolved scalar/list bindings from capture time. All values
    use working units. Sections stay opaque until their resource/session owner
    consumes them. Unknown envelope fields are ignored; missing sections remain
    missing, not an empty environment. No target or timestamp is synthesized.
    """

    point: ParameterContent = Field(default_factory=lambda: ParameterContent(None))
    library: LibraryContent = Field(default_factory=LibraryContent)
    aliases: YamlValue = None
    names: YamlValue = None


@dataclass(frozen=True, kw_only=True)
class RunSnapshot:
    """Historical entry state, never a live entry or hardware reference.

    entry_id identifies the entry; point is the optional work-point label.
    roles maps aliases to scalar names
    or ordered name lists. Lists preserve acquisition order, not tone indexes.
    Writers never refresh these values from a live entry. Python and JSON
    serialization copy roles into a detached dict, including read-only mappings.
    """

    entry_id: UUID
    entry_name: str | None = None
    point: str | None
    roles: Annotated[
        Mapping[str, AliasValue],
        PlainSerializer(dict, return_type=dict[str, AliasValue]),
    ]

    def to_labber_snapshot(self) -> JsonObject:
        """Return historical point and roles for a portable Labber comment.

        Omit entry_id; native metadata retains the complete snapshot.
        Do not refresh aliases or resolve names from current entry state.
        """
        return TypeAdapter(JsonObject).validate_python(
            {"point": self.point, "roles": dict(self.roles)}, strict=True
        )


@dataclass(frozen=True, kw_only=True)
class SoftwareProvenance:
    """Historical software/host facts, with None for unavailable evidence.

    software_versions maps package names to versions. git_commit is a SHA or
    None; git_dirty is true, false or unknown. qick_version, soc_fingerprint and
    hostname identify the runtime when known. simulation_id is an optional public
    environment identifier, never replay inputs or model truth. No field captures
    live state; an empty version mapping means no package evidence was recorded.
    """

    software_versions: Annotated[
        Mapping[str, str],
        PlainSerializer(dict, return_type=dict[str, str], when_used="json"),
    ]
    git_commit: str | None
    git_dirty: bool | None
    qick_version: str | None
    soc_fingerprint: str | None
    hostname: str | None
    simulation_id: str | None = None


class RunMetadata(BaseModel):
    """Historical identity and run-boundary evidence stored in docs/metadata.

    run_id must fully match one of two ASCII grammars:
    experiment [0-9]{8}T[0-9]{6}Z-[a-z0-9]{6}, or workflow operation
    [0-9a-f]{32}-[0-9]{6}-[1-9][0-9]*. The latter keeps the complete workflow
    root, six-digit iteration and positive attempt without leading zeros.
    These grammars do not validate calendar semantics.
    experiment is the declaration's tag.
    started_at/finished_at are UTC ISO timestamps or None for absent evidence.
    completion is complete, partial, stopped or unknown, not a save status.
    snapshot is captured entry state; provenance is software/host evidence.
    device_snapshot maps connected device names to finite JSON settings; None
    means unrecorded and {} means no connected devices.
    The envelope is frozen; nested mappings remain mutable. Save revalidates
    even unchecked model_copy updates. Unknown doc fields are not projected into
    the live model; Container retains them for lossless rewrites.
    """

    model_config = ConfigDict(
        frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )

    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    experiment: str = Field(min_length=1)
    started_at: str | None
    finished_at: str | None
    completion: Literal["complete", "partial", "stopped", "unknown"]
    snapshot: RunSnapshot
    provenance: SoftwareProvenance
    parameter_snapshot: ParameterSnapshot | None = None
    device_snapshot: (
        Annotated[
            Mapping[str, JsonObject],
            PlainSerializer(dict, return_type=dict[str, JsonObject], when_used="json"),
        ]
        | None
    ) = None

    def project(self) -> RunMetadata:
        """Validate all fields, including unchecked updates, for whole consumers.

        Return detached typed historical evidence. Invalid fields raise ValueError
        naming /docs/metadata. Reading a deferred record uses the same contract.
        """
        try:
            return RunMetadata.model_validate(self)
        except ValidationError as error:
            raise ValueError(f"/docs/metadata: {error}") from error

    @field_validator("started_at", "finished_at")
    @classmethod
    def utc_timestamp(cls, value: str | None) -> str | None:
        """Reject non-ISO or non-UTC timestamps; keep absent evidence as None."""
        if value is not None:
            parsed = datetime.fromisoformat(value)
            if parsed.utcoffset() != timedelta(0):
                raise ValueError("timestamp must be UTC ISO")
        return value


class RunMetadataContent:
    """Detached historical metadata with validation at each requested field.

    document is an experiment-owned metadata mapping from an already-read file.
    Construction copies it without validating unused fields. Properties enforce
    RunMetadata's field rules and name /docs/metadata/<field> on failure. snapshot
    and provenance request complete typed sections; parameter_snapshot retains
    resource-owned opaque contents. Missing optional fields use schema defaults.
    project() explicitly validates the complete document for Save or whole reads.
    No live values, identity or empty environment substitute malformed evidence.
    """

    def __init__(self, document: Mapping[str, object]) -> None:
        self._document = deepcopy(dict(document))

    def project(self) -> RunMetadata:
        """Return a detached, fully validated RunMetadata or a path-qualified error."""
        try:
            return RunMetadata.model_validate(deepcopy(self._document))
        except ValidationError as error:
            raise ValueError(f"/docs/metadata: {error}") from error

    def _field[T](self, name: str, adapter: TypeAdapter[T]) -> T:
        schema = model_field_schema(RunMetadata, name)
        try:
            value = SchemaValidator(schema).validate_python(
                deepcopy(self._document.get(name, PydanticUndefined))
            )
            return adapter.validate_python(value)
        except ValueError as error:
            raise ValueError(f"/docs/metadata/{name}: {error}") from error

    @property
    def run_id(self) -> str:
        """Recorded experiment/workflow operation ID, validated when requested."""
        return self._field("run_id", TypeAdapter(str))

    @property
    def experiment(self) -> str:
        """Nonempty experiment declaration tag used for load admission."""
        return self._field("experiment", TypeAdapter(str))

    @property
    def started_at(self) -> str | None:
        """Recorded UTC start timestamp, or explicit None for absent evidence."""
        return self._field("started_at", TypeAdapter(str | None))

    @property
    def finished_at(self) -> str | None:
        """Recorded UTC finish timestamp, or explicit None for absent evidence."""
        return self._field("finished_at", TypeAdapter(str | None))

    @property
    def completion(self) -> Literal["complete", "partial", "stopped", "unknown"]:
        """Acquisition completion, never inferred from a successful load/save."""
        return self._field(
            "completion",
            TypeAdapter(Literal["complete", "partial", "stopped", "unknown"]),
        )

    @property
    def snapshot(self) -> RunSnapshot:
        """Complete historical entry/point/roles projection, not a live binding."""
        return self._field("snapshot", TypeAdapter(RunSnapshot))

    @property
    def provenance(self) -> SoftwareProvenance:
        """Complete software/host evidence, validated only when requested."""
        return self._field("provenance", TypeAdapter(SoftwareProvenance))

    @property
    def parameter_snapshot(self) -> ParameterSnapshot | None:
        """Captured analysis content; only an absent/null snapshot returns None.

        Malformed envelope raises instead of degrading to an empty environment.
        Resource owners validate its point/library/aliases at actual use.
        """
        return self._field("parameter_snapshot", TypeAdapter(ParameterSnapshot | None))

    @property
    def device_snapshot(self) -> Mapping[str, JsonObject] | None:
        """Complete recorded device settings, or None when unrecorded."""
        return self._field(
            "device_snapshot", TypeAdapter(Mapping[str, JsonObject] | None)
        )


def validate_cfg_snapshot(cfg: CfgSnapshot) -> None:
    """Validate finite cfg JSON, nonempty cfg_type and ASCII major.minor.

    Unknown cfg keys are valid. Invalid envelopes raise ValueError naming /cfg;
    concrete cfg model validation belongs to the declaration's typed restore.
    """
    try:
        TypeAdapter(JsonObject).validate_json(
            json.dumps(cfg.values, allow_nan=False), strict=True
        )
    except (ValidationError, ValueError) as error:
        raise ValueError(f"/cfg: {error}") from error
    if re.fullmatch(r"[0-9]+\.[0-9]+", cfg.schema_version) is None:
        raise ValueError("/cfg: schema_version must be major.minor")
    if not cfg.cfg_type:
        raise ValueError("/cfg: cfg_type must not be empty")


def validate_workflow_run_id(run_id: str) -> None:
    """Validate a workflow root before admission or acquisition.

    run_id must fully match [0-9a-f]{32}, for example uuid4().hex. This root
    excludes the iteration and attempt suffix used by operation metadata.
    Return None without changing valid strings. Invalid strings raise ValueError
    naming the required format; nonstrings raise TypeError.
    """
    if re.fullmatch(_WORKFLOW_RUN_ID_PATTERN, run_id) is None:
        raise ValueError(
            "run_id must be 32 lowercase hexadecimal characters ([0-9a-f]{32})"
        )


def validate_run_id(run_id: str) -> None:
    """Accept exactly one of the experiment or workflow operation ID grammars.

    run_id must fully match [0-9]{8}T[0-9]{6}Z-[a-z0-9]{6} for an experiment,
    or [0-9a-f]{32}-[0-9]{6}-[1-9][0-9]* for a workflow operation. The workflow
    form preserves its lowercase hex root, six-digit iteration and positive
    attempt without leading zeros. No calendar semantics are inferred.
    Return None for valid strings without changing them. Invalid strings raise
    ValueError naming run_id; nonstrings raise TypeError.
    """
    if re.fullmatch(_RUN_ID_PATTERN, run_id) is None:
        raise ValueError(
            "/: run_id must be an experiment timestamp with six-character identity "
            "or a workflow's 32-character lowercase hex root, six-digit iteration "
            "and positive attempt without leading zeros"
        )
