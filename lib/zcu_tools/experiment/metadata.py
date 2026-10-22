"""Experiment-owned historical run documents and configuration envelopes."""

import json
import re
from collections.abc import Mapping
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

from zcu_tools.datafile import JsonObject, LabberMetadata


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


@dataclass(frozen=True, kw_only=True)
class RunSnapshot:
    """Historical entry state, never a live entry or hardware reference.

    entry_id identifies the entry; entry_name is its name at capture. point and
    description are optional work-point labels. roles maps role to component name.
    Writers never refresh these values from a live entry.
    """

    entry_id: UUID
    entry_name: str
    point: str | None
    description: str | None
    roles: Annotated[
        Mapping[str, str],
        PlainSerializer(dict, return_type=dict[str, str], when_used="json"),
    ]


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
    """Historical identity and acquisition facts stored in docs/metadata.

    run_id is a UTC timestamp plus six-character identity, optionally followed
    by a workflow's six-digit commit sequence and numeric attempt.
    experiment is the declaration's tag.
    started_at/finished_at are UTC ISO timestamps or None for absent evidence.
    completion is complete, partial, stopped or unknown, not a save status.
    snapshot is captured entry state; provenance is software/host evidence.
    device_snapshot maps connected device names to finite JSON settings; None
    means unrecorded and {} means no connected devices. labber optionally retains
    loaded comment/tags/project/user/creation_time, never a saved file path.
    The envelope is frozen; nested mappings remain mutable. Save revalidates
    even unchecked model_copy updates. Unknown doc fields are not projected into
    the live model; Container retains them for lossless rewrites.
    """

    model_config = ConfigDict(
        frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )

    run_id: str = Field(
        pattern=r"^[0-9]{8}T[0-9]{6}Z-[a-z0-9]{6}(?:-[0-9]{6}-[0-9]+)?$"
    )
    experiment: str = Field(min_length=1)
    started_at: str | None
    finished_at: str | None
    completion: Literal["complete", "partial", "stopped", "unknown"]
    snapshot: RunSnapshot
    provenance: SoftwareProvenance
    device_snapshot: (
        Annotated[
            Mapping[str, JsonObject],
            PlainSerializer(dict, return_type=dict[str, JsonObject], when_used="json"),
        ]
        | None
    ) = None
    labber: LabberMetadata | None = None

    @field_validator("started_at", "finished_at")
    @classmethod
    def utc_timestamp(cls, value: str | None) -> str | None:
        """Reject non-ISO or non-UTC timestamps; keep absent evidence as None."""
        if value is not None:
            parsed = datetime.fromisoformat(value)
            if parsed.utcoffset() != timedelta(0):
                raise ValueError("timestamp must be UTC ISO")
        return value


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


def validate_run_id(run_id: str) -> None:
    """Validate ordinary acquisition YYYYMMDDTHHMMSSZ-six_alphanumerics tokens.

    Invalid strings raise ValueError naming run_id; nonstrings raise TypeError.
    This acquisition-token check does not constrain workflow identities
    or infer calendar semantics or acquisition evidence from the token.
    """
    if re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[a-z0-9]{6}", run_id) is None:
        raise ValueError(
            "/: run_id must be a UTC timestamp plus six-character identity"
        )
