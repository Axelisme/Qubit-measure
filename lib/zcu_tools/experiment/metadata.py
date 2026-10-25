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

from zcu_tools.datafile import JsonObject
from zcu_tools.experiment.acquisition_facts import AcquisitionFacts
from zcu_tools.resources.entry.aliases import AliasValue

_WORKFLOW_RUN_ID_PATTERN = r"[0-9a-f]{32}"
_RUN_ID_PATTERN = (
    rf"^(?:[0-9]{{8}}T[0-9]{{6}}Z-[a-z0-9]{{6}}|"
    rf"{_WORKFLOW_RUN_ID_PATTERN}(?:-[0-9]{{6}}-[1-9][0-9]*)?)$"
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
    point: str | None
    roles: Annotated[
        Mapping[str, AliasValue],
        PlainSerializer(dict, return_type=dict[str, AliasValue]),
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

    run_id must fully match one of three ASCII grammars:
    experiment [0-9]{8}T[0-9]{6}Z-[a-z0-9]{6}, workflow root [0-9a-f]{32},
    or workflow operation [0-9a-f]{32}-[0-9]{6}-[1-9][0-9]*. The operation
    keeps the complete root, six-digit iteration and positive attempt without
    leading zeros. A merged workflow output uses the root unchanged, with
    timestamps and completion from the manifest, not a single acquisition.
    These grammars do not validate calendar semantics.
    experiment is the declaration's tag or merged output's workflow name.
    started_at/finished_at are UTC ISO timestamps or None for absent evidence.
    completion is complete, partial, stopped or unknown, not a save status.
    snapshot is captured entry state; provenance is software/host evidence.
    device_snapshot maps connected device names to finite JSON settings; None
    means unrecorded and {} means no connected devices.
    acquisitions contains ordered program acquisition observations, including
    retries and data-driven stopping. Historical documents without it read as
    an empty tuple; no rounds are inferred from configuration.
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
    device_snapshot: (
        Annotated[
            Mapping[str, JsonObject],
            PlainSerializer(dict, return_type=dict[str, JsonObject], when_used="json"),
        ]
        | None
    ) = None
    acquisitions: tuple[AcquisitionFacts, ...] = ()

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
    """Accept an experiment, workflow root or workflow operation ID.

    run_id must fully match [0-9]{8}T[0-9]{6}Z-[a-z0-9]{6} for an experiment,
    [0-9a-f]{32} for a workflow root used by merged outputs, or
    [0-9a-f]{32}-[0-9]{6}-[1-9][0-9]* for a workflow operation. The operation
    preserves its lowercase hex root, six-digit iteration and positive attempt
    without leading zeros. No calendar semantics are inferred.
    Return None for valid strings without changing them. Invalid strings raise
    ValueError naming run_id; nonstrings raise TypeError.
    """
    if re.fullmatch(_RUN_ID_PATTERN, run_id) is None:
        raise ValueError(
            "/: run_id must be an experiment timestamp with six-character identity "
            "or a workflow's 32-character lowercase hex root, optionally followed "
            "by a six-digit iteration and positive attempt without leading zeros"
        )
