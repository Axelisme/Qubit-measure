"""Typed envelopes for native experiment data; arrays are already in disk units."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Literal
from uuid import UUID

from pydantic import PlainSerializer

from .container_models import JsonObject


@dataclass(frozen=True, kw_only=True)
class CfgSnapshot:
    """Complete JSON cfg plus its declaration identity.

    values includes unknown nested JSON keys; it is not a typed cfg projection.
    cfg_type is the declaring cfg class name, not an import path.
    schema_version is a major.minor string maintained by that declaration.
    """

    values: JsonObject
    cfg_type: str
    schema_version: str


@dataclass(frozen=True, kw_only=True)
class RunSnapshot:
    """Historical entry state without live handles or a container reference.

    entry_id is the entry UUID. entry_name is its name at context capture.
    point and description are optional work-point label and description.
    roles maps role names to component names. No writer recaptures values from a
    live entry. The producer owns validation of the historical context.
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
    """Historical software and host evidence; never a live capture service.

    software_versions maps package names to versions (empty means no evidence).
    git_commit is a SHA or None; git_dirty is True, False or unknown None.
    qick_version, soc_fingerprint and hostname are nonempty strings or None.
    None represents genuinely unavailable evidence, including legacy files,
    non-Git installations or opaque cached-config handles, not capture failures.
    simulation_id is an optional public environment identifier, never replay
    inputs or model truth. Older files omit it and read as None. The original
    six disk attrs remain required. Optional strings use empty UTF-8 attrs for
    None, and git_dirty uses true/false/unknown, never bool(string).
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


@dataclass(frozen=True, kw_only=True)
class RunMetadata:
    """Run identity, progress and immutable historical input to native saving.

    run_id is the UTC timestamp plus six-character random identity.
    experiment is the declaring spec tag, never derived from a filename.
    started_at and finished_at are UTC ISO or None when historical evidence
    is absent. Normal acquisition always supplies started_at.
    completion is complete, partial, stopped or unknown measurement progress, not
    a transaction status. unknown means no historical completion evidence.
    labber_path is the last successful Labber path or None.
    snapshot is caller-captured entry state; provenance is historical environment
    evidence. device_snapshot maps device names to detached finite JSON settings
    captured for this Run. Empty means no devices; None means not recorded.
    Native stores it at /device, separate from /context, and never reads hardware.
    Native I/O rejects malformed values with a located ValueError.
    """

    run_id: str
    experiment: str
    started_at: str | None
    finished_at: str | None
    completion: Literal["complete", "partial", "stopped", "unknown"]
    labber_path: str | None
    snapshot: RunSnapshot
    provenance: SoftwareProvenance
    device_snapshot: (
        Annotated[
            Mapping[str, JsonObject],
            PlainSerializer(dict, return_type=dict[str, JsonObject], when_used="json"),
        ]
        | None
    ) = None
