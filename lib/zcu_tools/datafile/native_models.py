"""Typed envelopes for native experiment data; arrays are already in disk units."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

import numpy as np
from pydantic import JsonValue

from .models import DataVariable, LabberMetadata, LabberPayload

type JsonObject = dict[str, JsonValue]


@dataclass(frozen=True, kw_only=True)
class ExperimentPayload:
    """Ordered variables and shared Labber metadata for either writer.

    variables maps stable DataVariable identities to inner-first LabberPayloads
    in SI (or explicitly declared discrete units). It must be nonempty.
    metadata carries comments, tags, project, user and optional creation time.
    representation is single (exactly one variable) or grouped (one or more).
    Native I/O validates names, numeric arrays and reversed-axis shapes;
    different variables may use different grids. Freezing does not copy arrays.
    """

    variables: Mapping[DataVariable, LabberPayload]
    metadata: LabberMetadata
    representation: Literal["single", "grouped"]


@dataclass(frozen=True, kw_only=True)
class AxisSchema:
    """Expected disk axis name, units and numeric dtype.

    name is a nonempty HDF5 path segment; unit is the declared SI/discrete unit,
    including an explicit empty unit for dimensionless coordinates.
    dtype is the expected NumPy disk dtype, not a memory conversion rule.
    """

    name: str
    unit: str
    dtype: np.dtype[np.generic]


@dataclass(frozen=True, kw_only=True)
class VariableSchema:
    """Disk declaration for one stable variable identity.

    variable is its DataVariable group key. axes is ordered inner-first.
    signal_name is a nonempty dataset name distinct from axes and timestamps.
    signal_unit is the declared SI/discrete unit (empty is dimensionless).
    signal_dtype is the expected real or complex numeric NumPy disk dtype.
    Arrays must match reversed axis lengths. Timestamps use the fixed native
    per-inner-trace Unix epoch contract, not a per-experiment schema.
    """

    variable: DataVariable
    axes: tuple[AxisSchema, ...]
    signal_name: str
    signal_unit: str
    signal_dtype: np.dtype[np.generic]


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
class CloneOrigin:
    """Historical source entry UUID and optional work-point label."""

    entry_id: UUID
    point: str | None


@dataclass(frozen=True, kw_only=True)
class ParameterSource:
    """Audit origin of a parameter value in working units.

    source is manual or a local ledger event ID. kind and run_id are optional
    historical identifiers. at is a UTC ISO timestamp. stderr is an optional
    uncertainty in the parameter's working units. cloned_from identifies the
    original entry/work point for a clone, or None for a non-cloned value.
    Persistence does not resolve or invent ledger evidence.
    """

    source: str
    kind: str | None
    run_id: str | None
    at: str
    stderr: float | None
    cloned_from: CloneOrigin | None


@dataclass(frozen=True, kw_only=True)
class ParameterSnapshot:
    """JSON value, optional working-unit label and its audit source."""

    value: JsonValue
    unit: str | None
    source: ParameterSource


@dataclass(frozen=True, kw_only=True)
class RunSnapshot:
    """Start-of-run entry state without live handles or a container reference.

    entry_id is the entry UUID. entry_name is its name at run start.
    point and description are optional work-point label and description.
    roles maps role names to component names; params maps dotted parameter paths
    to historical ParameterSnapshots, in working units. No writer recaptures
    values from a live entry. Unknown context fields use NativeExtensions.
    """

    entry_id: UUID
    entry_name: str
    point: str | None
    description: str | None
    roles: Mapping[str, str]
    params: Mapping[str, ParameterSnapshot]


@dataclass(frozen=True, kw_only=True)
class SoftwareProvenance:
    """Historical software and host evidence; never a live capture service.

    software_versions maps package names to versions (empty means no evidence).
    git_commit is a SHA or None; git_dirty is True, False or unknown None.
    qick_version, soc_fingerprint and hostname are nonempty strings or None.
    None represents absent legacy evidence, not swallowed live capture failures.
    All six disk attrs are required. Optional strings use empty UTF-8 attrs for
    None, and git_dirty uses true/false/unknown, never bool(string).
    """

    software_versions: Mapping[str, str]
    git_commit: str | None
    git_dirty: bool | None
    qick_version: str | None
    soc_fingerprint: str | None
    hostname: str | None


@dataclass(frozen=True, kw_only=True)
class RunMetadata:
    """Run identity, progress and immutable historical input to native saving.

    run_id is the UTC timestamp plus six-character random identity.
    experiment is the declaring spec tag, never derived from a filename.
    started_at is UTC ISO; finished_at is UTC ISO or absent evidence None.
    completion is complete, partial or stopped measurement progress, not a
    transaction status. labber_path is the last successful Labber path or None.
    snapshot is start-of-run entry state; provenance is historical environment
    evidence. Native I/O rejects malformed values with a located ValueError.
    """

    run_id: str
    experiment: str
    started_at: str
    finished_at: str | None
    completion: Literal["complete", "partial", "stopped"]
    labber_path: str | None
    snapshot: RunSnapshot
    provenance: SoftwareProvenance


@dataclass(frozen=True, kw_only=True)
class NativeExtensions:
    """Complete validated HDF5 file image for explicit lossless minor rewrites.

    file_image owns detached bytes, not a source path or an open file handle.
    Only load_run_data(preserve_unknown=True) produces this envelope. Generic
    callers pass it unchanged to save_run_data to preserve unknown attrs, nodes,
    links, dtypes and JSON siblings. Shape-changing reference remapping is not
    promised. Ordinary typed experiment loading does not allocate this image.
    """

    file_image: bytes


@dataclass(frozen=True, kw_only=True)
class StoredRun:
    """Known native data plus an optional detached unknown-field envelope.

    payload is the validated generic SI/discrete variable mapping.
    metadata contains the known context projection and run/provenance evidence.
    cfg contains the complete JSON cfg, including unknown nested keys.
    extensions is None by default or NativeExtensions when explicitly requested.
    Source files are closed before this envelope is returned.
    """

    payload: ExperimentPayload
    metadata: RunMetadata
    cfg: CfgSnapshot
    extensions: NativeExtensions | None
