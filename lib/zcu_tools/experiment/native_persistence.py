"""Native typed-result persistence, with the caller supplying an explicit spec."""

import warnings
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
from pydantic import TypeAdapter, ValidationError

from zcu_tools.datafile import (
    CfgSnapshot,
    ExperimentPayload,
    JsonObject,
    LabberMetadata,
    RunMetadata,
    RunSnapshot,
    StoredRun,
    VariableSchema,
    load_run_data,
    save_run_data,
    validate_cfg_snapshot,
    validate_experiment_payload,
)

from .axes_spec import (
    AxesSpec,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from .cfg_model import ExpCfgModel
from .records import RunRecord

CfgT = TypeVar("CfgT", bound=ExpCfgModel)
ResultT = TypeVar("ResultT")
RecordT = TypeVar("RecordT", covariant=True)


class CfgUnavailableWarning(UserWarning):
    """Valid run data was restored, but its historical cfg cannot be typed."""


@dataclass(frozen=True)
class RestoredRun(Generic[RecordT]):
    """A restored result and an explicit historical configuration diagnostic.

    record is the core's typed RunRecord or an adapter-owned opaque result.
    cfg_issue is None when typed cfg is available, otherwise a nonempty readable
    reason why it is unavailable. It never substitutes current or default cfg.
    """

    record: RecordT
    cfg_issue: str | None


@dataclass(frozen=True)
class PreparedRunData:
    """One typed record mapped for the native and Labber writers.

    payload contains SI/discrete arrays, ordered variable identities and the
    original user comment; representation preserves single/grouped semantics.
    cfg contains the same record's complete JSON values and declaration identity.
    Construction isolates cfg JSON, variable mappings and metadata containers,
    without copying large arrays. Callers must not modify arrays during saving.
    Format-specific constraints and I/O remain the respective writer's concern.
    """

    payload: ExperimentPayload
    cfg: CfgSnapshot

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "payload",
            replace(
                self.payload,
                variables=dict(self.payload.variables),
                metadata=replace(
                    self.payload.metadata, tags=list(self.payload.metadata.tags)
                ),
            ),
        )
        object.__setattr__(
            self, "cfg", replace(self.cfg, values=deepcopy(self.cfg.values))
        )


def prepare_run_data(
    source: RunRecord[CfgT, ResultT],
    *,
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
    comment: str,
) -> PreparedRunData:
    """Prepare disk-unit payload and JSON cfg from one typed record, without I/O.

    source.cfg must be present and match spec.cfg_type; result must match
    spec.result_type. Validate declared variables, shapes, units and an optional
    grouped result validator. Preserve single/grouped representation, including
    a one-member group, and retain comment as raw text for the writer's codec.
    Invalid cfg/result/comment raises TypeError or ValueError; missing Result
    fields raise AttributeError. Do not require Labber-compatible common grids
    or axes, choose paths, capture metadata or mutate the source record.
    """
    if source.cfg is None:
        raise ValueError("RunRecord.cfg is None; cannot save without configuration")
    if not isinstance(source.cfg, spec.cfg_type):
        raise TypeError(f"cfg must be {spec.cfg_type.__name__}")
    if not isinstance(source.result, spec.result_type):
        raise TypeError(f"Result must be {spec.result_type.__name__}")
    comment = TypeAdapter(str).validate_python(comment, strict=True)
    variables = _variable_specs(spec)
    if isinstance(spec, GroupedAxesSpec) and spec.result_validator is not None:
        spec.result_validator(source.result)
    payload = ExperimentPayload(
        variables={
            variable.data_variable: variable.payload_from_result(
                source.result, context=f"{spec.tag}: /data/{variable.data_variable}"
            )
            for variable in variables
        },
        metadata=LabberMetadata(comment=comment, tags=spec.tag),
        representation="grouped" if isinstance(spec, GroupedAxesSpec) else "single",
    )
    validate_experiment_payload(payload, schema=native_schemas(spec))
    cfg = CfgSnapshot(
        values=TypeAdapter(JsonObject).validate_python(
            source.cfg.model_dump(mode="json")
        ),
        cfg_type=spec.cfg_type.__name__,
        schema_version=spec.cfg_schema_version,
    )
    return PreparedRunData(payload=payload, cfg=cfg)


def native_schemas(
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
) -> tuple[VariableSchema, ...]:
    """Project an explicit experiment declaration to generic disk-unit schemas.

    spec is the same single or grouped declaration accepted by save_run/load_run.
    Return one schema per declared variable, preserving axes, names, units,
    dtypes and required identities. Do not inspect files, cfg or live hardware.
    Invalid declared units/dtypes propagate datafile schema validation errors.
    """
    return tuple(variable.native_schema() for variable in _variable_specs(spec))


def save_run(
    source: RunRecord[CfgT, ResultT],
    destination: Path,
    *,
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
    metadata: RunMetadata,
    replace: bool = False,
) -> None:
    """Save a typed record to native data.h5 using the explicit mapping spec.

    source.cfg must be non-None and match spec.cfg_type. Map arrays to declared
    SI/discrete units and preserve spec variable identities/representation.
    metadata is caller-captured historical evidence; its experiment must match
    spec.tag. cfg_type and schema_version come from the supplied declaration.
    Validate through datafile; do not inspect HDF5 layout here. Existing files
    raise FileExistsError unless replace=True, using atomic single-file replace.
    Invalid cfg/Result/spec/metadata raises ValueError or TypeError; I/O errors
    propagate. No live context capture, filename inference or legacy fallback.
    """
    prepared = prepare_run_data(source, spec=spec, comment="")
    if metadata.experiment != spec.tag:
        raise ValueError(
            f"{destination}: /: experiment {metadata.experiment!r} does not match {spec.tag!r}"
        )
    save_run_data(
        destination, prepared.payload, metadata, cfg=prepared.cfg, replace=replace
    )


def load_run(
    source: Path,
    *,
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
) -> tuple[RunRecord[CfgT, ResultT], RunSnapshot]:
    """Load one native record and its caller-captured historical snapshot.

    source is the exact native file; spec declares its typed Result/cfg mapping.
    Delegate to load_run_with_metadata and return only its snapshot for the
    existing tuple contract. The same validation and I/O errors propagate;
    no live context is captured and this instance-free entry mutates no caller.
    """
    record, metadata = load_run_with_metadata(source, spec=spec)
    return record, metadata.snapshot


def load_run_with_metadata(
    source: Path,
    *,
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
) -> tuple[RunRecord[CfgT, ResultT], RunMetadata]:
    """Load typed data and all historical run metadata using an explicit spec.

    source is an exact native file. Metadata retains its run identity, UTC
    times, completion, Labber path, snapshot and provenance without recapture.
    Read through datafile once without capturing a full file image, then use
    restore_run for typed conversion. Incompatible cfg type/schema or concrete
    validation returns cfg=None and emits CfgUnavailableWarning. Valid arrays
    remain available. Same-major minor and unknown cfg keys use known projection.
    Wrong tag, missing variables or malformed native data/cfg envelope raises a
    located ValueError (format/version retain public subtypes). I/O errors
    propagate. No registry lookup, dynamic imports or Labber/legacy fallback.
    This typed tuple is not a lossless unknown-field rewrite envelope; callers
    needing that promise use datafile.load_run_data(preserve_unknown=True).
    """
    stored = load_run_data(source)
    try:
        restored = restore_run(stored, spec=spec)
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error
    return restored.record, stored.metadata


def restore_run(
    stored: StoredRun,
    *,
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
) -> RestoredRun[RunRecord[CfgT, ResultT]]:
    """Convert already-read native data to a typed record without I/O.

    stored supplies generic SI arrays and historical cfg/metadata; spec explicitly
    declares the experiment tag, required variables, memory units and cfg model.
    Validate tag and data before interpreting typed cfg. Missing variables, bad
    axes/units/dtype/shape, invalid cfg envelope or rejected builder results raise
    ValueError or TypeError. Extra variables are projected out. No registry or
    dynamic imports are used, and raw cfg/metadata are not modified.

    A valid cfg envelope with an incompatible cfg_type, schema major or concrete
    model returns record.cfg=None and a readable cfg_issue, also emitted as one
    CfgUnavailableWarning after successful reconstruction. Same-major minor and
    unknown cfg keys use typed known projection; validators see isolated raw
    values. This result is not a lossless rewrite envelope.
    """
    if stored.metadata.experiment != spec.tag:
        raise ValueError(
            f"/: experiment {stored.metadata.experiment!r} does not match {spec.tag!r}"
        )
    variables = _variable_specs(spec)
    missing = {
        variable.data_variable for variable in variables
    } - stored.payload.variables.keys()
    if missing:
        raise ValueError(f"/data: missing variables {sorted(missing)}")
    payload = ExperimentPayload(
        variables={
            variable.data_variable: stored.payload.variables[variable.data_variable]
            for variable in variables
        },
        metadata=stored.payload.metadata,
        representation="grouped" if isinstance(spec, GroupedAxesSpec) else "single",
    )
    validate_experiment_payload(
        payload, schema=tuple(variable.native_schema() for variable in variables)
    )
    loaded = {
        variable.data_variable: variable.loaded_from_payload(
            payload.variables[variable.data_variable],
            context=f"/data/{variable.data_variable}",
        )
        for variable in variables
    }
    validate_cfg_snapshot(stored.cfg)
    cfg, cfg_issue = _restore_cfg(stored.cfg, spec=spec)
    if isinstance(spec, GroupedAxesSpec):
        result = spec.result_builder(
            GroupedLoadData(
                variables=loaded, metadata=payload.metadata, cfg_snapshot=cfg
            )
        )
        if not isinstance(result, spec.result_type):
            raise TypeError(
                f"GroupedAxesSpec builder must return {spec.result_type.__name__}"
            )
        if spec.result_validator is not None:
            spec.result_validator(result)
    else:
        variable = loaded[spec.data_variable]
        fields: dict[str, np.ndarray] = {
            axis.field_name: values
            for axis, values in zip(spec.axes, variable.axes, strict=True)
        }
        fields[spec.z.field_name] = variable.z
        result = spec.result_type(**fields)
    if cfg_issue is not None:
        warnings.warn(cfg_issue, CfgUnavailableWarning, stacklevel=2)
    return RestoredRun(record=RunRecord(cfg=cfg, result=result), cfg_issue=cfg_issue)


def _restore_cfg(
    cfg: CfgSnapshot,
    *,
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
) -> tuple[CfgT | None, str | None]:
    if cfg.cfg_type != spec.cfg_type.__name__:
        return None, (
            f"/cfg: cfg_type {cfg.cfg_type!r} does not match {spec.cfg_type.__name__!r}"
        )
    if int(cfg.schema_version.split(".")[0]) != int(
        spec.cfg_schema_version.split(".")[0]
    ):
        return None, (
            f"/cfg: schema major {cfg.schema_version!r} does not match "
            f"{spec.cfg_schema_version!r} for {cfg.cfg_type}"
        )
    try:
        return spec.cfg_type.model_validate(deepcopy(cfg.values), extra="ignore"), None
    except ValidationError as error:
        return None, f"/cfg: {cfg.cfg_type} validation failed: {error}"


def _variable_specs(
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
) -> tuple[VariableSpec, ...]:
    if isinstance(spec, GroupedAxesSpec):
        return spec.variables
    # Reuse the same unit/dtype mapping for single and grouped native variables.
    return (
        VariableSpec(
            variable=spec.data_variable,
            axes=tuple(
                VariableAxisSpec(
                    field_name=axis.field_name,
                    label=axis.label,
                    unit=axis.unit,
                    scale=axis.scale,
                    dtype=axis.dtype,
                )
                for axis in spec.axes
            ),
            z=VariableZSpec(
                field_name=spec.z.field_name,
                label=spec.z.label,
                unit=spec.z.unit,
                scale=spec.z.scale,
                dtype=spec.z.dtype,
            ),
        ),
    )
