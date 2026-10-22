"""Typed Result/metadata mapping over the standalone numeric/JSON Container."""

import warnings
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Generic, Literal, TypeVar

import numpy as np
from pydantic import TypeAdapter, ValidationError

from zcu_tools.datafile import (
    Container,
    ContainerAttrs,
    JsonObject,
    LabberMetadata,
    load,
    save,
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
from .metadata import CfgSnapshot, RunMetadata, RunSnapshot, validate_cfg_snapshot
from .records import RunRecord

CfgT = TypeVar("CfgT", bound=ExpCfgModel)
ResultT = TypeVar("ResultT")
RecordT = TypeVar("RecordT", covariant=True)
_METADATA = TypeAdapter(RunMetadata)
_CFG = TypeAdapter(CfgSnapshot)
_JSON = TypeAdapter(JsonObject)


class CfgUnavailableWarning(UserWarning):
    """Valid run data was restored, but its historical cfg cannot be typed."""


@dataclass(frozen=True)
class RestoredRun(Generic[RecordT]):
    """Typed record and its optional historical configuration diagnostic.

    record contains restored arrays and metadata. cfg_issue is None for valid cfg,
    otherwise the readable reason why cfg is unavailable. No defaults substitute
    for unavailable historical cfg.
    """

    record: RecordT
    cfg_issue: str | None


@dataclass(frozen=True)
class PreparedRunData:
    """One record's detached cfg, numeric container and Labber export options.

    container holds SI arrays with NumPy-ordered axes. cfg retains complete JSON
    values and declaration identity. labber_metadata carries raw comment and tags,
    not a cfg envelope. labber_layout is common or independent, from the spec.
    Construction copies mappings and cfg/metadata but shares large arrays.
    Callers must not mutate arrays during saving. Preparation performs no I/O.
    """

    container: Container
    cfg: CfgSnapshot
    labber_metadata: LabberMetadata
    labber_layout: Literal["common", "independent"] = "common"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "container",
            replace(
                self.container,
                vars=dict(self.container.vars),
                docs=deepcopy(self.container.docs),
                groups=dict(self.container.groups),
            ),
        )
        object.__setattr__(self, "cfg", deepcopy(self.cfg))
        object.__setattr__(self, "labber_metadata", deepcopy(self.labber_metadata))

    def to_container(self, metadata: RunMetadata) -> Container:
        """Return the prepared arrays with validated docs/cfg and docs/metadata.

        metadata is the saving record's historical evidence. Revalidate even
        unchecked model_copy updates. Invalid cfg or metadata raises ValueError;
        no live state is captured, paths chosen or files written. Unknown prepared
        docs/groups and container attributes remain intact.
        """
        validate_cfg_snapshot(self.cfg)
        try:
            metadata = _METADATA.validate_python(metadata)
        except ValidationError as error:
            raise ValueError(f"/docs/metadata: {error}") from error
        docs = dict(self.container.docs)
        docs["cfg"] = _JSON.validate_python(_CFG.dump_python(self.cfg, mode="json"))
        docs["metadata"] = _JSON.validate_python(metadata.model_dump(mode="json"))
        return replace(self.container, docs=docs)


def prepare_run_data(
    source: RunRecord[CfgT, ResultT],
    *,
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
    comment: str,
) -> PreparedRunData:
    """Prepare SI variables/cfg and raw Labber comment from one typed record.

    cfg must be present and match spec.cfg_type; result must match result_type.
    Validate shapes/dtypes and the grouped result validator. Preserve the spec's
    Labber layout without adding common-grid restrictions. Metadata may be absent
    during preparation; native saving requires it. Bad inputs raise TypeError,
    ValueError or AttributeError. No I/O, live capture or source mutation occurs.
    """
    if source.cfg is None:
        raise ValueError("RunRecord.cfg is None; cannot save without configuration")
    if not isinstance(source.cfg, spec.cfg_type):
        raise TypeError(f"cfg must be {spec.cfg_type.__name__}")
    if not isinstance(source.result, spec.result_type):
        raise TypeError(f"Result must be {spec.result_type.__name__}")
    comment = TypeAdapter(str).validate_python(comment, strict=True)
    if isinstance(spec, GroupedAxesSpec) and spec.result_validator is not None:
        spec.result_validator(source.result)
    container = Container(
        vars={
            str(variable.data_variable): variable.variable_from_result(
                source.result, context=f"{spec.tag}: /vars/{variable.data_variable}"
            )
            for variable in _variable_specs(spec)
        },
        docs={},
        groups={},
        attrs=ContainerAttrs("3.0", "zcu_tools.experiment", "experiment-run", {}),
    )
    cfg = CfgSnapshot(
        values=_JSON.validate_python(source.cfg.model_dump(mode="json")),
        cfg_type=spec.cfg_type.__name__,
        schema_version=spec.cfg_schema_version,
    )
    return PreparedRunData(
        container,
        cfg,
        LabberMetadata(comment=comment, tags=spec.tag),
        spec.labber_layout if isinstance(spec, GroupedAxesSpec) else "common",
    )


def save_run(
    source: RunRecord[CfgT, ResultT],
    destination: Path,
    *,
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
    replace: bool = False,
) -> None:
    """Save a typed record with its own metadata at the exact native path.

    source.cfg and metadata must be present; metadata.experiment must match tag.
    Map arrays to declared SI/discrete units and serialize cfg/metadata as JSON
    docs. Invalid inputs raise ValueError/TypeError before I/O. Existing files
    raise FileExistsError unless replace=True. Atomic single-file publication
    and I/O errors belong to datafile. No live state or legacy format is guessed.
    """
    if source.metadata is None:
        raise ValueError("RunRecord.metadata is None; cannot save without metadata")
    if source.metadata.experiment != spec.tag:
        raise ValueError(
            f"{destination}: experiment {source.metadata.experiment!r} does not match {spec.tag!r}"
        )
    prepared = prepare_run_data(source, spec=spec, comment="")
    save(destination, prepared.to_container(source.metadata), replace=replace)


def load_run(
    source: Path,
    *,
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
) -> tuple[RunRecord[CfgT, ResultT], RunSnapshot]:
    """Load a record and its historical snapshot using an explicit declaration.

    This tuple retains the snapshot convenience contract. Both items come from
    the same validated metadata document; invalid data/metadata/cfg raises a
    source-located ValueError. I/O errors propagate. Incompatible typed cfg emits
    CfgUnavailableWarning and leaves valid arrays/metadata available with cfg=None.
    """
    record = load_run_with_metadata(source, spec=spec)
    if record.metadata is None:
        raise ValueError(f"{source}: missing metadata")
    return record, record.metadata.snapshot


def load_run_with_metadata(
    source: Path,
    *,
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
) -> RunRecord[CfgT, ResultT]:
    """Read a Container once and return a typed record carrying all run metadata.

    source is the exact path, not a filename-derived identity. Container format
    and document/data validation errors name source; I/O errors propagate.
    Incompatible typed cfg warns and yields cfg=None after successful Result
    reconstruction. No live capture, registry lookup or Labber fallback occurs.
    The record is not a lossless rewrite envelope; retain Container for that use.
    """
    container = load(source)
    try:
        return restore_run(container, spec=spec).record
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error


def restore_run(
    stored: Container,
    *,
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
) -> RestoredRun[RunRecord[CfgT, ResultT]]:
    """Restore typed arrays/cfg/metadata from an already-read Container, without I/O.

    Validate metadata/tag, required variables, units/axes/dtypes/shapes and cfg
    envelope. Reconstruct and validate Result before concrete cfg validation;
    grouped builders receive cfg_snapshot=None. Extra variables/unknown cfg keys
    remain in the Container, not the typed record. Malformed docs or data raise
    ValueError/TypeError. An incompatible cfg type/schema/model yields cfg=None,
    cfg_issue and one CfgUnavailableWarning, retaining data and record.metadata.
    Inputs are not changed and no current settings substitute historical values.
    """
    try:
        metadata = _METADATA.validate_python(stored.docs["metadata"])
    except (KeyError, ValidationError) as error:
        raise ValueError(f"/docs/metadata: {error}") from error
    if metadata.experiment != spec.tag:
        raise ValueError(
            f"/docs/metadata: experiment {metadata.experiment!r} does not match {spec.tag!r}"
        )
    variables = _variable_specs(spec)
    missing = {
        str(variable.data_variable) for variable in variables
    } - stored.vars.keys()
    if missing:
        raise ValueError(f"/vars: missing variables {sorted(missing)}")
    loaded = {
        variable.data_variable: variable.loaded_from_variable(
            stored.vars[str(variable.data_variable)],
            context=f"/vars/{variable.data_variable}",
        )
        for variable in variables
    }
    try:
        cfg_snapshot = _CFG.validate_python(stored.docs["cfg"])
    except (KeyError, ValidationError) as error:
        raise ValueError(f"/docs/cfg: {error}") from error
    validate_cfg_snapshot(cfg_snapshot)
    if isinstance(spec, GroupedAxesSpec):
        result = spec.result_builder(
            GroupedLoadData(
                variables=loaded,
                metadata=metadata.labber or LabberMetadata(),
                cfg_snapshot=None,
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
    cfg, cfg_issue = _restore_cfg(cfg_snapshot, spec=spec)
    if cfg_issue is not None:
        warnings.warn(cfg_issue, CfgUnavailableWarning, stacklevel=2)
    return RestoredRun(RunRecord(cfg=cfg, result=result, metadata=metadata), cfg_issue)


def _restore_cfg(
    cfg: CfgSnapshot,
    *,
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
) -> tuple[CfgT | None, str | None]:
    if cfg.cfg_type != spec.cfg_type.__name__:
        return (
            None,
            f"/cfg: cfg_type {cfg.cfg_type!r} does not match {spec.cfg_type.__name__!r}",
        )
    if int(cfg.schema_version.split(".")[0]) != int(
        spec.cfg_schema_version.split(".")[0]
    ):
        return (
            None,
            f"/cfg: schema major {cfg.schema_version!r} does not match {spec.cfg_schema_version!r} for {cfg.cfg_type}",
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
