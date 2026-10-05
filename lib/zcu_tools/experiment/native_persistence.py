"""Native typed-result persistence, with the caller supplying an explicit spec."""

from pathlib import Path
from typing import TypeVar

import numpy as np
from pydantic import TypeAdapter

from zcu_tools.datafile import (
    CfgSnapshot,
    ExperimentPayload,
    JsonObject,
    LabberMetadata,
    RunMetadata,
    RunSnapshot,
    load_run_data,
    save_run_data,
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
    metadata is start-of-run historical evidence; its experiment must match
    spec.tag. cfg_type and schema_version come from the supplied declaration.
    Validate through datafile; do not inspect HDF5 layout here. Existing files
    raise FileExistsError unless replace=True, using atomic single-file replace.
    Invalid cfg/Result/spec/metadata raises ValueError or TypeError; I/O errors
    propagate. No live context capture, filename inference or legacy fallback.
    """
    if source.cfg is None:
        raise ValueError("RunRecord.cfg is None; cannot save without configuration")
    if not isinstance(source.cfg, spec.cfg_type):
        raise TypeError(f"cfg must be {spec.cfg_type.__name__}")
    if not isinstance(source.result, spec.result_type):
        raise TypeError(f"Result must be {spec.result_type.__name__}")
    if metadata.experiment != spec.tag:
        raise ValueError(
            f"{destination}: /: experiment {metadata.experiment!r} does not match {spec.tag!r}"
        )
    variables = _variable_specs(spec)
    if isinstance(spec, GroupedAxesSpec) and spec.result_validator is not None:
        spec.result_validator(source.result)
    payload = ExperimentPayload(
        variables={
            variable.data_variable: variable.payload_from_result(
                source.result, context=f"{destination}: /data/{variable.data_variable}"
            )
            for variable in variables
        },
        metadata=LabberMetadata(tags=spec.tag),
        representation="grouped" if isinstance(spec, GroupedAxesSpec) else "single",
    )
    validate_experiment_payload(
        payload, schema=tuple(variable.native_schema() for variable in variables)
    )
    cfg = CfgSnapshot(
        values=TypeAdapter(JsonObject).validate_python(
            source.cfg.model_dump(mode="json")
        ),
        cfg_type=spec.cfg_type.__name__,
        schema_version=spec.cfg_schema_version,
    )
    save_run_data(destination, payload, metadata, cfg=cfg, replace=replace)


def load_run(
    source: Path,
    *,
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
) -> tuple[RunRecord[CfgT, ResultT], RunSnapshot]:
    """Load one native record and its start-of-run snapshot using an explicit spec.

    Read through datafile once without capturing a full file image. Match tag,
    cfg_type and cfg schema major; permit same-major minor. Select declared
    variables, validate their schema and reconstruct Result in memory units.
    Validate cfg with the supplied cfg model and extra="ignore" for typed known
    projection, including nested models; before validators still see raw input.
    Invalid cfg propagates its validation error, not a nullable cfg fallback.
    Wrong tag/schema, missing variables or malformed native data raises a
    located ValueError (format/version retain public subtypes). I/O errors
    propagate. No registry lookup, dynamic imports or Labber/legacy fallback.
    This typed tuple is not a lossless unknown-field rewrite envelope; callers
    needing that promise use datafile.load_run_data(preserve_unknown=True).
    """
    stored = load_run_data(source)
    if stored.metadata.experiment != spec.tag:
        raise ValueError(
            f"{source}: /: experiment {stored.metadata.experiment!r} does not match {spec.tag!r}"
        )
    if stored.cfg.cfg_type != spec.cfg_type.__name__:
        raise ValueError(f"{source}: /cfg: cfg_type must be {spec.cfg_type.__name__}")
    if int(stored.cfg.schema_version.split(".")[0]) != int(
        spec.cfg_schema_version.split(".")[0]
    ):
        raise ValueError(
            f"{source}: /cfg: cfg_schema_version major does not match {spec.cfg_schema_version}"
        )
    variables = _variable_specs(spec)
    missing = {
        variable.data_variable for variable in variables
    } - stored.payload.variables.keys()
    if missing:
        raise ValueError(f"{source}: /data: missing variables {sorted(missing)}")
    payload = ExperimentPayload(
        variables={
            variable.data_variable: stored.payload.variables[variable.data_variable]
            for variable in variables
        },
        metadata=stored.payload.metadata,
        representation="grouped" if isinstance(spec, GroupedAxesSpec) else "single",
    )
    try:
        validate_experiment_payload(
            payload, schema=tuple(variable.native_schema() for variable in variables)
        )
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error
    cfg = spec.cfg_type.model_validate(stored.cfg.values, extra="ignore")
    loaded = {
        variable.data_variable: variable.loaded_from_payload(
            payload.variables[variable.data_variable],
            context=f"{source}: /data/{variable.data_variable}",
        )
        for variable in variables
    }
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
    return RunRecord(cfg=cfg, result=result), stored.metadata.snapshot


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
