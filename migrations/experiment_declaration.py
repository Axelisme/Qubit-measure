"""Neutral typed declaration passed to the offline CLI composition root."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    GroupedAxesSpec,
    VariableAxisSpec,
    VariableZSpec,
    ZSpec,
    load_run,
)
from zcu_tools.experiment.metadata import CfgSnapshot, validate_cfg_snapshot

from migrations.legacy_labber import LegacyAxisSchema, LegacyVariableSchema


@dataclass(frozen=True)
class MigrationExperiment:
    """Bind a concrete typed reader to its generic schema without erasing its cfg.

    source_tag and cfg_type are the exact historical experiment/cfg identities;
    neither determines the other. native_tag is the canonical spec tag written
    into native RunMetadata; it may differ from source_tag only by an explicit
    migration declaration. schemas is the generic
    disk-unit variable declaration. validate_cfg checks a historical CfgSnapshot
    for storage format, this pair's cfg_type, major version and concrete model.
    It returns None without changing raw values and raises ValueError for
    unconvertible cfg.
    validate_native reads an exact native Path
    with the corresponding typed load_run, returns None and propagates failures.
    These bindings perform no acquisition, registration or live-context capture.
    """

    source_tag: str
    native_tag: str
    cfg_type: str
    schemas: tuple[LegacyVariableSchema, ...]
    validate_cfg: Callable[[CfgSnapshot], None]
    validate_native: Callable[[Path], None]


ResultT = TypeVar("ResultT")
CfgT = TypeVar("CfgT", bound=ConfigBase)


def _legacy_schema(
    variable: str, axes: Sequence[Axis | VariableAxisSpec], z: ZSpec | VariableZSpec
) -> LegacyVariableSchema:
    return LegacyVariableSchema(
        variable=variable,
        axes=tuple(
            LegacyAxisSchema(
                name=axis.label,
                unit=axis.unit,
                dtype=np.dtype(axis.dtype)
                if isinstance(axis, VariableAxisSpec) and axis.generated == "arange"
                else (np.empty(0, dtype=axis.dtype) * axis.scale).dtype,
            )
            for axis in axes
        ),
        signal_name=z.label,
        signal_unit=z.unit,
        signal_dtype=(np.empty(0, dtype=z.dtype) * z.scale).dtype,
    )


def _legacy_schemas(
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
) -> tuple[LegacyVariableSchema, ...]:
    if isinstance(spec, GroupedAxesSpec):
        return tuple(
            _legacy_schema(item.data_variable, item.axes, item.z)
            for item in spec.variables
        )
    return (_legacy_schema(spec.data_variable, spec.axes, spec.z),)


def declare_migration_experiment(
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT] | None,
    *,
    source_tag: str | None = None,
    schemas: tuple[LegacyVariableSchema, ...] | None = None,
) -> MigrationExperiment:
    """Bind the current native spec to its explicit historical migration identity.

    spec supplies the native cfg and file validators; None raises ValueError.
    source_tag defaults to the native tag. Pass schemas to retain an exact
    historical Labber disk declaration. An empty tuple raises ValueError.
    None derives the schemas from all native variables. No files are read
    until a validator is called.
    """
    if spec is None:
        raise ValueError("Missing native experiment declaration")
    if schemas is not None and not schemas:
        raise ValueError("schemas must declare at least one legacy log")

    def validate_cfg(snapshot: CfgSnapshot) -> None:
        validate_cfg_snapshot(snapshot)
        if snapshot.cfg_type != spec.cfg_type.__name__:
            raise ValueError(f"cfg_type must be {spec.cfg_type.__name__}")
        if int(snapshot.schema_version.split(".")[0]) != int(
            spec.cfg_schema_version.split(".")[0]
        ):
            raise ValueError(
                f"cfg_schema_version major must match {spec.cfg_schema_version}"
            )
        spec.cfg_type.model_validate(deepcopy(snapshot.values), extra="ignore")

    def validate(path: Path) -> None:
        load_run(path, spec=spec)

    return MigrationExperiment(
        source_tag=spec.tag if source_tag is None else source_tag,
        native_tag=spec.tag,
        cfg_type=spec.cfg_type.__name__,
        schemas=_legacy_schemas(spec) if schemas is None else schemas,
        validate_cfg=validate_cfg,
        validate_native=validate,
    )
