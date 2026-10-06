"""Stateless persistence for record-based experiments."""

from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np

from zcu_tools.datafile import (
    AxisSchema,
    LabberData,
    RunMetadata,
    RunSnapshot,
    VariableSchema,
    cast_labber_values,
    validate_labber_payload,
)
from zcu_tools.experiment.axes_spec import AxesSpec, GroupedAxesSpec
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord

T_Result = TypeVar("T_Result")
T_Config = TypeVar("T_Config", bound=ExpCfgModel)


class PersistableExperiment(Generic[T_Result, T_Config]):
    """Stateless persistence through a single or grouped AXES_SPEC declaration.

    save/load remain explicit Labber entries; save_run/load_run are native.
    Callers choose the RunRecord and exact local Path. Loading returns a new
    record without changing this instance. Callers own path reservation and
    native replace policy. No entry guesses the other file format.
    """

    #: Single or grouped mapping required by both explicit persistence formats.
    AXES_SPEC: ClassVar[AxesSpec[Any, Any] | GroupedAxesSpec[Any, Any] | None] = None

    def _spec(
        self,
    ) -> AxesSpec[T_Result, T_Config] | GroupedAxesSpec[T_Result, T_Config]:
        spec = type(self).AXES_SPEC
        if spec is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no AXES_SPEC; "
                "not migrated to native persistence"
            )
        return spec

    def save_run(
        self,
        source: RunRecord[T_Config, T_Result],
        destination: Path,
        *,
        metadata: RunMetadata,
        replace: bool = False,
    ) -> None:
        """Save native data with this instance's explicit persistence declaration.

        source.cfg must be present. metadata is historical run/snapshot/provenance
        evidence, never recaptured from live state. Destination and replace policy
        belong to the caller; conflicts raise FileExistsError, malformed records
        raise ValueError or TypeError and I/O errors propagate. Existing save()
        remains the separate Labber entry, with no format guessing.
        """
        from .native_persistence import save_run

        save_run(
            source, destination, spec=self._spec(), metadata=metadata, replace=replace
        )

    def load_run(
        self, source: Path
    ) -> tuple[RunRecord[T_Config, T_Result], RunSnapshot]:
        """Read native typed data and its historical snapshot using this spec.

        Read once without a full file image. Invalid native format, tag, schema,
        variable or cfg fails with the public located validation error; I/O
        errors propagate. Do not guess a legacy format or mutate this instance.
        Use datafile's generic image seam for unknown-field lossless rewrites.
        """
        from .native_persistence import load_run

        return load_run(source, spec=self._spec())

    def _validate_canonical_labber_data(
        self, data: LabberData, spec: AxesSpec[T_Result, T_Config]
    ) -> None:
        schema = VariableSchema(
            variable=spec.data_variable,
            axes=tuple(
                AxisSchema(name=axis.label, unit=axis.unit, dtype=np.dtype(axis.dtype))
                for axis in spec.axes
            ),
            signal_name=spec.z.label,
            signal_unit=spec.z.unit,
            signal_dtype=np.dtype(spec.z.dtype),
        )
        validate_labber_payload(
            data.payload, schema=schema, context=f"{type(self).__name__} canonical"
        )

    def save(
        self,
        source: RunRecord[T_Config, T_Result],
        destination: Path,
        *,
        comment: str | None = None,
        tag: str | None = None,
    ) -> None:
        """Save this typed record at the caller's Labber-normalized local Path.

        source.cfg must be present. comment is appended to its JSON cfg comment;
        tag overrides the declaration's Labber tag. Single mappings write one
        log, grouped mappings require a common grid. Invalid arrays or missing
        cfg raise ValueError; existing paths raise FileExistsError and I/O
        errors propagate. This entry neither selects a path nor writes native.
        """
        from zcu_tools.datafile import (
            ExperimentPayload,
            LabberMetadata,
            LabberPayload,
            format_ext,
            write_labber,
        )
        from zcu_tools.experiment.utils import make_labber_cfg_snapshot

        spec = self._spec()
        if isinstance(spec, GroupedAxesSpec):
            spec.save(source, destination, comment=comment, tag=tag)
            return
        result = source.result

        cfg = source.cfg
        if cfg is None:
            raise ValueError("RunRecord.cfg is None; cannot save without configuration")

        axes = [
            (ax.label, ax.unit, np.asarray(getattr(result, ax.field_name)) * ax.scale)
            for ax in spec.axes
        ]
        z = (spec.z.label, spec.z.unit, np.asarray(getattr(result, spec.z.field_name)))

        payload = ExperimentPayload(
            variables={spec.data_variable: LabberPayload(z, axes=axes)},
            metadata=LabberMetadata(tags=tag or spec.tag),
            representation="single",
        )
        write_labber(
            Path(format_ext(str(destination))),
            payload,
            cfg=make_labber_cfg_snapshot(cfg, schema_version=spec.cfg_schema_version),
            comment=comment,
        )

    def load(self, source: Path) -> RunRecord[T_Config, T_Result]:
        """Read declared single/grouped Labber data into a new memory-unit record.

        source is an exact local Path. Incorrect variables, axes, units or shape
        raise ValueError and I/O errors propagate. Missing cfg or non-envelope
        comment text returns cfg=None with valid Result data. Recognized comment
        envelopes with invalid cfg/comment/timestamp field types raise ValueError.
        A valid envelope whose cfg object fails the declared cfg model warns and
        returns cfg=None while keeping valid Result data. This entry does not
        guess native or legacy layouts.
        """
        from zcu_tools.datafile import load_labber_data
        from zcu_tools.experiment.utils import parse_comment

        spec = self._spec()
        if isinstance(spec, GroupedAxesSpec):
            return spec.load(source)

        filepath = str(source)
        ld = load_labber_data(filepath)
        self._validate_canonical_labber_data(ld, spec)

        cfg_snapshot = None
        if ld.comment:
            cfg_dict, _, _ = parse_comment(ld.comment)
            if cfg_dict is not None:
                cfg_snapshot = spec.cfg_type.validate_or_warn(cfg_dict, source=filepath)

        kwargs: dict[str, Any] = {
            ax.field_name: (
                np.asarray(ld.axes[i].values, dtype=np.float64) / ax.scale
            ).astype(ax.dtype)
            for i, ax in enumerate(spec.axes)
        }
        kwargs[spec.z.field_name] = cast_labber_values(
            ld.z,
            np.dtype(spec.z.dtype),
            context=f"{type(self).__name__} canonical z channel {spec.z.label!r}",
        )
        return RunRecord(cfg=cfg_snapshot, result=spec.result_type(**kwargs))
