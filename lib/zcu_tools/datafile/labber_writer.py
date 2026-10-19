"""Exact-path Labber export for the shared experiment payload."""

from pathlib import Path
from typing import Literal

import numpy as np

from .comment import encode_labber_comment
from .grouped import write_grouped_labber_data_file
from .labber import write_labber_data_file
from .labber_schema import validate_labber_payload
from .models import GroupedLabberData, LabberData, LabberMetadata
from .native_models import AxisSchema, CfgSnapshot, ExperimentPayload, VariableSchema
from .streaming import StreamingLabberVariableSpec, open_streaming_grouped_labber_data


def write_labber(
    destination: Path,
    payload: ExperimentPayload,
    *,
    cfg: CfgSnapshot,
    comment: str | None = None,
    grouped_layout: Literal["common", "independent"] = "common",
) -> None:
    """Write SI/discrete payload as Labber at the caller's exact local path.

    representation selects single-log or canonical common-grid grouped output,
    including one-member grouped. cfg.values is stored in the cfg JSON comment;
    comment overrides nonempty payload.metadata.comment when supplied. Other shared
    metadata is retained. This entry neither normalizes the filename nor calls
    the native writer. Existing paths raise FileExistsError; malformed payloads
    or incompatible grouped grids raise ValueError; I/O errors propagate.
    Labber requires at least one step axis; scalar payloads are rejected before
    creating the destination, without restricting native scalar persistence.
    A failed write may leave a partial new file; no cross-file transaction is
    promised. cfg identity/version remains native-only metadata.

    grouped_layout='common' preserves canonical one-shot grouped v2. Explicit
    'independent' is for explicitly declared grouped data with 1D/2D variables:
    it uses the existing marker-qualified heterogeneous streaming v1 layout,
    writes all rows and marks the file finalized. The public grouped reader
    retains each variable's axes, values and per-trace timestamps. This is not
    canonical common-grid output or an automatic format fallback. Unknown options,
    single representation with independent layout, and independent variables
    above rank two raise ValueError before creating the destination.
    """
    if grouped_layout not in {"common", "independent"}:
        raise ValueError("grouped_layout must be common or independent")
    if grouped_layout == "independent" and payload.representation != "grouped":
        raise ValueError("independent layout requires grouped representation")
    if not payload.variables:
        raise ValueError("payload variables must not be empty")
    if payload.representation not in {"single", "grouped"}:
        raise ValueError("representation must be single or grouped")
    if payload.representation == "single" and len(payload.variables) != 1:
        raise ValueError("single representation requires exactly one variable")
    for variable, signal in payload.variables.items():
        if not signal.axes:
            raise ValueError(
                f"{destination}: {variable} requires at least one Labber step axis"
            )
        validate_labber_payload(
            signal,
            schema=VariableSchema(
                variable=variable,
                axes=tuple(
                    AxisSchema(
                        name=axis.name, unit=axis.unit, dtype=np.dtype(np.float64)
                    )
                    for axis in signal.axes
                ),
                signal_name=signal.data.name,
                signal_unit=signal.data.unit,
                signal_dtype=np.dtype(np.complex128),
            ),
            context=f"{destination}: {variable}",
        )
    original = payload.metadata
    metadata = LabberMetadata(
        comment=encode_labber_comment(
            cfg.values, (original.comment or None) if comment is None else comment
        ),
        tags=original.tags,
        project=original.project,
        user=original.user,
        creation_time=original.creation_time,
    )
    if grouped_layout == "independent":
        _write_independent(destination, payload, metadata)
        return
    if payload.representation == "grouped":
        write_grouped_labber_data_file(
            str(destination),
            GroupedLabberData(
                {
                    str(variable): signal
                    for variable, signal in payload.variables.items()
                },
                metadata=metadata,
            ),
        )
        return
    signal = next(iter(payload.variables.values()))
    write_labber_data_file(
        str(destination), LabberData(payload=signal, metadata=metadata)
    )


def _write_independent(
    destination: Path, payload: ExperimentPayload, metadata: LabberMetadata
) -> None:
    specs: list[StreamingLabberVariableSpec] = []
    for variable, signal in payload.variables.items():
        values = np.asarray(signal.z)
        if values.ndim not in {1, 2}:
            raise ValueError("independent workflow layout requires rank one or two")
        if signal.timestamps is not None:
            timestamps = np.asarray(signal.timestamps)
            count = 1 if values.ndim == 1 else len(values)
            if (
                timestamps.ndim != 1
                or timestamps.dtype.kind not in "iuf"
                or len(timestamps) != count
            ):
                raise ValueError(
                    "independent timestamps require one real epoch value per inner trace"
                )
        specs.append(
            StreamingLabberVariableSpec(
                variable,
                signal.data.name,
                signal.data.unit,
                signal.axes,
                values.shape,
            )
        )
    with open_streaming_grouped_labber_data(
        str(destination), specs, metadata=metadata
    ) as writer:
        for variable, signal in payload.variables.items():
            values = np.asarray(signal.z)
            timestamps = (
                None
                if signal.timestamps is None
                else np.asarray(signal.timestamps, dtype=float)
            )
            for index, row in enumerate(values):
                timestamp = None
                if timestamps is not None:
                    timestamp = float(timestamps[0 if values.ndim == 1 else index])
                writer.write_outer_slice(variable, index, row, timestamp=timestamp)
        writer.finalize()
