"""Exact-path Labber export for the shared experiment payload."""

from pathlib import Path

import numpy as np

from .comment import encode_labber_comment
from .grouped import write_grouped_labber_data_file
from .labber import write_labber_data_file
from .labber_schema import validate_labber_payload
from .models import GroupedLabberData, LabberData, LabberMetadata
from .native_models import AxisSchema, CfgSnapshot, ExperimentPayload, VariableSchema


def write_labber(
    destination: Path,
    payload: ExperimentPayload,
    *,
    cfg: CfgSnapshot,
    comment: str | None = None,
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
    """
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
