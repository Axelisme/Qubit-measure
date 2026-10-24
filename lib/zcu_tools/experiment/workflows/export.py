"""Experiment-independent saved workflow trace loading and output writers."""

import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import (
    LabberPayload,
    encode_labber_comment,
    load,
    save,
    write_labber,
)
from zcu_tools.experiment.axes_spec import AxesSpec, GroupedAxesSpec
from zcu_tools.experiment.metadata import RunMetadata, RunSnapshot
from zcu_tools.experiment.native_persistence import (
    PreparedRunData,
    metadata_from_container,
    restore_run,
)
from zcu_tools.resources.entry import ArtifactKey, SaveLayout

from .artifacts import RunManifest
from .saved_run import native_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FluxdepExportOutput:
    """One selected writer attempt: exact path, saved/failed status and diagnosis.

    format selects native data_h5 or Labber; status is saved or failed.
    Only allocation failure has path=None. A writer failure retains its path.
    Saved outputs have error=None; failed outputs have a nonempty diagnostic.
    """

    format: Literal["data_h5", "labber"]
    path: Path | None
    status: Literal["saved", "failed"]
    error: str | None


def load_trace[R, C: ConfigBase](
    directory: Path,
    filename: str,
    spec: AxesSpec[R, C] | GroupedAxesSpec[R, C],
    *,
    allow_stopped: bool = False,
) -> LabberPayload:
    """Load exactly the point-named native trace using its experiment declaration.

    The filename must stay within the invocation directory. Only complete
    acquisitions are accepted by default; allow_stopped=True also accepts stopped
    raw traces, preserving NaN values. Partial and unknown acquisitions always
    reject. Wrong experiment, disallowed completion, schema mismatch or empty
    data raises ValueError naming that file. Public native reader I/O errors
    propagate. Return the experiment's single LabberPayload in disk units with
    inner-first axes; never re-fit, interpolate or transpose the values.
    """
    path = native_path(directory, filename)
    try:
        stored = load(path)
        metadata = metadata_from_container(stored)
        completion_allowed = metadata.completion == "complete" or (
            allow_stopped and metadata.completion == "stopped"
        )
        if metadata.experiment != spec.tag or not completion_allowed:
            expected = "complete or stopped" if allow_stopped else "complete"
            raise ValueError(f"Expected {expected} {spec.tag} acquisition")
        restore_run(stored, spec=spec)
        name = (
            str(spec.required_variables[0])
            if isinstance(spec, GroupedAxesSpec)
            else str(spec.data_variable)
        )
        variable = stored.vars[name]
        trace = LabberPayload(
            (name, variable.unit, variable.values), list(reversed(variable.axes))
        )
        if not len(trace.z):
            raise ValueError("Empty acquisition")
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    return trace


def write_outputs(
    manifest: RunManifest,
    prepared: PreparedRunData,
    snapshot: RunSnapshot,
    *,
    merged: bool,
    database: Path | None,
) -> tuple[FluxdepExportOutput, ...]:
    """Write selected outputs independently, without ledger or run mutations.

    merged replaces data_root/merged.h5 atomically. Non-None database explicitly
    requests a fresh SaveLayout Labber date-tree path. Science preparation and
    destination preflight belong to the caller and must precede this function.
    Allocation/writer exceptions become typed failed outputs, preserving known
    paths. Native run_id rejection is not repaired and does not block Labber.
    """
    outputs: list[FluxdepExportOutput] = []
    if merged:
        path = manifest.roots.data_root / "merged.h5"
        try:
            metadata = RunMetadata(
                run_id=manifest.identity.run_id,
                experiment=manifest.workflow,
                started_at=manifest.started_at.isoformat(),
                finished_at=None
                if manifest.ended_at is None
                else manifest.ended_at.isoformat(),
                completion="complete"
                if manifest.lifecycle == "done"
                else "stopped"
                if manifest.lifecycle == "cancelled"
                else "partial",
                snapshot=snapshot,
                provenance=manifest.identity.provenance,
            )
            save(path, prepared.to_container(metadata), replace=True)
            outputs.append(FluxdepExportOutput("data_h5", path, "saved", None))
        except Exception as error:
            logger.exception("Native merged output failed: %s", path)
            outputs.append(
                FluxdepExportOutput(
                    "data_h5", path, "failed", f"{type(error).__name__}: {error}"
                )
            )
    if database is not None:
        path = None
        try:
            layout = SaveLayout(
                result_path=manifest.roots.metadata_root,
                database_path=database,
                run_id=manifest.identity.run_id,
                experiment=manifest.workflow,
                point=snapshot.point,
                saved_at=datetime.now(UTC),
            )
            output = next(
                item
                for item in layout.outputs(
                    ArtifactKey(section="workflow", name="merged", member="data")
                )
                if item.format == "labber"
            )
            path = output.path
            path.parent.mkdir(parents=True, exist_ok=True)
            write_labber(
                path,
                prepared.container,
                metadata=replace(
                    prepared.labber_metadata,
                    comment=encode_labber_comment(
                        prepared.cfg.values, prepared.labber_metadata.comment
                    ),
                ),
                grouped_layout=prepared.labber_layout,
            )
            outputs.append(FluxdepExportOutput("labber", path, "saved", None))
        except Exception as error:
            logger.exception("Labber output failed: %s", path)
            outputs.append(
                FluxdepExportOutput(
                    "labber", path, "failed", f"{type(error).__name__}: {error}"
                )
            )
    return tuple(outputs)


def validate_destinations(
    outputs: tuple[str, ...],
    supported: set[str],
    *,
    result: Path | None,
    database: Path | None,
) -> None:
    """Reject unknown/empty/duplicate selections or missing selected destinations.

    This preflight performs no writes and never requires unused destinations.
    Merged needs neither run-external path. Samples needs result, Labber database.
    """
    if (
        not outputs
        or len(outputs) != len(set(outputs))
        or not set(outputs) <= supported
    ):
        raise ValueError(f"Select unique outputs from {sorted(supported)}")
    if "samples" in outputs and result is None:
        raise ValueError("samples requires --result")
    if "labber" in outputs and database is None:
        raise ValueError("labber requires --database")
