"""Typed scientific payloads and isolated writers for fluxdep tools, not a dispatcher."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from zcu_tools.datafile import (
    CfgSnapshot,
    ExperimentPayload,
    LabberPayload,
    RunMetadata,
    RunSnapshot,
    load_run_data,
    save_run_data,
    validate_experiment_payload,
    write_labber,
)
from zcu_tools.experiment.axes_spec import AxesSpec, GroupedAxesSpec
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.native_persistence import native_schemas
from zcu_tools.resources.entry import ArtifactKey, SaveLayout

from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoExp
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyExp

from .coherence_measurements import LifetimeKind
from .export_inputs import SavedManifest, native_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedFluxdepExport:
    """Detached grouped SI scientific variables and the complete fixed plan cfg."""

    payload: ExperimentPayload
    cfg: CfgSnapshot


@dataclass(frozen=True)
class FluxdepExportOutput:
    """One selected writer attempt: exact path, saved/failed status and diagnosis.

    Only allocation failure has path=None. A writer failure retains its path.
    Saved outputs have error=None; failed outputs have a nonempty diagnostic.
    """

    format: Literal["data_h5", "labber"]
    path: Path | None
    status: Literal["saved", "failed"]
    error: str | None


def load_trace[R, C: ExpCfgModel](
    directory: Path, filename: str, spec: AxesSpec[R, C] | GroupedAxesSpec[R, C]
) -> LabberPayload:
    """Load exactly the point-named native trace using its experiment declaration.

    The filename must stay within the invocation directory. Wrong experiment,
    incomplete acquisition, schema mismatch or empty data raises ValueError
    naming that file. Public native reader I/O errors propagate. Return the
    experiment's single LabberPayload in disk units; never re-fit or interpolate.
    """
    path = native_path(directory, filename)
    try:
        stored = load_run_data(path)
        if (
            stored.metadata.experiment != spec.tag
            or stored.metadata.completion != "complete"
        ):
            raise ValueError(f"Expected complete {spec.tag} acquisition")
        schemas = native_schemas(spec)
        validate_experiment_payload(stored.payload, schema=schemas)
        trace = stored.payload.variables[schemas[0].variable]
        if not len(trace.z):
            raise ValueError("Empty acquisition")
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    return trace


def load_lifetime_trace(
    directory: Path, filename: str, kind: LifetimeKind
) -> LabberPayload:
    """Read a named completed lifetime with its own native declaration.

    kind selects T1, Ramsey or Echo, not a union schema or common delay grid.
    Directory/reference/schema/I/O errors retain load_trace's source diagnosis.
    """
    if kind == "t1":
        return load_trace(directory, filename, T1Exp().persistence_spec())
    if kind == "t2ramsey":
        return load_trace(directory, filename, T2RamseyExp().persistence_spec())
    return load_trace(directory, filename, T2EchoExp().persistence_spec())


def write_outputs(
    manifest: SavedManifest,
    prepared: PreparedFluxdepExport,
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
                if manifest.lifecycle == "stopped"
                else "partial",
                labber_path=None,
                snapshot=snapshot,
                provenance=manifest.identity.provenance,
            )
            save_run_data(
                path, prepared.payload, metadata, cfg=prepared.cfg, replace=True
            )
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
                path, prepared.payload, cfg=prepared.cfg, grouped_layout="independent"
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
