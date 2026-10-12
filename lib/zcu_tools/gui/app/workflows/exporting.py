"""Host admission, independent output attempts and saved ledger publication."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from zcu_tools.datafile import RunMetadata, save_run_data, write_labber
from zcu_tools.experiment.workflows import (
    CommittedRecord,
    Engine,
    InvalidRunState,
    RunIdentity,
    RunMismatch,
)
from zcu_tools.experiment.workflows.exporting import (
    FluxdepExportOutput,
    FluxdepExportReport,
    FluxdepExportSnapshot,
    PreparedFluxdepExport,
)
from zcu_tools.resources.entry import (
    ArtifactKey,
    LedgerEvent,
    Origin,
    RecordsLedger,
    SavedOutput,
    SavedPayload,
    SaveLayout,
)
from zcu_tools.resources.entry.ledger_models import validate_origin

from .samples import SampleExportBinding

logger = logging.getLogger(__name__)


class FluxdepPayloadPreparer(Protocol):
    """Science callback bound to the fixed plan/root, with no write side effects.

    records is the detached tuple returned by this run's Engine.records. The
    keyword-only call matches the T1/T2/MIST adapter seam. The callback returns
    the shared payload/cfg; unusable evidence or policy errors raise before any
    export attempt. It must not read GUI drafts, devices or uncommitted files.
    """

    def __call__(
        self, *, records: Sequence[CommittedRecord]
    ) -> PreparedFluxdepExport: ...


@dataclass(frozen=True)
class FluxdepExportBinding:
    """One Start's fixed storage and science callback owned by Controller.

    snapshot captures identity/paths/historical entry and host UTC times.
    prepare is bound to a detached native plan and data root. result_path and
    database_path are exact entry directories from Start storage, not parent
    roots or inferred workflow paths. ledger is the captured entry-scoped owner.
    Each export builds a fresh SaveLayout with these paths and its saved_at.
    No resources or files are opened by constructing this binding.
    """

    snapshot: FluxdepExportSnapshot
    prepare: FluxdepPayloadPreparer
    result_path: Path
    database_path: Path
    ledger: RecordsLedger


@dataclass(frozen=True)
class WorkflowExportBindings:
    """One Start's optional export capabilities and their common identity check.

    fluxdep is the captured merged/Labber binding, or None when unsupported.
    samples is the captured sample owner/producer/reason, or None when no sample
    capability was supplied. No run state, I/O or source selection is owned here.
    The Controller unpacks these bindings once; it does not retain this envelope.
    """

    fluxdep: FluxdepExportBinding | None = None
    samples: SampleExportBinding | None = None

    def require_identity(self, identity: RunIdentity) -> None:
        """Reject bindings for a different Start before publication or submission.

        identity is the proposed native Start identity. Fluxdep compares its full
        captured RunIdentity; samples compares its run_id. Mismatch raises
        ValueError without mutation, I/O, admission or replacement of an old run.
        """
        if self.fluxdep is not None and self.fluxdep.snapshot.identity != identity:
            raise ValueError("Export binding identity must match Start identity")
        if self.samples is not None and self.samples.run_id != identity.run_id:
            raise ValueError("Sample binding identity must match Start identity")


@dataclass(frozen=True)
class FluxdepExportRequest:
    """One explicit export's complete write intent, independent of Start binding.

    snapshot is the fixed Start identity/roots/entry capture and host UTC times.
    prepare is the science callback bound to the fixed plan/root; None explicitly
    denotes an unsupported workflow. layout is this call's fresh injected
    SaveLayout using captured entry paths and saved_at, never inferred from roots.
    ledger is the captured entry-scoped RecordsLedger. origin is gui, mcp or
    notebook. saved_at is timezone-aware UTC for the saved event and destination
    date; invalid time/origin raises ValueError before writes. Construction does
    not allocate paths, open files, execute science or read devices.
    """

    snapshot: FluxdepExportSnapshot
    prepare: FluxdepPayloadPreparer | None
    layout: SaveLayout
    ledger: RecordsLedger
    origin: Origin
    saved_at: datetime

    def __post_init__(self) -> None:
        validate_origin(self.origin)
        if self.saved_at.utcoffset() != timedelta(0):
            raise ValueError("saved_at must be timezone-aware UTC")


def export_fluxdep_run[C](
    run_id: str, *, engine: Engine[C], request: FluxdepExportRequest
) -> FluxdepExportReport:
    """Explicitly export committed fluxdep data without changing run state.

    run_id must match both the Engine's attached native run and fixed snapshot.
    engine.status must be done/stopped/failed/aborted; mismatch raises RunMismatch,
    nonterminal raises InvalidRunState. prepare=None rejects an unsupported
    workflow with ValueError. The keyword callback receives Engine.records;
    preparation errors propagate before output writes. identity, provenance,
    roots and entry snapshot are host captures, never live input or journal reads.
    request binds the fixed capture, science callback, fresh SaveLayout, ledger,
    origin and UTC saved_at for this call. It is never rebuilt from live drafts.

    Native atomically replaces data_root/merged.h5 through its public writer.
    Labber allocation uses layout.outputs, then writes the same payload using
    explicit independent workflow layout at a new exact path. No retry or
    cross-file transaction is attempted. Each allocation/writer error is
    isolated and returned, with native then Labber receipts. Allocation failure
    has Labber path=None. Native identity validation remains the writer's own
    contract and does not suppress Labber. No hardware is queried.

    Append one saved event with every known-path attempt; its record retains
    the full report including unallocated Labber errors. Ledger append failure
    is returned as ledger_error without discarding successful paths. Canonical
    records, workflow lifecycle and device/acquisition evidence are unchanged.
    """
    snapshot = request.snapshot
    prepare = request.prepare
    status = engine.status()
    if status is None or status.run_id != run_id:
        raise RunMismatch(run_id, None if status is None else status.run_id)
    if snapshot.identity.run_id != run_id:
        raise RunMismatch(run_id, snapshot.identity.run_id)
    if status.lifecycle not in {"done", "stopped", "failed", "aborted"}:
        raise InvalidRunState("export_fluxdep", status.lifecycle)
    if prepare is None:
        raise ValueError("Workflow has no fluxdep export binding")
    if str(snapshot.snapshot.entry_id) != snapshot.identity.entry_id:
        raise ValueError("Export identity and captured entry must match")
    prepared = prepare(records=engine.records(run_id))
    metadata = RunMetadata(
        run_id=run_id,
        experiment=status.workflow,
        started_at=snapshot.started_at.isoformat(),
        finished_at=None
        if snapshot.finished_at is None
        else snapshot.finished_at.isoformat(),
        completion="complete"
        if status.lifecycle == "done"
        else "stopped"
        if status.lifecycle == "stopped"
        else "partial",
        labber_path=None,
        snapshot=snapshot.snapshot,
        provenance=snapshot.identity.provenance,
    )
    native_path = snapshot.paths.data_root / "merged.h5"
    try:
        save_run_data(
            native_path, prepared.payload, metadata, cfg=prepared.cfg, replace=True
        )
        native = FluxdepExportOutput("data_h5", native_path, "saved", None)
    except Exception as error:
        logger.exception("Native fluxdep export failed")
        native = FluxdepExportOutput(
            "data_h5", native_path, "failed", _diagnostic(error)
        )
    labber_path: Path | None = None
    try:
        output = next(
            item
            for item in request.layout.outputs(
                ArtifactKey(section="workflow", name="merged", member="data")
            )
            if item.format == "labber"
        )
        labber_path = output.path
        labber_path.parent.mkdir(parents=True, exist_ok=True)
        write_labber(
            labber_path,
            prepared.payload,
            cfg=prepared.cfg,
            grouped_layout="independent",
        )
        labber = FluxdepExportOutput("labber", labber_path, "saved", None)
    except Exception as error:
        logger.exception("Labber fluxdep allocation/write failed")
        labber = FluxdepExportOutput(
            "labber", labber_path, "failed", _diagnostic(error)
        )
    outputs = (native, labber)
    ledger_error = None
    try:
        _append_saved(
            request.ledger,
            snapshot,
            status.workflow,
            outputs,
            request.origin,
            request.saved_at,
        )
    except Exception as error:
        logger.exception("Fluxdep saved ledger append failed")
        ledger_error = _diagnostic(error)
    return FluxdepExportReport(run_id, outputs, ledger_error)


def _diagnostic(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _append_saved(
    ledger: RecordsLedger,
    snapshot: FluxdepExportSnapshot,
    workflow: str,
    outputs: tuple[FluxdepExportOutput, ...],
    origin: Origin,
    saved_at: datetime,
) -> None:
    ledger.append(
        LedgerEvent(
            id=str(uuid4()),
            kind="saved",
            at=saved_at.isoformat(),
            entry_id=str(snapshot.snapshot.entry_id),
            origin=origin,
            payload=SavedPayload(
                run_id=snapshot.identity.run_id,
                labber_path=next(
                    (
                        str(out.path)
                        for out in outputs
                        if out.format == "labber" and out.status == "saved"
                    ),
                    None,
                ),
                outputs=tuple(
                    SavedOutput(
                        artifact="merged",
                        member="data",
                        format=out.format,
                        path=str(out.path),
                        status=out.status,
                        error=out.error,
                    )
                    for out in outputs
                    if out.path is not None
                ),
            ),
        ),
        record={
            "workflow": workflow,
            "point": snapshot.snapshot.point,
            "outputs": [
                {
                    "format": out.format,
                    "path": None if out.path is None else str(out.path),
                    "status": out.status,
                    "error": out.error,
                }
                for out in outputs
            ],
        },
    )
