"""One-call artifact persistence through exact format and ledger owners."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, TypeVar
from uuid import uuid4

from matplotlib.figure import Figure

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import encode_labber_comment, save, write_labber
from zcu_tools.resources.entry import (
    ArtifactKey,
    JsonObject,
    LedgerEvent,
    Origin,
    Output,
    RecordsLedger,
    SavedOutput,
    SavedPayload,
    SaveLayout,
)
from zcu_tools.resources.entry.ledger_models import validate_origin
from zcu_tools.utils import atomic_replace

from ..metadata import RunMetadata
from ..records import RunRecord
from .analysis_document import encode_analysis_document
from .items import (
    AnalysisItem,
    DataItem,
    DataPreparationFailure,
    FigureItem,
    SaveArtifact,
)

CfgT = TypeVar("CfgT", bound=ConfigBase)
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class OutputResult:
    """One attempted destination from the latest Save call.

    key identifies the artifact/member; output contains format and exact path.
    status is saved or failed. error is None for saved, a nonempty diagnostic
    for failed. It says nothing about earlier calls or current file existence.
    """

    key: ArtifactKey
    output: Output
    status: Literal["saved", "failed"]
    error: str | None

    def __post_init__(self) -> None:
        """Require a failure diagnostic exactly when status is failed."""
        if self.status == "failed":
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError("failed status requires a nonempty error")
        elif self.status == "saved":
            if self.error is not None:
                raise ValueError("saved status requires error=None")
        else:
            raise ValueError("status must be saved or failed")


@dataclass(frozen=True)
class SaveReport:
    """Latest Save results for one opaque run_id, with ordered output attempts.

    outputs follows artifact, member and SaveLayout output order. No prior-call
    successes or ledger receipts are retained.
    """

    run_id: str
    outputs: tuple[OutputResult, ...]


def save_artifacts(
    artifacts: tuple[SaveArtifact, ...],
    *,
    layout: SaveLayout,
    record: RunRecord[CfgT, ResultT],
    ledger: RecordsLedger,
    origin: Origin,
    figure_encoder: Callable[[Figure], bytes] | None = None,
) -> SaveReport:
    """Rewrite every selected output, isolate failures, then append one saved event.

    record.metadata supplies fixed run/context/acquisition evidence; missing
    or invalid metadata raises ValueError before writes. layout supplies
    exact entry destinations and ledger belongs to that entry. origin is gui,
    notebook or mcp. Validate members,
    but do not preflight path/content conflicts or establish a transaction.
    Data preparation failures mark all data destinations failed without I/O.
    Encode each figure/analysis item once. figure_encoder receives a native Figure
    and returns PNG bytes; None uses Figure.savefig. The callback runs synchronously
    on the calling thread, so callers must encode a live GUI Figure on its owner
    thread. AnalysisDocument uses the artifacts-owned analysis encoder.
    Encoding failure marks all destinations of that member failed and still
    attempts other members. Do not encode in the caller or accept bytes aliases.
    Labber paths appear only in output results and the saved ledger event.
    Native/PNG/JSON
    use single-file replacement; Labber never overwrites or retries.
    Return every output attempt, even when some fail. Ledger append errors
    propagate after the writes, without deleting files or returning a receipt.
    """
    metadata = record.metadata
    if metadata is None:
        raise ValueError("RunRecord.metadata is None; cannot save without metadata")
    metadata = RunMetadata.model_validate(metadata)
    validate_origin(origin)
    for artifact in artifacts:
        artifact.__post_init__()
    results: list[OutputResult] = []
    for artifact in artifacts:
        for member in artifact.members:
            outputs = layout.outputs(member.key)
            if isinstance(member, DataItem):
                attempts = _save_data(member, outputs, metadata)
            else:
                attempts = _save_file(member, outputs, figure_encoder)
            results.extend(attempts)
    report = SaveReport(metadata.run_id, tuple(results))
    successful_labber = next(
        (
            str(item.output.path)
            for item in reversed(results)
            if item.output.format == "labber" and item.status == "saved"
        ),
        None,
    )
    saved_record: JsonObject = {
        "point": metadata.snapshot.point,
        "experiment": metadata.experiment,
        "outputs": [
            {
                "section": item.key.section,
                "artifact": item.key.name,
                "member": item.key.member,
                "member_name": item.key.member_name,
                "format": item.output.format,
                "path": str(item.output.path),
                "status": item.status,
                "error": item.error,
            }
            for item in results
        ],
    }
    ledger.append(
        LedgerEvent(
            id=str(uuid4()),
            kind="saved",
            at=datetime.now(UTC).isoformat(),
            entry_id=str(metadata.snapshot.entry_id),
            origin=origin,
            payload=SavedPayload(
                run_id=report.run_id,
                labber_path=successful_labber,
                outputs=tuple(
                    SavedOutput(
                        artifact=item.key.name,
                        member=item.key.member,
                        format=item.output.format,
                        path=str(item.output.path),
                        status=item.status,
                        error=item.error,
                    )
                    for item in results
                ),
            ),
        ),
        record=saved_record,
    )
    return report


def _failed(key: ArtifactKey, output: Output, error: Exception) -> OutputResult:
    return OutputResult(key, output, "failed", f"{type(error).__name__}: {error}")


def _save_data(
    member: DataItem, outputs: tuple[Output, ...], metadata: RunMetadata
) -> tuple[OutputResult, ...]:
    if isinstance(member.data, DataPreparationFailure):
        return tuple(
            OutputResult(member.key, output, "failed", member.data.error)
            for output in outputs
        )
    attempts: list[OutputResult] = []
    for output in outputs:
        try:
            output.path.parent.mkdir(parents=True, exist_ok=True)
            if output.format == "labber":
                write_labber(
                    output.path,
                    member.data.container,
                    metadata=replace(
                        member.data.labber_metadata,
                        comment=encode_labber_comment(
                            member.data.cfg.values,
                            member.data.labber_metadata.comment,
                            snapshot=metadata.snapshot.to_labber_snapshot(),
                        ),
                    ),
                    grouped_layout=member.data.labber_layout,
                )
            else:
                save(output.path, member.data.to_container(metadata), replace=True)
        except Exception as error:
            # Isolate this destination; other formats still receive prepared data.
            logging.exception("Failed saving %s", output.path)
            attempts.append(_failed(member.key, output, error))
        else:
            attempts.append(OutputResult(member.key, output, "saved", None))
    return tuple(attempts)


def _save_file(
    member: FigureItem | AnalysisItem,
    outputs: tuple[Output, ...],
    figure_encoder: Callable[[Figure], bytes] | None,
) -> tuple[OutputResult, ...]:
    try:
        if isinstance(member.content, Figure):
            if figure_encoder is None:
                with BytesIO() as stream:
                    member.content.savefig(stream, format="png")
                    content = stream.getvalue()
            else:
                content = figure_encoder(member.content)
        else:
            content = encode_analysis_document(member.content)
    except Exception as error:
        # Encoding is member-scoped, so every equivalent destination shares the failure.
        logging.exception("Failed encoding member %s", member.key)
        return tuple(_failed(member.key, output, error) for output in outputs)
    attempts: list[OutputResult] = []
    for output in outputs:
        try:
            _replace_file(output.path, content)
        except Exception as error:
            # A failed run copy must not prevent the point copy or later members.
            logging.exception("Failed saving %s", output.path)
            attempts.append(_failed(member.key, output, error))
        else:
            attempts.append(OutputResult(member.key, output, "saved", None))
    return tuple(attempts)


def _replace_file(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            stream.close()
            atomic_replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
