"""One-call artifact persistence through exact format and ledger owners."""

import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal
from uuid import uuid4

from matplotlib.figure import Figure

from zcu_tools.datafile import (
    AnalysisDocument,
    RunMetadata,
    encode_analysis_document,
    save_run_data,
    write_labber,
)
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

from .native_persistence import PreparedRunData


@dataclass(frozen=True)
class ArtifactRef:
    """Opaque safe section and artifact names, validated like ArtifactKey."""

    section: str
    name: str

    def __post_init__(self) -> None:
        """Reject unsafe names through ArtifactKey's existing validation."""
        ArtifactKey(self.section, self.name, "data")


@dataclass(frozen=True)
class DataMember:
    """One data key and its prepared SI payload/cfg, shared by both writers.

    key must have member=data. data is prepared from one typed RunRecord;
    callers must not mutate its arrays while saving.
    """

    key: ArtifactKey
    data: PreparedRunData

    def __post_init__(self) -> None:
        """Require the data member kind; reuse already prepared typed data."""
        if self.key.member != "data":
            raise ValueError("DataMember requires member=data")


@dataclass(frozen=True)
class FileMember:
    """One figure or analysis key with its typed, not-yet-encoded content.

    key must have member=figure with a native Figure, or member=analysis with
    an AnalysisDocument. The shared saver encodes each member once and reuses
    its bytes for equivalent outputs. Do not mutate content while saving.
    """

    key: ArtifactKey
    content: Figure | AnalysisDocument

    def __post_init__(self) -> None:
        """Require figure/analysis identity and the corresponding content type."""
        if self.key.member == "figure":
            if not isinstance(self.content, Figure):
                raise TypeError("figure content must be a Figure")
        elif self.key.member == "analysis":
            if not isinstance(self.content, AnalysisDocument):
                raise TypeError("analysis content must be an AnalysisDocument")
        else:
            raise ValueError("FileMember requires figure or analysis member")


@dataclass(frozen=True)
class SaveArtifact:
    """One artifact ref with nonempty, uniquely identified matching members.

    All member keys use ref's section/name. Members retain producer order;
    data payloads and once-encoded files are reused for equivalent outputs.
    """

    ref: ArtifactRef
    members: tuple[DataMember | FileMember, ...]

    def __post_init__(self) -> None:
        """Require a nonempty selection of unique keys matching this ref."""
        if not self.members:
            raise ValueError("artifact requires nonempty members")
        keys: set[ArtifactKey] = set()
        for member in self.members:
            member.__post_init__()
            if (member.key.section, member.key.name) != (
                self.ref.section,
                self.ref.name,
            ):
                raise ValueError("member section/artifact must match ref")
            if member.key in keys:
                raise ValueError("artifact member keys must be unique; duplicate key")
            keys.add(member.key)


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
    metadata: RunMetadata,
    ledger: RecordsLedger,
    origin: Origin,
    call_id: str | None,
) -> SaveReport:
    """Rewrite every selected output, isolate failures, then append one saved event.

    metadata supplies fixed run/context/acquisition evidence; layout supplies
    exact entry destinations and ledger belongs to that entry. origin is gui,
    notebook or mcp; only mcp requires a nonempty call_id. Validate members,
    but do not preflight path/content conflicts or establish a transaction.
    Encode each FileMember once through Figure.savefig or the analysis encoder.
    Encoding failure marks all destinations of that member failed and still
    attempts other members. Do not encode in the caller or accept bytes aliases.
    Attempt Labber before native for each data member, retaining the previous
    successful metadata.labber_path if this Labber output fails. Native/PNG/JSON
    use single-file replacement; Labber never overwrites or retries.
    Return every output attempt, even when some fail. Ledger append errors
    propagate after the writes, without deleting files or returning a receipt.
    """
    validate_origin(origin, call_id)
    for artifact in artifacts:
        artifact.__post_init__()
    results: list[OutputResult] = []
    current_metadata = metadata
    for artifact in artifacts:
        for member in artifact.members:
            outputs = layout.outputs(member.key)
            if isinstance(member, DataMember):
                attempts, current_metadata = _save_data(
                    member, outputs, current_metadata
                )
            else:
                attempts = _save_file(member, outputs)
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
    record: JsonObject = {
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
            call_id=call_id,
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
        record=record,
    )
    return report


def _failed(key: ArtifactKey, output: Output, error: Exception) -> OutputResult:
    return OutputResult(key, output, "failed", f"{type(error).__name__}: {error}")


def _save_data(
    member: DataMember, outputs: tuple[Output, ...], metadata: RunMetadata
) -> tuple[tuple[OutputResult, ...], RunMetadata]:
    attempts: dict[Output, OutputResult] = {}
    # Native must see the successful Labber link, but the report retains layout order.
    ordered = sorted(outputs, key=lambda output: output.format != "labber")
    for output in ordered:
        try:
            output.path.parent.mkdir(parents=True, exist_ok=True)
            if output.format == "labber":
                write_labber(output.path, member.data.payload, cfg=member.data.cfg)
                metadata = replace(metadata, labber_path=str(output.path))
            else:
                save_run_data(
                    output.path,
                    member.data.payload,
                    metadata,
                    cfg=member.data.cfg,
                    replace=True,
                )
        except Exception as error:
            # Isolate this destination; other formats still receive the same prepared data.
            logging.exception("Failed saving %s", output.path)
            attempts[output] = _failed(member.key, output, error)
        else:
            attempts[output] = OutputResult(member.key, output, "saved", None)
    return tuple(attempts[output] for output in outputs), metadata


def _save_file(
    member: FileMember, outputs: tuple[Output, ...]
) -> tuple[OutputResult, ...]:
    try:
        if isinstance(member.content, Figure):
            with BytesIO() as stream:
                member.content.savefig(stream, format="png")
                content = stream.getvalue()
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
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
