"""One-call artifact persistence through exact format and ledger owners."""

from dataclasses import dataclass
from typing import Literal

from matplotlib.figure import Figure

from zcu_tools.datafile import AnalysisDocument, RunMetadata
from zcu_tools.resources.entry import (
    ArtifactKey,
    Origin,
    Output,
    RecordsLedger,
    SaveLayout,
)

from .native_persistence import PreparedRunData


@dataclass(frozen=True)
class ArtifactRef:
    """Opaque safe section and artifact names, validated like ArtifactKey."""

    section: str
    name: str

    def __post_init__(self) -> None:
        """Reject unsafe names through ArtifactKey's existing validation."""
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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
    raise NotImplementedError
