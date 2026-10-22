"""Fixed typed artifact items; format selection belongs to the shared saver."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TypeVar

from matplotlib.figure import Figure

from zcu_tools.resources.entry import ArtifactKey

from ..axes_spec import AxesSpec, GroupedAxesSpec
from ..cfg_model import ExpCfgModel
from ..native_persistence import PreparedRunData, prepare_run_data
from ..records import RunRecord
from .analysis_document import AnalysisDocument

CfgT = TypeVar("CfgT", bound=ExpCfgModel)
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class ArtifactRef:
    """Safe artifact identity, validated like ArtifactKey.

    section identifies the producing pane; name identifies its artifact save unit.
    Both are safe path segments, never file paths or format names.
    """

    section: str
    name: str

    def __post_init__(self) -> None:
        """Reject unsafe names through ArtifactKey's existing validation."""
        ArtifactKey(self.section, self.name, "data")


@dataclass(frozen=True)
class ArtifactItem(ABC):
    """One typed member identified by key; only the three fixed kinds are supported.

    key supplies safe section/artifact/member identity, never an output path.
    Callers must not mutate arrays, figures or documents while saving.
    """

    key: ArtifactKey

    @abstractmethod
    def __post_init__(self) -> None:
        """Reject identity/content mismatches before saving."""


@dataclass(frozen=True)
class DataPreparationFailure:
    """Failed data preparation, not a prepared payload.

    error is a nonempty human-readable diagnostic shared by all data outputs.
    """

    error: str

    def __post_init__(self) -> None:
        """Reject an empty diagnostic."""
        if not self.error.strip():
            raise ValueError("data preparation failure requires an error")


@dataclass(frozen=True)
class DataItem(ArtifactItem):
    """Data identity and either prepared SI content or an explicit failure.

    key must use member=data. data is shared by native and Labber writers.
    Use from_run for typed records, or failed when a producer hook raises.
    No unvalidated content can stand in for PreparedRunData.
    """

    data: PreparedRunData | DataPreparationFailure

    def __post_init__(self) -> None:
        """Require data identity and a typed preparation outcome."""
        if self.key.member != "data":
            raise ValueError("DataItem requires member=data")
        _require_content(self.data, (PreparedRunData, DataPreparationFailure))

    @classmethod
    def from_run(
        cls,
        key: ArtifactKey,
        source: RunRecord[CfgT, ResultT],
        *,
        spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT],
        comment: str,
    ) -> "DataItem":
        """Prepare a typed run once, retaining errors as failed data outputs.

        source/spec/comment follow prepare_run_data's contract. Mapping errors,
        including cfg=None, become diagnostics; unsafe/non-data keys still raise.
        No I/O, context capture or fabricated cfg is performed.
        """
        try:
            prepared = prepare_run_data(source, spec=spec, comment=comment)
        except Exception as error:
            logging.exception("Failed preparing data item %s", key)
            return cls.failed(key, error)
        return cls(key, prepared)

    @classmethod
    def failed(cls, key: ArtifactKey, error: Exception) -> "DataItem":
        """Capture a producer's exception as a data-only failure, without traceback.

        key must use member=data; error is the actual preparation exception.
        All destinations report its type and message; other items still save.
        """
        return cls(key, DataPreparationFailure(f"{type(error).__name__}: {error}"))

    def with_comment(self, comment: str) -> "DataItem":
        """Return prepared content with a new raw comment, retaining any failure.

        No arrays are copied and preparation is not retried. Writers validate
        comment according to their format contracts during save.
        """
        if isinstance(self.data, DataPreparationFailure):
            return self
        return replace(
            self,
            data=replace(
                self.data,
                payload=replace(
                    self.data.payload,
                    metadata=replace(self.data.payload.metadata, comment=comment),
                ),
            ),
        )


@dataclass(frozen=True)
class FigureItem(ArtifactItem):
    """Figure identity and native content, encoded as PNG once per save.

    key must use member=figure. content is a native matplotlib Figure, not bytes.
    """

    content: Figure

    def __post_init__(self) -> None:
        """Require figure identity and native Figure content."""
        if self.key.member != "figure":
            raise ValueError("FigureItem requires member=figure")
        _require_content(self.content, Figure)


@dataclass(frozen=True)
class AnalysisItem(ArtifactItem):
    """Analysis identity and document, encoded as JSON once per save.

    key must use member=analysis. content is an AnalysisDocument, not JSON bytes.
    """

    content: AnalysisDocument

    def __post_init__(self) -> None:
        """Require analysis identity and typed document content."""
        if self.key.member != "analysis":
            raise ValueError("AnalysisItem requires member=analysis")
        _require_content(self.content, AnalysisDocument)


@dataclass(frozen=True)
class SaveArtifact:
    """One artifact ref with nonempty, unique matching items in producer order.

    ref identifies the save unit. members contains only the fixed item kinds;
    each key must match ref's section/name. Formats and paths remain framework-owned.
    """

    ref: ArtifactRef
    members: tuple[DataItem | FigureItem | AnalysisItem, ...]

    def __post_init__(self) -> None:
        """Reject unsupported kinds, empty selections and mismatched/duplicate keys."""
        if not self.members:
            raise ValueError("artifact requires nonempty members")
        keys: set[ArtifactKey] = set()
        for member in self.members:
            if type(member) not in (DataItem, FigureItem, AnalysisItem):
                raise TypeError(
                    "artifact requires a fixed DataItem, FigureItem or AnalysisItem"
                )
            member.__post_init__()
            if (member.key.section, member.key.name) != (
                self.ref.section,
                self.ref.name,
            ):
                raise ValueError("member section/artifact must match ref")
            if member.key in keys:
                raise ValueError("artifact member keys must be unique; duplicate key")
            keys.add(member.key)


def _require_content(
    content: object, expected: type[object] | tuple[type[object], ...]
) -> None:
    # Keep the runtime boundary for untyped callers as well as the typed fields.
    if not isinstance(content, expected):
        raise TypeError(f"content must be {expected}")
