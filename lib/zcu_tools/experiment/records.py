"""Explicit data provenance for stateless experiments and their callers."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, TypeVar

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.plotting.figures import NamedFigures

from .metadata import RunMetadata, RunMetadataContent

CfgT = TypeVar("CfgT", bound=ConfigBase)
ResultT = TypeVar("ResultT")
OptionsT = TypeVar("OptionsT")
AnalysisT = TypeVar("AnalysisT")


@dataclass(frozen=True)
class RunRecord(Generic[CfgT, ResultT]):
    """Pair a nullable configuration snapshot with one typed acquisition result.

    Construction copies cfg to isolate later edits to caller inputs. The fields
    stay paired, while cfg and result data remain mutable. Callers own the result
    data; hardware handles and presentation do not belong in this envelope.
    Loaded data may remain usable without a valid configuration.
    metadata holds complete captured RunMetadata, deferred RunMetadataContent
    after native Load, or None until the host captures historical facts. Deferred
    properties validate requested fields; project() requests all fields.
    Construction detaches metadata from the caller; nested values remain mutable.
    Native saving requires metadata; no writer captures live evidence.
    """

    cfg: CfgT | None
    result: ResultT
    metadata: RunMetadata | RunMetadataContent | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cfg", deepcopy(self.cfg))
        object.__setattr__(self, "metadata", deepcopy(self.metadata))


@dataclass(frozen=True)
class AnalysisRecord(Generic[CfgT, ResultT, OptionsT, AnalysisT]):
    """Keep one successful analysis with its explicit source and named figures.

    Construction isolates caller options. The source, numerical result and native
    figures are retained, not copied. Presentation belongs to a separate handle.
    """

    source: RunRecord[CfgT, ResultT]
    options: OptionsT
    result: AnalysisT
    figures: NamedFigures

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", deepcopy(self.options))
