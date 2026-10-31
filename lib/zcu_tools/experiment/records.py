"""Explicit data provenance for stateless experiments and their callers."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import JsonObject
from zcu_tools.plotting.figures import NamedFigures

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
    metadata is a neutral JSON object, empty until callers add their own facts.
    Construction detaches metadata from the caller; nested values remain mutable.
    Native saving validates its schema; this envelope does not interpret keys.
    """

    cfg: CfgT | None
    result: ResultT
    metadata: JsonObject = field(default_factory=dict)

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
