"""Explicit data provenance for stateless experiments and their callers."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, TypeVar

from zcu_tools.plotting.figures import NamedFigures

from .cfg_model import ExpCfgModel

CfgT = TypeVar("CfgT", bound=ExpCfgModel)
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
    """

    cfg: CfgT | None
    result: ResultT

    def __post_init__(self) -> None:
        object.__setattr__(self, "cfg", deepcopy(self.cfg))


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
