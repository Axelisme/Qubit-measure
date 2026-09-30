"""Explicit data provenance for stateless experiments and their callers."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from .cfg_model import ExpCfgModel

CfgT = TypeVar("CfgT", bound=ExpCfgModel)
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class RunRecord(Generic[CfgT, ResultT]):
    """Pair a nullable configuration snapshot with one typed acquisition result.

    Callers own the result data. Hardware handles and presentation do not belong
    in this envelope. Loaded data may remain usable without a valid configuration.
    """

    cfg: CfgT | None
    result: ResultT
