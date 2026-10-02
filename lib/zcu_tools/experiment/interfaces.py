"""Minimum record-based core contract; synchronous analysis is opt-in."""

from pathlib import Path
from typing import Protocol, TypeVar

from zcu_tools.plotting.plots import Plots

from .cfg_model import ExpCfgModel
from .context import RunContext
from .records import RunRecord

CfgT = TypeVar("CfgT", bound=ExpCfgModel)
ResultT = TypeVar("ResultT")
OptionsT = TypeVar("OptionsT", contravariant=True)
AnalysisT = TypeVar("AnalysisT", covariant=True)


class RecordExperiment(Protocol[CfgT, ResultT]):
    """A stateless core with explicit acquisition and persistence sources."""

    def run(self, config: CfgT, *, context: RunContext) -> ResultT: ...

    def save(
        self,
        source: RunRecord[CfgT, ResultT],
        destination: Path,
        *,
        comment: str | None = None,
        tag: str | None = None,
    ) -> None: ...

    def load(self, source: Path) -> RunRecord[CfgT, ResultT]: ...


class SynchronousExperiment(
    RecordExperiment[CfgT, ResultT],
    Protocol[CfgT, ResultT, OptionsT, AnalysisT],
):
    """A core that can analyze a specified record without interactive input."""

    def analyze(
        self, source: RunRecord[CfgT, ResultT], options: OptionsT, *, plots: Plots
    ) -> AnalysisT: ...
