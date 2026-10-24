"""Stateless experiment declarations shared by GUI and Notebook callers."""

from typing import Any, ClassVar, Generic, TypeVar

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.plotting.plots import Plots

from .axes_spec import AxesSpec, GroupedAxesSpec
from .context import RunContext
from .records import RunRecord

T_Result = TypeVar("T_Result")
T_Config = TypeVar("T_Config", bound=ConfigBase)


class Experiment(Generic[T_Result, T_Config]):
    """Declare acquisition and optional analysis without retaining run state.

    Subclasses implement run and whichever analysis stages they support.
    Unimplemented operations raise NotImplementedError. AXES_SPEC maps records
    for external persistence functions; None means persistence is unavailable.
    Hosts own records, plots, hardware lifetimes and file paths.
    """

    #: Single/grouped mapping of Result and cfg, or None without persistence.
    AXES_SPEC: ClassVar[AxesSpec[Any, Any] | GroupedAxesSpec[Any, Any] | None] = None

    def run(self, cfg: T_Config, /, *, context: RunContext) -> T_Result:
        """Acquire a Result from positional cfg using this operation's context.

        The caller owns context hardware and plots. Subclasses define their
        device requirements, cancellation and partial-result behavior.
        """
        raise NotImplementedError("Experiment must implement run")

    def analyze(
        self, source: RunRecord[T_Config, T_Result], options: Any, *, plots: Plots
    ) -> Any:
        """Analyze an explicit record with experiment-specific options and plots.

        Subclasses define options and output types, including whether cfg=None
        is accepted. The base provides no analysis or retained result.
        """
        raise NotImplementedError("Experiment does not implement analyze")

    def post_analyze(
        self,
        source: RunRecord[T_Config, T_Result],
        primary: Any,
        options: Any,
        *,
        plots: Plots,
    ) -> Any:
        """Derive a second analysis from this record and its primary analysis.

        Subclasses define primary/options/output types and publish to plots.
        The base provides no post-analysis or retained result.
        """
        raise NotImplementedError("Experiment does not implement post_analyze")
