"""Recording experiment core shared by Notebook adapter and widget-host tests."""

from dataclasses import dataclass

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.base import Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.plotting.plots import Plots


class Cfg(ConfigBase):
    """Run input; scale is the scalar value plotted and returned by the fake."""

    scale: float = 1.0


@dataclass
class Options:
    """Analysis input; the first weight scales the retained scalar result."""

    weights: list[float]


class RecordingCore(Experiment[float, Cfg]):
    """Record run contexts; flags inject producer failures.

    request_stop asks the run to stop before returning normally. This core has
    no persistence mapping.
    """

    def __init__(self) -> None:
        self.fail_analysis = False
        self.fail_run = False
        self.request_stop = False
        self.contexts: list[RunContext] = []

    def run(self, config: Cfg, *, context: RunContext) -> float:
        """Plot and return scale, then mutate the working cfg or inject failure."""
        self.contexts.append(context)
        result = config.scale
        _, axes = context.plots.subplots(num="raw")
        axes.plot([0.0], [result])
        config.scale = 99.0
        if self.fail_run:
            raise ValueError("Run failed after creating a diagnostic figure")
        if self.request_stop:
            context.scope.stop.set()
        return result

    def analyze(
        self,
        source: RunRecord[Cfg, float],
        options: Options,
        *,
        plots: Plots,
    ) -> float:
        """Plot the weighted result, mutate working inputs, or inject failure."""
        analysis = source.result * options.weights[0]
        if source.cfg is not None:
            source.cfg.scale = 42.0
        options.weights[0] = 99.0
        _, axes = plots.subplots(num="fit")
        axes.plot([0.0], [analysis])
        if self.fail_analysis:
            raise ValueError("Analysis failed after creating a diagnostic figure")
        return analysis
