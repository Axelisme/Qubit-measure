"""Recording experiment core shared by Notebook adapter and widget-host tests."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.plotting.plots import Plots


class Cfg(ExpCfgModel):
    """Run input; scale is the scalar value plotted and returned by the fake."""

    scale: float = 1.0


@dataclass
class Options:
    """Analysis input; the first weight scales the retained scalar result."""

    weights: list[float]


class RecordingCore:
    """Record run contexts and saved sources; flags inject producer failures.

    signal_outcome selects stopped, failed, interrupted, or ordinary success
    (None). metadata retains the latest save comment and tag.
    """

    def __init__(self) -> None:
        self.fail_analysis = False
        self.fail_run = False
        self.signal_outcome: Literal["stopped", "failed", "interrupted"] | None = None
        self.contexts: list[RunContext] = []
        self.saved: list[tuple[RunRecord[Cfg, float], Path]] = []
        self.metadata: tuple[str | None, str | None] | None = None

    def run(self, config: Cfg, *, context: RunContext) -> float:
        """Plot and return scale, then mutate the working cfg or inject failure."""
        self.contexts.append(context)
        result = config.scale
        _, axes = context.plots.subplots("raw")
        axes.plot([0.0], [result])
        config.scale = 99.0
        if self.fail_run:
            raise ValueError("Run failed after creating a diagnostic figure")
        if self.signal_outcome == "stopped":
            context.cancel_signal.set()
        elif self.signal_outcome is not None:
            context.cancel_signal.set_error(
                self.signal_outcome, "Acquisition failed", OSError("Device unavailable")
            )
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
        _, axes = plots.subplots("fit")
        axes.plot([0.0], [analysis])
        if self.fail_analysis:
            raise ValueError("Analysis failed after creating a diagnostic figure")
        return analysis

    def save(
        self,
        source: RunRecord[Cfg, float],
        destination: Path,
        *,
        comment: str | None = None,
        tag: str | None = None,
    ) -> None:
        """Create a local text file exclusively and retain the supplied source."""
        with destination.open("x", encoding="utf-8") as file:
            file.write(str(source.result))
        self.saved.append((source, destination))
        self.metadata = (comment, tag)

    def load(self, source: Path) -> RunRecord[Cfg, float]:
        """Return a fixed record; the path named missing raises FileNotFoundError."""
        if source.name == "missing":
            raise FileNotFoundError(source)
        return RunRecord(cfg=Cfg(scale=5.0), result=5.0)
