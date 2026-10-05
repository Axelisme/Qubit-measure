"""stub experiment core."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg


@dataclass(frozen=True)
class FakeResult:
    data: NDArray[np.float64]


class FakeExpCfg(ExpCfgModel):
    reps: int = 100
    rounds: int = 10
    gain: float = 0.1
    noise_scale: float = 0.1
    sweep: SweepCfg


FakeRunResult: TypeAlias = RunRecord[FakeExpCfg, FakeResult]


@dataclass(frozen=True)
class FakeAnalyzeOptions:
    threshold: float = 0.5


@dataclass(frozen=True)
class FakeAnalysis:
    peak: float


class FakeExp:
    """Fixed seeded harness with explicit records and intentionally inert save."""

    def run(self, config: FakeExpCfg, *, context: RunContext) -> FakeResult:
        del context  # The fixed harness has no hardware or live Run presentation.
        rng = np.random.default_rng(seed=42)
        signals = rng.normal(0.0, config.noise_scale, size=11)
        return FakeResult(data=signals)

    def analyze(
        self, source: FakeRunResult, options: FakeAnalyzeOptions, *, plots: Plots
    ) -> FakeAnalysis:
        threshold = options.threshold
        data = source.result.data
        peak = float(np.max(np.abs(data)))
        _, ax = plots.subplots("fit")
        xs = np.arange(len(data))
        ax.plot(xs, data, label="signal")
        if peak > threshold:
            idx = int(np.argmax(np.abs(data)))
            ax.axvline(idx, color="red", linestyle="--", label=f"peak={peak:.3f}")
        ax.axhline(
            threshold, color="gray", linestyle=":", label=f"threshold={threshold}"
        )
        ax.set_title("FakeAdapter analysis")
        ax.legend()
        return FakeAnalysis(peak=peak)

    def save(
        self,
        source: FakeRunResult,
        destination: Path,
        *,
        comment: str | None = None,
        tag: str | None = None,
    ) -> None:
        """The no-hardware harness intentionally leaves data persistence inert."""
        _ = source, destination, comment, tag

    def load(self, source: Path) -> FakeRunResult:
        del source
        raise NotImplementedError(
            "The inert FakeAdapter harness has no data file format"
        )
