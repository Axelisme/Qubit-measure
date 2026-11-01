"""Synthetic experiment shared by probe API and import-by-name entry tests."""

from typing import Any

import numpy as np
from zcu_tools.experiment.base import Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.plotting.plots import Plots

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult


class ProbeExperiment(Experiment[SavingResult, SavingCfg]):
    """Publish one acquisition figure and analyze restored synthetic samples."""

    AXES_SPEC = SavingCore.AXES_SPEC

    def run(self, config: SavingCfg, *, context: RunContext) -> SavingResult:
        result = SavingCore().run(config, context=context)
        result.signal[-1] = complex(np.nan, 0)
        figure, axes = context.plots.subplots(num="signal/trace")
        axes.plot(result.frequency, result.signal.real)
        figure.show()
        return result

    def analyze(
        self, source: RunRecord[SavingCfg, SavingResult], options: Any, *, plots: Plots
    ) -> dict[str, object]:
        if source.cfg is None:
            raise ValueError("Analysis requires the restored record")
        _, axes = plots.subplots(num="fit")
        axes.plot(source.result.frequency, source.result.signal.imag)
        return {"gain": source.cfg.gain, "options": options}
