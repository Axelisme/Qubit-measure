"""Headless analysis of caller-constructed records through the public stage."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import probe
from zcu_tools.experiment.base import Experiment
from zcu_tools.experiment.records import RunRecord
from zcu_tools.plotting.plots import Plots


@dataclass
class Analysis:
    mean: float


class SyntheticExperiment(Experiment[np.ndarray, ConfigBase]):
    def analyze(self, source, options, *, plots: Plots) -> Analysis:
        _, axes = plots.subplots("fit/trace", figsize=(3, 2))
        axes.plot(source.result * options)
        return Analysis(float(np.mean(source.result) * options))


def test_analysis_returns_actual_value_and_png_bytes():
    source = RunRecord(ConfigBase(), np.array([1.0, 2.0, 3.0]))
    result = probe.analysis_stage(SyntheticExperiment(), source, 2.0)
    assert result.analysis == Analysis(4.0)
    assert set(result.pngs) == {"analysis-fit%2Ftrace.png"}
    with Image.open(BytesIO(result.pngs["analysis-fit%2Ftrace.png"])) as image:
        assert image.format == "PNG"
        assert image.size == (300, 200)
