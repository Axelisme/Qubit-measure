"""Headless analysis of caller-constructed records through the public stage."""

from collections.abc import Generator
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pytest
from matplotlib import rc_context
from PIL import Image
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import probe
from zcu_tools.experiment.base import Experiment
from zcu_tools.experiment.records import RunRecord
from zcu_tools.plotting.plots import Plots


@pytest.fixture(autouse=True)
def isolated_matplotlib_style() -> Generator[None]:
    with rc_context():
        yield


@dataclass
class Analysis:
    mean: float


class SyntheticExperiment(Experiment[np.ndarray, ConfigBase]):
    def analyze(self, source, options, *, plots: Plots) -> Analysis:
        _, axes = plots.subplots(num="fit/trace", figsize=(3, 2))
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


def test_analysis_propagates_original_exception():
    error = RuntimeError("analyze failed")

    class FailingExperiment(SyntheticExperiment):
        def analyze(self, source, options, *, plots: Plots) -> Analysis:
            raise error

    with pytest.raises(RuntimeError) as caught:
        probe.analysis_stage(
            FailingExperiment(), RunRecord(None, np.array([1.0])), None
        )
    assert caught.value is error
