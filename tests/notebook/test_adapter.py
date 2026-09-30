"""Behavior of the common Notebook record and operation ownership seam."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import QickContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.plotting.plots import NonPresentingHost, Plots


class _Cfg(ExpCfgModel):
    scale: float = 1.0


@dataclass
class _Options:
    weights: list[float]


class _Core:
    def run(self, cfg: _Cfg, *, context: QickContext) -> float:
        return cfg.scale

    def analyze(
        self,
        source: RunRecord[_Cfg, float],
        options: _Options,
        *,
        plots: Plots,
    ) -> float:
        analysis = source.result * options.weights[0]
        options.weights[0] = 99.0
        _, axes = plots.subplots("fit")
        axes.plot([0.0], [analysis])
        return analysis

    def save(
        self,
        source: RunRecord[_Cfg, float],
        destination: Path,
        *,
        comment: str | None = None,
        tag: str | None = None,
        server_ip: str | None = None,
        port: int = 4999,
    ) -> None:
        raise NotImplementedError("Persistence is not part of this collaborator")

    def load(
        self,
        source: Path,
        *,
        server_ip: str | None = None,
        port: int = 4999,
    ) -> RunRecord[_Cfg, float]:
        if source.name == "missing":
            raise FileNotFoundError(source)
        return RunRecord(cfg=_Cfg(scale=5.0), result=5.0)


def test_analysis_returns_explicit_source_and_isolates_working_options() -> None:
    adapter = NotebookAdapter(_Core(), host=NonPresentingHost())
    source = RunRecord(cfg=_Cfg(scale=3.0), result=3.0)
    options = _Options(weights=[2.0])

    record = adapter.analyze(options, source=source)
    options.weights[0] = 7.0

    assert record.source is source
    assert record.options.weights == [2.0]
    assert record.result == 6.0
    assert adapter.analysis is record
    assert adapter.last_run is None
    np.testing.assert_array_equal(
        record.figures["fit"].axes[0].lines[0].get_ydata(), [6.0]
    )


def test_successful_load_clears_analysis_but_old_source_stays_explicit() -> None:
    adapter = NotebookAdapter(_Core(), host=NonPresentingHost())
    source = RunRecord[_Cfg, float](cfg=None, result=3.0)
    previous = adapter.analyze(_Options(weights=[2.0]), source=source)

    loaded = adapter.load(Path("loaded"))

    assert adapter.last_run is loaded
    assert adapter.analysis is None
    assert adapter.analysis_presentation is None
    np.testing.assert_array_equal(
        previous.figures["fit"].axes[0].lines[0].get_ydata(), [6.0]
    )
    selected = adapter.analyze(_Options(weights=[3.0]), source=source)
    assert selected.source is source
    assert selected.result == 9.0
    assert adapter.last_run is loaded

    with pytest.raises(FileNotFoundError):
        adapter.load(Path("missing"))
    assert adapter.last_run is loaded
    assert adapter.analysis is selected

    latest = adapter.analyze(_Options(weights=[2.0]))
    assert latest.source is loaded
    assert latest.result == 10.0
