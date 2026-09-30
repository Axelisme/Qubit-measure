"""Behavior of the common Notebook record and operation ownership seam."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
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
    def __init__(self) -> None:
        self.fail_analysis = False
        self.contexts: list[QickContext] = []

    def run(self, cfg: _Cfg, *, context: QickContext) -> float:
        self.contexts.append(context)
        result = cfg.scale
        _, axes = context.plots.subplots("raw")
        axes.plot([0.0], [result])
        cfg.scale = 99.0
        return result

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
        if self.fail_analysis:
            raise ValueError("Analysis failed after creating a diagnostic figure")
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


class _Host(NonPresentingHost):
    def __init__(self) -> None:
        self.fail_present = False
        self.released: list[Figure] = []

    def present(self, figure: Figure) -> None:
        if self.fail_present:
            raise ValueError("Presentation failed")

    def release(self, figure: Figure) -> None:
        self.released.append(figure)


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


def test_run_isolates_config_and_replaces_only_current_records() -> None:
    core = _Core()
    soc, soccfg = object(), object()
    adapter = NotebookAdapter(core, soc=soc, soccfg=soccfg, host=_Host())
    cfg = _Cfg(scale=3.0)
    first = adapter.run(cfg)
    previous = adapter.analyze(_Options(weights=[2.0]))
    cfg.scale = 7.0

    second = adapter.run(cfg)

    assert first.cfg == _Cfg(scale=3.0)
    assert first.result == 3.0
    assert second.cfg == _Cfg(scale=7.0)
    assert second.result == 7.0
    assert cfg.scale == 7.0
    assert adapter.last_run is second
    assert adapter.analysis is None
    assert adapter.analysis_presentation is None
    first_context, second_context = core.contexts
    assert first_context is not second_context
    assert first_context.plots is not second_context.plots
    assert first_context.soc is soc and second_context.soc is soc
    assert first_context.soccfg is soccfg and second_context.soccfg is soccfg
    assert adapter.run_presentation is second_context.plots
    np.testing.assert_array_equal(
        first_context.plots["raw"].axes[0].lines[0].get_ydata(), [3.0]
    )
    np.testing.assert_array_equal(
        previous.figures["fit"].axes[0].lines[0].get_ydata(), [6.0]
    )


@pytest.mark.parametrize("failure", ["core", "finish"])
def test_failed_analysis_preserves_success_and_releases_diagnostic(
    failure: str,
) -> None:
    core = _Core()
    host = _Host()
    adapter = NotebookAdapter(core, host=host)
    source = adapter.load(Path("loaded"))
    previous = adapter.analyze(_Options(weights=[2.0]))
    previous_presentation = adapter.analysis_presentation
    core.fail_analysis = failure == "core"
    host.fail_present = failure == "finish"

    with pytest.raises(ValueError, match="failed"):
        adapter.analyze(_Options(weights=[3.0]))

    assert adapter.last_run is source
    assert adapter.analysis is previous
    assert adapter.analysis_presentation is previous_presentation
    assert len(host.released) == 1
    diagnostic = host.released[0]
    assert diagnostic is not previous.figures["fit"]
    np.testing.assert_array_equal(diagnostic.axes[0].lines[0].get_ydata(), [15.0])
    np.testing.assert_array_equal(
        previous.figures["fit"].axes[0].lines[0].get_ydata(), [10.0]
    )
