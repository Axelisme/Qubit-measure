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
        self.fail_run = False
        self.contexts: list[QickContext] = []
        self.saved: list[tuple[RunRecord[_Cfg, float], Path]] = []
        self.metadata: tuple[str | None, str | None, str | None, int] | None = None

    def run(self, config: _Cfg, *, context: QickContext) -> float:
        self.contexts.append(context)
        result = config.scale
        _, axes = context.plots.subplots("raw")
        axes.plot([0.0], [result])
        config.scale = 99.0
        if self.fail_run:
            raise ValueError("Run failed after creating a diagnostic figure")
        return result

    def analyze(
        self,
        source: RunRecord[_Cfg, float],
        options: _Options,
        *,
        plots: Plots,
    ) -> float:
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
        source: RunRecord[_Cfg, float],
        destination: Path,
        *,
        comment: str | None = None,
        tag: str | None = None,
        server_ip: str | None = None,
        port: int = 4999,
    ) -> None:
        with destination.open("x", encoding="utf-8") as file:
            file.write(str(source.result))
        self.saved.append((source, destination))
        self.metadata = (comment, tag, server_ip, port)

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
        self.fail_release = False
        self.released: list[Figure] = []

    def present(self, figure: Figure) -> None:
        if self.fail_present:
            raise ValueError("Presentation failed")

    def release(self, figure: Figure) -> None:
        self.released.append(figure)
        if self.fail_release:
            raise OSError("Presentation cleanup failed")


def test_analysis_returns_explicit_source_and_isolates_working_options() -> None:
    adapter = NotebookAdapter(_Core(), host=NonPresentingHost())
    cfg = _Cfg(scale=3.0)
    source = RunRecord(cfg=cfg, result=3.0)
    cfg.scale = 11.0
    options = _Options(weights=[2.0])

    record = adapter.analyze(options, source=source)
    options.weights[0] = 7.0

    assert record.source is source
    assert source.cfg == _Cfg(scale=3.0)
    assert cfg.scale == 11.0
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


def test_analyze_requires_a_source_before_starting_operation() -> None:
    adapter = NotebookAdapter(_Core(), host=_Host())

    with pytest.raises(ValueError, match="No run record"):
        adapter.analyze(_Options(weights=[2.0]))

    assert adapter.analysis is None
    assert adapter.analysis_presentation is None


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


def test_save_uses_explicit_source_and_returns_exact_or_unique_path(
    tmp_path: Path,
) -> None:
    core = _Core()
    adapter = NotebookAdapter(core, host=_Host())
    source = RunRecord[_Cfg, float](cfg=None, result=3.0)
    current = adapter.load(Path("loaded"))
    analysis = adapter.analyze(_Options(weights=[2.0]))
    destination = tmp_path / "data.hdf5"
    destination.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        adapter.save(source, destination)
    assert destination.read_text(encoding="utf-8") == "existing"

    unique = adapter.save(
        source,
        destination,
        unique=True,
        comment="measurement A",
        tag="trial",
        server_ip="example.invalid",
        port=8123,
    )
    assert unique == tmp_path / "data_1.hdf5"
    assert unique.read_text(encoding="utf-8") == "3.0"
    assert core.saved == [(source, unique)]
    assert core.metadata == ("measurement A", "trial", "example.invalid", 8123)

    exact = adapter.save(source, tmp_path / "exact.h5")
    assert exact == tmp_path / "exact.hdf5"
    assert exact.read_text(encoding="utf-8") == "3.0"
    assert core.saved[-1] == (source, exact)
    assert adapter.last_run is current
    assert adapter.analysis is analysis


@pytest.mark.parametrize("operation", ["run", "analyze"])
def test_producer_and_cleanup_failures_remain_observable(operation: str) -> None:
    core = _Core()
    host = _Host()
    adapter = NotebookAdapter(core, soc=object(), soccfg=object(), host=host)
    previous_run = adapter.run(_Cfg(scale=3.0))
    previous_analysis = adapter.analyze(_Options(weights=[2.0]))
    core.fail_run = operation == "run"
    core.fail_analysis = operation == "analyze"
    host.fail_release = True

    action = (
        (lambda: adapter.run(_Cfg(scale=9.0)))
        if operation == "run"
        else (lambda: adapter.analyze(_Options(weights=[3.0])))
    )
    with pytest.raises(ExceptionGroup) as errors:
        action()

    assert errors.value.subgroup(ValueError) is not None
    assert errors.value.subgroup(OSError) is not None
    assert adapter.last_run is previous_run
    assert adapter.analysis is previous_analysis
    assert len(host.released) == 1


@pytest.mark.parametrize("missing", ["soc", "soccfg", "both"])
def test_run_requires_handles_before_starting_operation(missing: str) -> None:
    core = _Core()
    adapter = NotebookAdapter(
        core,
        soc=object() if missing == "soccfg" else None,
        soccfg=object() if missing == "soc" else None,
        host=_Host(),
    )

    with pytest.raises(ValueError, match="soc and soccfg"):
        adapter.run(_Cfg(scale=3.0))

    assert core.contexts == []
    assert adapter.last_run is None
    assert adapter.run_presentation is None


@pytest.mark.parametrize("failure", ["core", "finish"])
def test_failed_run_preserves_current_run_and_analysis(failure: str) -> None:
    core = _Core()
    host = _Host()
    adapter = NotebookAdapter(core, soc=object(), soccfg=object(), host=host)
    previous_run = adapter.run(_Cfg(scale=3.0))
    previous_run_presentation = adapter.run_presentation
    previous_analysis = adapter.analyze(_Options(weights=[2.0]))
    previous_analysis_presentation = adapter.analysis_presentation
    core.fail_run = failure == "core"
    host.fail_present = failure == "finish"
    cfg = _Cfg(scale=9.0)

    with pytest.raises(ValueError, match="failed"):
        adapter.run(cfg)

    assert cfg.scale == 9.0
    assert adapter.last_run is previous_run
    assert adapter.run_presentation is previous_run_presentation
    assert adapter.analysis is previous_analysis
    assert adapter.analysis_presentation is previous_analysis_presentation
    assert len(host.released) == 1
    np.testing.assert_array_equal(host.released[0].axes[0].lines[0].get_ydata(), [9.0])
    np.testing.assert_array_equal(
        previous_analysis.figures["fit"].axes[0].lines[0].get_ydata(), [6.0]
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
