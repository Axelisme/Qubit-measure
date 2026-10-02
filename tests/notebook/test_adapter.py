"""Behavior of the common Notebook record and operation ownership seam."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn, assert_type
from unittest.mock import Mock

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.device import DeviceManager, FakeDevice
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import AnalysisRecord, RunRecord
from zcu_tools.experiment.stop_signal import ScheduleOutcomeError
from zcu_tools.notebook import NotebookAdapter, NotebookExperiment
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
        self.signal_outcome: Literal["stopped", "failed", "interrupted"] | None = None
        self.contexts: list[RunContext] = []
        self.saved: list[tuple[RunRecord[_Cfg, float], Path]] = []
        self.metadata: tuple[str | None, str | None] | None = None

    def run(self, config: _Cfg, *, context: RunContext) -> float:
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
    ) -> None:
        with destination.open("x", encoding="utf-8") as file:
            file.write(str(source.result))
        self.saved.append((source, destination))
        self.metadata = (comment, tag)

    def load(self, source: Path) -> RunRecord[_Cfg, float]:
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
    adapter = NotebookAdapter(host=NonPresentingHost())(_Core())
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
    adapter = NotebookAdapter(host=NonPresentingHost())(_Core())
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
    adapter = NotebookAdapter(host=_Host())(_Core())

    with pytest.raises(ValueError, match="No run record"):
        adapter.analyze(_Options(weights=[2.0]))

    assert adapter.analysis is None
    assert adapter.analysis_presentation is None


def test_run_isolates_config_and_replaces_only_current_records() -> None:
    core = _Core()
    soc, soccfg = object(), object()
    manager = DeviceManager()
    first_device = FakeDevice(fast_mode=True)
    manager.register_device("flux", first_device)
    adapter = NotebookAdapter(
        soc=soc, soccfg=soccfg, device_manager=manager, host=_Host()
    )(core)
    cfg = _Cfg(scale=3.0)
    first = adapter.run(cfg)
    previous = adapter.analyze(_Options(weights=[2.0]))
    cfg.scale = 7.0
    second_device = FakeDevice(fast_mode=True)
    manager.drop_device("flux")
    manager.register_device("flux", second_device)
    core.contexts[0].cancel_signal.set_error("failed", "old run", None)

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
    assert first_context.cancel_signal is not second_context.cancel_signal
    assert not second_context.cancel_signal.is_set()
    assert second_context.cancel_signal.error is None
    assert first_context.devices["flux"] is first_device
    assert second_context.devices["flux"] is second_device
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
    adapter = NotebookAdapter(host=_Host())(core)
    source = RunRecord[_Cfg, float](cfg=None, result=3.0)
    current = adapter.load(Path("loaded"))
    analysis = adapter.analyze(_Options(weights=[2.0]))
    destination = tmp_path / "data.hdf5"
    destination.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        adapter.save(destination, source=source)
    assert destination.read_text(encoding="utf-8") == "existing"

    unique = adapter.save(
        destination, source=source, unique=True, comment="measurement A", tag="trial"
    )
    assert unique == tmp_path / "data_1.hdf5"
    assert unique.read_text(encoding="utf-8") == "3.0"
    assert core.saved == [(source, unique)]
    assert core.metadata == ("measurement A", "trial")

    exact = adapter.save(tmp_path / "exact.h5", source=source)
    assert exact == tmp_path / "exact.hdf5"
    assert exact.read_text(encoding="utf-8") == "3.0"
    assert core.saved[-1] == (source, exact)
    assert adapter.last_run is current
    assert adapter.analysis is analysis


@pytest.mark.parametrize("operation", ["run", "analyze"])
def test_producer_and_cleanup_failures_remain_observable(operation: str) -> None:
    core = _Core()
    host = _Host()
    adapter = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=DeviceManager(), host=host
    )(core)
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
        soc=object() if missing == "soccfg" else None,
        soccfg=object() if missing == "soc" else None,
        host=_Host(),
    )(core)

    with pytest.raises(ValueError, match="soc and soccfg"):
        adapter.run(_Cfg(scale=3.0))

    assert core.contexts == []
    assert adapter.last_run is None
    assert adapter.run_presentation is None


def test_run_requires_device_manager_before_starting_operation() -> None:
    core = _Core()
    adapter = NotebookAdapter(soc=object(), soccfg=object(), host=_Host())(core)

    with pytest.raises(ValueError, match="explicit device manager"):
        adapter.run(_Cfg())

    assert core.contexts == []
    assert adapter.last_run is None


@pytest.mark.parametrize(
    "failure", ["core", "finish", "signal-failed", "signal-interrupted"]
)
def test_failed_run_preserves_current_run_and_analysis(failure: str) -> None:
    core = _Core()
    host = _Host()
    adapter = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=DeviceManager(), host=host
    )(core)
    previous_run = adapter.run(_Cfg(scale=3.0))
    previous_run_presentation = adapter.run_presentation
    previous_analysis = adapter.analyze(_Options(weights=[2.0]))
    previous_analysis_presentation = adapter.analysis_presentation
    core.fail_run = failure == "core"
    host.fail_present = failure == "finish"
    if failure.startswith("signal-"):
        core.signal_outcome = "failed" if failure == "signal-failed" else "interrupted"
    cfg = _Cfg(scale=9.0)
    error_type = ScheduleOutcomeError if failure.startswith("signal-") else ValueError

    with pytest.raises(error_type, match="failed") as raised:
        adapter.run(cfg)

    if isinstance(raised.value, ScheduleOutcomeError):
        assert raised.value.status == core.signal_outcome
        assert isinstance(raised.value.__cause__, OSError)
        assert str(raised.value.__cause__) == "Device unavailable"

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


def test_stopped_partial_run_without_error_is_committed() -> None:
    core = _Core()
    host = _Host()
    adapter = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=DeviceManager(), host=host
    )(core)
    previous = adapter.load(Path("loaded"))
    core.signal_outcome = "stopped"

    partial = adapter.run(_Cfg(scale=9.0))

    assert partial is adapter.last_run
    assert partial is not previous
    assert partial.result == 9.0
    assert core.contexts[-1].cancel_signal.is_set()
    assert host.released == []


@pytest.mark.parametrize("failure", ["core", "finish"])
def test_failed_analysis_preserves_success_and_releases_diagnostic(
    failure: str,
) -> None:
    core = _Core()
    host = _Host()
    adapter = NotebookAdapter(host=host)(core)
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


def test_factory_keeps_each_experiment_records_independent() -> None:
    factory = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=DeviceManager(), host=_Host()
    )
    first_core, second_core = _Core(), _Core()
    first, second = factory(first_core), factory(second_core)
    assert_type(first, NotebookExperiment[_Core])

    first_run = first.run(_Cfg(scale=2.0))
    assert_type(first_run, RunRecord[_Cfg, float])
    first_analysis = first.analyze(_Options(weights=[3.0]))
    assert_type(first_analysis, AnalysisRecord[_Cfg, float, _Options, float])
    assert second.last_run is None
    assert second.analysis is None

    second_run = second.run(_Cfg(scale=7.0))
    second_analysis = second.analyze(_Options(weights=[2.0]))
    first.load(Path("loaded"))

    assert first.last_run is not first_run
    assert first.analysis is None
    assert first_analysis.source is first_run
    assert first_analysis.result == 6.0
    assert second.last_run is second_run
    assert second.analysis is second_analysis
    assert second_analysis.result == 14.0
    assert first_core.contexts[0].plots is not second_core.contexts[0].plots
    assert (
        first_core.contexts[0].cancel_signal
        is not second_core.contexts[0].cancel_signal
    )


def test_binding_and_offline_operations_do_not_query_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = DeviceManager()

    def reject_query() -> NoReturn:
        raise AssertionError("Offline operations must not query devices")

    monkeypatch.setattr(manager, "get_all_devices", reject_query)
    monkeypatch.setattr(manager, "get_all_info", reject_query)
    factory = NotebookAdapter(device_manager=manager, host=_Host())
    first, second = factory(_Core()), factory(_Core())
    loaded = first.load(Path("loaded"))
    analysis = first.analyze(_Options(weights=[2.0]))
    saved = first.save(tmp_path / "offline")

    assert analysis.source is loaded
    assert saved.read_text(encoding="utf-8") == "5.0"
    assert second.last_run is None
    assert second.analysis is None


def test_run_refreshes_driver_mapping_once_and_keeps_it_fixed_until_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = DeviceManager()
    first_device, replacement = FakeDevice(fast_mode=True), FakeDevice(fast_mode=True)
    manager.register_device("flux", first_device)
    query = Mock(wraps=manager.get_all_devices)
    monkeypatch.setattr(manager, "get_all_devices", query)

    def reject_info() -> NoReturn:
        raise AssertionError("Run must not refresh cfg from device information")

    monkeypatch.setattr(manager, "get_all_info", reject_info)

    class ReplacingCore(_Core):
        def run(self, config: _Cfg, *, context: RunContext) -> float:
            result = super().run(config, context=context)
            if len(self.contexts) == 1:
                manager.drop_device("flux")
                manager.register_device("flux", replacement)
                assert context.devices["flux"] is first_device
            return result

    core = ReplacingCore()
    entry = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=manager, host=_Host()
    )(core)
    assert query.call_count == 0
    first = entry.run(_Cfg(scale=2.0))
    assert query.call_count == 1
    second = entry.run(_Cfg(scale=3.0))

    assert query.call_count == 2
    assert first.result == 2.0 and second.result == 3.0
    assert core.contexts[0].devices["flux"] is first_device
    assert core.contexts[1].devices["flux"] is replacement
    assert manager.get_device("flux") is replacement


def test_reassigning_environment_variables_does_not_rebind_existing_factory() -> None:
    manager = DeviceManager()
    original_manager = manager
    driver = FakeDevice(fast_mode=True)
    manager.register_device("flux", driver)
    soc, soccfg = object(), object()
    original_soc, original_soccfg = soc, soccfg
    factory = NotebookAdapter(
        soc=soc, soccfg=soccfg, device_manager=manager, host=_Host()
    )

    manager = DeviceManager()
    manager.register_device("flux", FakeDevice(fast_mode=True))
    soc, soccfg = object(), object()
    core = _Core()
    entry = factory(core)
    entry.run(_Cfg())
    context = core.contexts[0]

    assert manager is not original_manager
    assert context.soc is original_soc and context.soc is not soc
    assert context.soccfg is original_soccfg and context.soccfg is not soccfg
    assert context.devices["flux"] is driver


def test_mapping_failure_preserves_successful_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = DeviceManager()
    core = _Core()
    entry = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=manager, host=_Host()
    )(core)
    previous = entry.run(_Cfg())
    analysis = entry.analyze(_Options(weights=[2.0]))
    presentation = entry.run_presentation

    def fail_mapping() -> NoReturn:
        raise RuntimeError("Registry snapshot unavailable")

    monkeypatch.setattr(manager, "get_all_devices", fail_mapping)
    with pytest.raises(RuntimeError, match="Registry snapshot unavailable"):
        entry.run(_Cfg(scale=9.0))

    assert entry.last_run is previous
    assert entry.analysis is analysis
    assert entry.run_presentation is presentation
    assert len(core.contexts) == 1


def test_save_defaults_to_last_run_not_explicit_historical_analysis(
    tmp_path: Path,
) -> None:
    core = _Core()
    entry = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=DeviceManager(), host=_Host()
    )(core)
    first = entry.run(_Cfg(scale=2.0))
    assert entry.save(tmp_path / "run").read_text(encoding="utf-8") == "2.0"
    loaded = entry.load(Path("loaded"))
    analysis = entry.analyze(_Options(weights=[3.0]), source=first)

    latest_path = entry.save(tmp_path / "latest")
    old_path = entry.save(tmp_path / "historical", source=first)

    assert latest_path.read_text(encoding="utf-8") == "5.0"
    assert old_path.read_text(encoding="utf-8") == "2.0"
    assert core.saved[-2][0] is loaded
    assert core.saved[-1][0] is first
    assert entry.last_run is loaded
    assert entry.analysis is analysis
    assert analysis.source is first


@pytest.mark.parametrize("unique", [False, True])
def test_save_without_a_record_fails_before_creating_a_file(
    tmp_path: Path, unique: bool
) -> None:
    core = _Core()
    entry = NotebookAdapter(host=_Host())(core)
    with pytest.raises(ValueError, match="No run record to save"):
        entry.save(tmp_path / "missing", unique=unique)

    assert list(tmp_path.iterdir()) == []
    assert core.saved == []
    source = RunRecord[_Cfg, float](cfg=None, result=3.0)
    saved = entry.save(tmp_path / "explicit", source=source)
    assert saved.read_text(encoding="utf-8") == "3.0"
    assert entry.last_run is None
