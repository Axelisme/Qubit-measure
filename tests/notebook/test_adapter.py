"""Behavior of the common Notebook record and operation ownership seam."""

from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import assert_type
from unittest.mock import Mock

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.device import DeviceManager, FakeDevice
from zcu_tools.experiment import AcquisitionEvidence
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.stop_signal import ScheduleOutcomeError
from zcu_tools.notebook import NotebookAdapter, NotebookExperiment
from zcu_tools.notebook import adapter as adapter_module
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.device._snapshot_fakes import SnapshotDevice
from tests.experiment._saving_support import SavingCfg
from tests.experiment._snapshot_support import SnapshotCore
from tests.notebook._adapter_fakes import Cfg as _Cfg
from tests.notebook._adapter_fakes import Options as _Options
from tests.notebook._adapter_fakes import RecordingCore as _Core


def test_run_captures_devices_before_core_and_freezes_evidence() -> None:
    flux, unused = SnapshotDevice(), SnapshotDevice()
    flux.set_value(0.25)
    unused.set_value(0.6)
    flux.events.clear()
    unused.events.clear()
    manager = DeviceManager()
    manager.register_device("flux", flux)
    manager.register_device("unused", unused)
    adapter = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=manager, host=NonPresentingHost()
    )(SnapshotCore())
    record = adapter.run(SavingCfg())
    assert flux.events == ["read", "write"]
    assert unused.events == ["read"]
    evidence = adapter.acquisition_for(record)
    assert evidence.device_snapshot is not None
    assert evidence.device_snapshot["flux"]["value"] == 0.25
    assert evidence.device_snapshot["unused"]["value"] == 0.6
    flux.set_value(0.9)
    evidence.device_snapshot["flux"]["value"] = 9.0
    retained = adapter.acquisition_for(record)
    assert retained.device_snapshot is not None
    assert retained.device_snapshot["flux"]["value"] == 0.25
    assert flux.events.count("read") == 1


def test_device_read_failure_retains_previous_notebook_acquisition() -> None:
    flux = SnapshotDevice()
    manager = DeviceManager()
    manager.register_device("flux", flux)
    adapter = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=manager, host=NonPresentingHost()
    )(SnapshotCore())
    previous = adapter.run(SavingCfg())
    evidence = adapter.acquisition_for(previous)
    flux.events.clear()
    failure = OSError("snapshot unavailable")
    flux.read_failure = failure
    with pytest.raises(OSError, match="snapshot unavailable") as caught:
        adapter.run(SavingCfg())
    assert caught.value is failure
    assert flux.events == ["read"]
    assert adapter.last_run is previous
    assert adapter.acquisition_for(previous) == evidence


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


def test_nullable_analysis_options_are_delegated_and_retained() -> None:
    received: list[_Options | None] = []

    class NullableCore(_Core):
        def analyze(
            self,
            source: RunRecord[_Cfg, float],
            options: _Options | None,
            *,
            plots: Plots,
        ) -> float:
            received.append(options)
            if options is None:
                return source.result
            return super().analyze(source, options, plots=plots)

    source = RunRecord[_Cfg, float](cfg=None, result=3.0)
    adapter = NotebookAdapter(host=NonPresentingHost())(NullableCore())

    record = adapter.analyze(None, source=source)

    assert received == [None]
    assert record.options is None
    assert record.source is source
    assert record.result == source.result
    assert adapter.analysis is record
    assert adapter.last_run is None


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
    core.contexts[0].scope.stop.set_error("failed", "old run", None)

    first_evidence = adapter.acquisition_for(first)
    second = adapter.run(cfg)
    second_evidence = adapter.acquisition_for(second)
    assert second_evidence.run_id != first_evidence.run_id
    with pytest.raises(ValueError, match="acquisition"):
        adapter.acquisition_for(first)

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
    assert first_context.scope.stop is not second_context.scope.stop
    assert not second_context.scope.stop.is_set()
    assert second_context.scope.stop.error is None
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


@pytest.mark.parametrize("failure", ["core", "finish", "signal-failed"])
def test_failed_run_preserves_current_run_and_analysis(failure: str) -> None:
    core = _Core()
    host = _Host()
    adapter = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=DeviceManager(), host=host
    )(core)
    previous_run = adapter.run(_Cfg(scale=3.0))
    previous_run_presentation = adapter.run_presentation
    previous_evidence = adapter.acquisition_for(previous_run)
    previous_analysis = adapter.analyze(_Options(weights=[2.0]))
    previous_analysis_presentation = adapter.analysis_presentation
    core.fail_run = failure == "core"
    host.fail_present = failure == "finish"
    if failure.startswith("signal-"):
        core.signal_outcome = "failed"
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
    assert adapter.acquisition_for(previous_run) == previous_evidence
    assert adapter.run_presentation is previous_run_presentation
    assert adapter.analysis is previous_analysis
    assert adapter.analysis_presentation is previous_analysis_presentation
    assert len(host.released) == 1
    np.testing.assert_array_equal(host.released[0].axes[0].lines[0].get_ydata(), [9.0])
    np.testing.assert_array_equal(
        previous_analysis.figures["fit"].axes[0].lines[0].get_ydata(), [6.0]
    )


@pytest.mark.parametrize("outcome", ["stopped", "interrupted"])
def test_stopped_partial_run_is_committed(outcome: str) -> None:
    core = _Core()
    host = _Host()
    adapter = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=DeviceManager(), host=host
    )(core)
    previous = adapter.run(_Cfg(scale=5.0))
    core.signal_outcome = "stopped" if outcome == "stopped" else "interrupted"

    partial = adapter.run(_Cfg(scale=9.0))

    assert partial is adapter.last_run
    assert partial is not previous
    assert partial.result == 9.0
    assert adapter.acquisition_for(partial).completion == "stopped"
    assert core.contexts[-1].scope.stop.is_set()
    assert host.released == []


@pytest.mark.parametrize("failure", ["core", "finish"])
def test_failed_analysis_preserves_success_and_releases_diagnostic(
    failure: str,
) -> None:
    core = _Core()
    host = _Host()
    adapter = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=DeviceManager(), host=host
    )(core)
    source = adapter.run(_Cfg(scale=5.0))
    evidence = adapter.acquisition_for(source)
    previous = adapter.analyze(_Options(weights=[2.0]))
    previous_presentation = adapter.analysis_presentation
    core.fail_analysis = failure == "core"
    host.fail_present = failure == "finish"

    with pytest.raises(ValueError, match="failed"):
        adapter.analyze(_Options(weights=[3.0]))

    assert adapter.last_run is source
    assert adapter.acquisition_for(source) == evidence
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
    assert second.last_run is None
    assert second.analysis is None

    second_run = second.run(_Cfg(scale=7.0))
    second_analysis = second.analyze(_Options(weights=[2.0]))
    first.run(_Cfg(scale=5.0))

    assert first.last_run is not first_run
    assert first.analysis is None
    assert first_analysis.source is first_run
    assert first_analysis.result == 6.0
    assert second.last_run is second_run
    assert second.analysis is second_analysis
    assert second_analysis.result == 14.0
    assert first_core.contexts[0].plots is not second_core.contexts[0].plots
    assert first_core.contexts[0].scope.stop is not second_core.contexts[0].scope.stop


def test_run_refreshes_driver_mapping_and_keeps_it_fixed_until_return() -> None:
    manager = DeviceManager()
    first_device, replacement = FakeDevice(fast_mode=True), FakeDevice(fast_mode=True)
    manager.register_device("flux", first_device)

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
    first = entry.run(_Cfg(scale=2.0))
    second = entry.run(_Cfg(scale=3.0))

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


def test_mapping_failure_preserves_successful_records() -> None:
    manager = Mock(spec=DeviceManager)
    manager.get_all_devices.side_effect = [
        {},
        RuntimeError("Registry snapshot unavailable"),
    ]
    core = _Core()
    entry = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=manager, host=_Host()
    )(core)
    previous = entry.run(_Cfg())
    analysis = entry.analyze(_Options(weights=[2.0]))
    presentation = entry.run_presentation

    with pytest.raises(RuntimeError, match="Registry snapshot unavailable"):
        entry.run(_Cfg(scale=9.0))

    assert entry.last_run is previous
    assert entry.analysis is analysis
    assert entry.run_presentation is presentation
    assert len(core.contexts) == 1


def test_acquisition_boundaries_exclude_presentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = datetime(2026, 10, 6, 12, tzinfo=UTC)
    finished = started + timedelta(seconds=10)
    current = started

    class Clock:
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            assert tz is UTC
            return current

    class TimedCore(_Core):
        def run(self, config: _Cfg, *, context: RunContext) -> float:
            nonlocal current
            assert current == started
            result = super().run(config, context=context)
            current = finished
            return result

    class TimedHost(_Host):
        def present(self, figure: Figure) -> None:
            nonlocal current
            current = finished + timedelta(seconds=20)
            super().present(figure)

    monkeypatch.setattr(adapter_module, "datetime", Clock)
    adapter = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=DeviceManager(), host=TimedHost()
    )(TimedCore())

    record = adapter.run(_Cfg(scale=3.0))
    evidence = adapter.acquisition_for(record)
    assert_type(evidence, AcquisitionEvidence)
    assert evidence.started_at == started
    assert evidence.finished_at == finished
    assert evidence.completion == "complete"
    assert evidence.run_id.startswith("20261006T120000Z-")
    assert current > evidence.finished_at
    adapter.analyze(_Options(weights=[2.0]))
    assert adapter.acquisition_for(record) == evidence


def test_acquisition_identity_rejects_foreign_and_handmade_sources() -> None:
    factory = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=DeviceManager(), host=_Host()
    )
    adapter, other = factory(_Core()), factory(_Core())
    foreign = other.run(_Cfg())
    with pytest.raises(ValueError, match="acquisition"):
        adapter.acquisition_for(foreign)

    record = adapter.run(_Cfg(scale=3.0))
    handmade = RunRecord(cfg=record.cfg, result=record.result)
    assert handmade == record
    for invalid in (foreign, handmade):
        with pytest.raises(ValueError, match="acquisition"):
            adapter.acquisition_for(invalid)


def test_acquisition_isolates_query_version_mappings() -> None:
    adapter = NotebookAdapter(
        soc=object(), soccfg=object(), device_manager=DeviceManager(), host=_Host()
    )(_Core())
    record = adapter.run(_Cfg())
    first = adapter.acquisition_for(record)
    original_versions = dict(first.provenance.software_versions)
    queried = first.provenance.software_versions
    assert isinstance(queried, dict)
    queried["python"] = "query changed"

    second = adapter.acquisition_for(record)
    assert second is not first
    assert second.provenance.software_versions == original_versions


def test_capture_failure_retains_success_without_starting_core() -> None:
    core = _Core()
    soccfg = Mock()
    soccfg.dump_cfg.side_effect = ["{}", OSError("Software capture failed")]
    adapter = NotebookAdapter(
        soc=object(), soccfg=soccfg, device_manager=DeviceManager(), host=_Host()
    )(core)
    record = adapter.run(_Cfg())
    evidence = adapter.acquisition_for(record)
    analysis = adapter.analyze(_Options(weights=[2.0]))

    with pytest.raises(OSError, match="Software capture failed"):
        adapter.run(_Cfg(scale=9.0))

    assert len(core.contexts) == 1
    assert adapter.last_run is record
    assert adapter.analysis is analysis
    assert adapter.acquisition_for(record) == evidence


@pytest.mark.parametrize("unique", [False, True])
def test_save_without_a_record_fails_before_creating_a_file(
    tmp_path: Path, unique: bool
) -> None:
    core = _Core()
    entry = NotebookAdapter(host=_Host())(core)
    with pytest.raises(ValueError, match="No run record to save"):
        entry.save(tmp_path / "missing", unique=unique)

    assert list(tmp_path.iterdir()) == []
    assert entry.last_run is None
