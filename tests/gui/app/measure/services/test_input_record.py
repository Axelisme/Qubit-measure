"""Adapter input stays separate from acquired, saved and restored experiment cfg."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
)
from zcu_tools.gui.app.measure.artifacts import SaveService
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.guard import GuardService
from zcu_tools.gui.app.measure.services.load import LoadService
from zcu_tools.gui.app.measure.services.operation_gate import OperationGate
from zcu_tools.gui.app.measure.services.run import RunService
from zcu_tools.gui.app.measure.state import Session, SessionEnv, State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import CommandOrigin, PointUseRequest
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.app.measure._form_fakes import make_form
from tests.gui.app.measure._record_support import require_record


class OperatorInput(ConfigBase):
    amplitude_percent: float = 25.0


def input_form(context: FormContext) -> Form[OperatorInput]:
    return Form(OperatorInput, {})


@dataclass(frozen=True)
class ObservedGain:
    value: float | None


def analyze_record(
    req: AnalyzeRequest[RunRecord[SavingCfg, SavingResult], None], *, plots: Plots
) -> ObservedGain:
    return ObservedGain(None if req.run_result.cfg is None else req.run_result.cfg.gain)


@dataclass
class RunHost:
    state: State
    service: RunService
    background: BackgroundRunner
    context: ContextControlFacet
    bus: BaseEventBus
    handles: OperationHandles


@pytest.fixture
def host(tmp_path: Path, qapp: QApplication) -> Iterator[RunHost]:
    entry = ResultEntry.create(
        "synthetic",
        result_root=tmp_path / "results",
        database_root=tmp_path / "database",
    )
    entry.new_point("p")
    state = State(
        SessionEnv(
            parameters=None, soc=MagicMock(), soccfg=MagicMock(dump_cfg=lambda: "{}")
        )
    )
    bus = BaseEventBus()
    context = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
    context.install_entry(entry)
    control = ContextControlFacet(context=context)
    control.use_point(PointUseRequest("p"))
    background = BackgroundRunner()
    handles = OperationHandles()
    gate = OperationGate(bus)
    runner = OperationRunner(
        gate, handles, ProgressService(DirectProgressTransport()), background, bus
    )
    service = RunService(
        state,
        runner,
        bus,
        handles,
        MagicMock(),
        gate=gate,
        devices=lambda: {},
        refresh_devices=lambda: None,
    )
    try:
        yield RunHost(state, service, background, control, bus, handles)
    finally:
        assert background.quiesce()
        service.release_view_plots("tab")


@pytest.mark.parametrize("cfg_available", [True, False], ids=["actual-cfg", "cfg-none"])
def test_host_retains_adapter_record_and_native_load_never_refills_input(
    host: RunHost, cfg_available: bool
) -> None:
    returned: list[RunRecord[SavingCfg, SavingResult]] = []

    def run(
        inp: OperatorInput, *, context: RunContext
    ) -> RunRecord[SavingCfg, SavingResult]:
        actual = SavingCfg(gain=inp.amplitude_percent / 100.0)
        record = RunRecord(
            cfg=actual if cfg_available else None,
            result=SavingCore().run(actual, context=context),
        )
        returned.append(record)
        return record

    definition = Adapter(
        spec=SavingCore.AXES_SPEC,
        make_form=input_form,
        run=run,
        guide=None,
        analysis=Analysis(fn=analyze_record),
    )
    registry = Registry()
    registry.register("converted", definition)
    state = host.state
    resource = make_form(input_form(FormContext(None)), state=state)
    state.add_tab(
        "tab",
        Session(
            adapter_name="converted", adapter=registry.create("converted"), cfg=resource
        ),
    )
    token = host.service.start_run(
        GuardService(state).capture_run_inputs("tab"), plots=Plots(NonPresentingHost())
    )
    assert host.background.quiesce()
    outcome = host.handles.poll(token)
    assert outcome is not None and outcome.status == "finished", outcome
    acquired = require_record(state.get_tab("tab").run)
    assert acquired.cfg == returned[0].cfg
    assert acquired.result is returned[0].result
    plots = Plots(NonPresentingHost())
    try:
        expected = ObservedGain(0.25 if cfg_available else None)
        assert (
            definition.analyze(AnalyzeRequest(acquired, None, None), plots=plots)
            == expected
        )
        if not cfg_available:
            return
        resource.write({("amplitude_percent",): 90.0})
        report = SaveService(state, host.bus).save_data(
            "tab", origin=CommandOrigin("gui")
        )
        native = next(
            item for item in report.outputs if item.output.format == "data_h5"
        )
        assert native.status == "saved", native.error
        loader = LoadService(
            state,
            MagicMock(),
            context=host.context,
            captured_parameters=lambda aliases: host.context.read_context(
                aliases=aliases
            ),
        )
        loader.load_result("tab", str(native.output.path))
        restored = require_record(state.get_tab("tab").run)
        assert restored.cfg == SavingCfg(gain=0.25)
        assert (
            definition.analyze(AnalyzeRequest(restored, None, None), plots=plots)
            == expected
        )
        assert resource.input.data["amplitude_percent"] == 90.0
    finally:
        plots.finish()
        plots.release()
