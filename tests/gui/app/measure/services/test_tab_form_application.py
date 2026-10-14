"""Measure composition uses one fixed form authority through Run and restoration."""

from collections.abc import Mapping
from typing import ClassVar

import pytest
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import ConfigBase, Expr, PulseCfg, Template
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterCfgDefinition,
    AnalysisMode,
    ExpAdapterProtocol,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload
from zcu_tools.gui.cfg.resource import CfgPreconditionError
from zcu_tools.gui.form import FormValue
from zcu_tools.gui.form_adapter import Form, FormContext

from tests.gui.app.measure.remote._helpers import Fixture


class TraceCfg(ExpCfgModel):
    """One concrete cfg field is both displayed and executed."""

    amount: float = 1.0


class TraceAdapter(BaseAdapter[TraceCfg, TraceCfg]):
    """Return accepted cfg without devices, acquisition or result persistence."""

    form = Form(TraceCfg, {"amount": Expr("general.ext.amount")})
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=False, requires_soc=False
    )

    @classmethod
    def cfg_definition(cls) -> AdapterCfgDefinition:
        raise AssertionError("The tab must use Form, not the old cfg definition")

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return "trace"

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> TraceCfg:
        return TraceCfg.model_validate(raw_cfg)


class FactoryCfg(ConfigBase):
    """A synthetic optional module exposes template binding at tab creation."""

    pulse: PulseCfg | None = None
    amount: float = 1.0


@pytest.mark.parametrize("failure", ["template", "factory"])
def test_tab_factory_errors_surface_at_use_and_reset_keeps_live_form(
    qapp: QApplication, failure: str,
) -> None:
    assert hasattr(ExpAdapterProtocol, "make_form"), (
        "Missing adapter factory contract for Open/Reset"
    )
    broken = True

    class FactoryAdapter(BaseAdapter[FactoryCfg, FactoryCfg]):
        capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
            analysis=AnalysisMode.NONE, load_data=False, requires_soc=False
        )

        @staticmethod
        def make_form(ctx: FormContext) -> Form[FactoryCfg]:
            if broken:
                if failure == "factory":
                    raise RuntimeError("factory source failed")
                return Form(FactoryCfg, {"pulse": Template("missing")})
            return Form(FactoryCfg, {"amount": 3.0})

        def make_filename_stem(self, ctx: SessionEnv) -> str:
            return "factory"

    application = Fixture(headless=True)
    application.registry.register("factory", FactoryAdapter)
    error = ValueError if failure == "template" else RuntimeError
    match = "factory.*pulse" if failure == "template" else "factory source failed"
    try:
        before_tabs = application.ctrl.list_tab_ids()
        with pytest.raises(error, match=match):
            application.ctrl.new_tab("factory")
        assert application.ctrl.list_tab_ids() == before_tabs
        broken = False
        tab_id = application.ctrl.new_tab("factory")
        resource = application.ctrl.cfg_resources.lookup(tab_id)
        resource.write({("amount",): 7.0})
        state, env, authored = resource.editor_state, resource.env, resource.input.data
        notifications: list[FormValue] = []
        resource.subscribe(lambda: notifications.append(resource.preview().values[("amount",)]))
        broken = True
        with pytest.raises(error, match=match):
            resource.reset()
        assert resource.editor_state is state
        assert resource.env is env
        assert resource.input.data == authored
        assert resource.result() == FactoryCfg(amount=7.0)
        assert notifications == []
        broken = False
        resource.reset()
        assert resource.result() == FactoryCfg(amount=3.0)
        assert notifications == [3.0]
    finally:
        application.stop()
        for tab_id in tuple(application.ctrl.list_tab_ids()):
            application.ctrl.close_tab(tab_id)


def test_open_edit_run_reset_and_reopen_restore_use_public_form_authority(
    qapp: QApplication,
) -> None:
    application = Fixture(headless=True)
    application.registry.register("trace", TraceAdapter, form=TraceAdapter.form)
    parameters = application.ctrl.context_control.read_context()
    application.ctrl.context_control.set_parameter(
        parameters.target, "general.ext.amount", 2.0
    )
    try:
        tab_id = application.ctrl.new_tab("trace")
        resource = application.ctrl.cfg_resources.lookup(tab_id)
        resource.write({("amount",): {"expr": "general.ext.amount * 3"}})
        application.ctrl.context_control.set_parameter(
            parameters.target, "general.ext.amount", 5.0
        )
        assert resource.result() == TraceCfg(amount=6.0)
        assert resource.preview().values[("amount",)] == 6.0
        finished: list[RunFinishedPayload] = []
        loop, timer = QEventLoop(), QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)

        def terminal(payload: RunFinishedPayload) -> None:
            if payload.tab_id == tab_id:
                finished.append(payload)
                loop.quit()

        subscription = application.bus.subscribe(RunFinishedPayload, terminal)
        try:
            application.ctrl.start_run(tab_id)
            timer.start(3000)
            if not finished:
                loop.exec()
        finally:
            timer.stop()
            subscription.unsubscribe()
        assert finished, "Run must reach a terminal notification before timeout"
        assert not application.state.is_run_active()
        assert application.state.get_tab(tab_id).run.result == TraceCfg(amount=6.0)
        resource.reset()
        assert resource.result() == TraceCfg(amount=5.0)
        resource.write({("amount",): {"expr": "unfinished +"}})
        saved = application.ctrl.capture_persisted_state()
        assert saved.session.tabs[0].form_input == {"amount": {"expr": "unfinished +"}}
        application.ctrl.close_tab(tab_id)
        with pytest.raises(CfgPreconditionError):
            resource.result()
        report = application.ctrl.restore_persisted_state(saved)
        assert report.restored_tabs == 1
        restored_id = application.ctrl.list_tab_ids()[0]
        assert restored_id != tab_id
        restored = application.ctrl.cfg_resources.lookup(restored_id)
        assert restored.input.data == {"amount": {"expr": "unfinished +"}}
    finally:
        application.stop()
        for tab_id in tuple(application.ctrl.list_tab_ids()):
            application.ctrl.close_tab(tab_id)


def test_public_restore_reports_dropped_inputs_and_keeps_compatible_values(
    qapp: QApplication,
) -> None:
    application = Fixture(headless=True)
    application.registry.register("trace", TraceAdapter, form=TraceAdapter.form)
    try:
        tab_id = application.ctrl.new_tab("trace")
        resource = application.ctrl.cfg_resources.lookup(tab_id)
        resource.write({("amount",): 7.0})
        saved = application.ctrl.capture_persisted_state()
        saved.session.tabs[0].form_input["retired"] = 9.0
        application.ctrl.close_tab(tab_id)
        report = application.ctrl.restore_persisted_state(saved)
        assert report.restored_tabs == 1
        assert report.dropped_inputs[0].adapter_name == "trace"
        assert report.dropped_inputs[0].paths == (("retired",),)
        restored = application.ctrl.cfg_resources.lookup(
            application.ctrl.list_tab_ids()[0]
        )
        assert restored.result() == TraceCfg(amount=7.0)
    finally:
        application.stop()
        for tab_id in tuple(application.ctrl.list_tab_ids()):
            application.ctrl.close_tab(tab_id)
