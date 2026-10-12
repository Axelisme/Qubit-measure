"""Measure composition uses one fixed form authority through Run and restoration."""

from collections.abc import Mapping
from typing import ClassVar

import pytest
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import Expr
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterCfgDefinition,
    AnalysisMode,
    Form,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload
from zcu_tools.gui.cfg.resource import CfgPreconditionError

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
