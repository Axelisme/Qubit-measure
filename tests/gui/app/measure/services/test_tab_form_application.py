"""Measure composition uses one fixed form authority through Run and restoration."""

import json
from pathlib import Path
from typing import ClassVar

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import ConfigBase, Expr, PulseCfg, Template
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    ExpAdapterProtocol,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload
from zcu_tools.gui.app.measure.services.caretaker import create_persistence_caretaker
from zcu_tools.gui.app.measure.services.form_contracts import CfgPreconditionError
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form import FormInput, FormValue
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.context_types import EntryRequest, PointUseRequest

from tests.gui._qt_waits import qt_until
from tests.gui.app.measure.remote._helpers import Fixture


class TraceCfg(ExpCfgModel):
    """One concrete cfg field is both displayed and executed."""

    amount: float = 1.0


class TraceAdapter(BaseAdapter[TraceCfg, TraceCfg]):
    """Return accepted cfg without devices, acquisition or result persistence."""

    @staticmethod
    def make_form(ctx: FormContext) -> Form[TraceCfg]:
        del ctx
        return Form(TraceCfg, {"amount": Expr("general.ext.amount")})

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=False, requires_soc=False
    )

    def run(
        self,
        req: RunRequest,
        cfg: TraceCfg,
        *,
        context: RunContext,
    ) -> TraceCfg:
        return cfg


def test_deferred_restore_preserves_pending_until_point_is_ready(
    qapp: QApplication,
) -> None:
    application = Fixture(headless=True)
    application.registry.register("trace", TraceAdapter)
    try:
        tab_id = application.ctrl.new_tab("trace")
        application.ctrl.cfg_resources.lookup(tab_id).write({("amount",): 7.0})
        saved = application.ctrl.capture_persisted_state()
        application.ctrl.close_tab(tab_id)
        application.ctrl.setup_control.open_entry(EntryRequest("fake_qubit"))
        report = application.ctrl.restore_persisted_state(saved)
        assert report.restored_tabs == 0, "Startup must wait for a selected point"
        assert application.ctrl.list_tab_ids() == []
        assert application.ctrl.capture_persisted_state().session == saved.session
        with pytest.raises(FailedPreconditionError) as rejected:
            application.ctrl.restore_session()
        assert rejected.value.reason_code == "no_context"
        assert application.ctrl.capture_persisted_state().session == saved.session
        application.ctrl.context_control.use_point(PointUseRequest("ctx001"))
        restored_ids = application.ctrl.list_tab_ids()
        assert len(restored_ids) == 1
        assert application.ctrl.cfg_resources.lookup(
            restored_ids[0]
        ).result() == TraceCfg(amount=7.0)
        assert application.ctrl.restore_session().restored_tabs == 0
        application.ctrl.context_control.use_point(PointUseRequest("ctx001"))
        assert application.ctrl.list_tab_ids() == restored_ids
        application.ctrl.close_tab(restored_ids[0])
        assert application.ctrl.capture_persisted_state().session.tabs == ()
    finally:
        application.stop()


@pytest.mark.parametrize("adopt", ["restore", "clean"])
def test_script_mode_preserves_original_until_explicit_adoption(
    qapp: QApplication, tmp_path: Path, adopt: str
) -> None:
    application = Fixture(headless=True)
    application.registry.register("trace", TraceAdapter)
    try:
        tab_id = application.ctrl.new_tab("trace")
        application.ctrl.cfg_resources.lookup(tab_id).write({("amount",): 7.0})
        saved = application.ctrl.capture_persisted_state()
        caretaker = create_persistence_caretaker(application.ctrl, cache_dir=tmp_path)
        application.ctrl.attach_caretaker(caretaker)
        caretaker.flush()
        application.ctrl.close_tab(tab_id)
        assert callable(getattr(application.ctrl, "finish_startup_script", None)), (
            "Script completion must explicitly adopt current tabs"
        )
        application.ctrl.restore_all(auto_restore_tabs=False)
        application.ctrl.context_control.use_point(PointUseRequest("ctx001"))
        assert application.ctrl.list_tab_ids() == []
        # Script may already have opened new tabs; interrupted startup retains disk tabs.
        live_id = application.ctrl.new_tab("trace")
        application.ctrl.cfg_resources.lookup(live_id).write({("amount",): 9.0})
        application.ctrl.persist_all()
        assert json.loads(caretaker.state_path.read_text())[
            "session"
        ] == saved.session.model_dump(mode="json")
        if adopt == "restore":
            assert application.ctrl.restore_session().restored_tabs == 1
        application.ctrl.finish_startup_script()
        application.ctrl.finish_startup_script()
        application.ctrl.persist_all()
        amounts = [
            tab.form_input["data"]
            for tab in application.ctrl.capture_persisted_state().session.tabs
        ]
        assert amounts == (
            [{"amount": 9.0}, {"amount": 7.0}]
            if adopt == "restore"
            else [{"amount": 9.0}]
        )
        assert application.ctrl.restore_session().restored_tabs == 0
        assert json.loads(caretaker.state_path.read_text())[
            "session"
        ] == application.ctrl.capture_persisted_state().session.model_dump(mode="json")
    finally:
        application.stop()


class FactoryCfg(ConfigBase):
    """A synthetic optional module exposes template binding at tab creation."""

    pulse: PulseCfg | None = None
    amount: float = 1.0


@pytest.mark.parametrize(
    ("command", "failure"),
    [("reset", "template"), ("reset", "factory"), ("refresh_sources", "factory")],
)
def test_tab_factory_errors_surface_at_use_and_reset_keeps_live_form(
    qapp: QApplication,
    failure: str,
    command: str,
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
            del ctx
            if broken:
                if failure == "factory":
                    raise RuntimeError("factory source failed")
                return Form(FactoryCfg, {"pulse": Template("missing")})
            return Form(FactoryCfg, {"amount": 3.0})

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
        resource.subscribe(
            lambda: notifications.append(resource.preview().values[("amount",)])
        )
        recapture = resource.reset if command == "reset" else resource.refresh_sources
        broken = True
        with pytest.raises(error, match=match):
            recapture()
        assert resource.editor_state is state
        assert resource.env is env
        assert resource.input.data == authored
        assert resource.result() == FactoryCfg(amount=7.0)
        assert notifications == []
        broken = False
        recapture()
        expected = 3.0 if command == "reset" else 7.0
        assert resource.result() == FactoryCfg(amount=expected)
        assert notifications == [expected]
    finally:
        application.stop()
        for tab_id in tuple(application.ctrl.list_tab_ids()):
            application.ctrl.close_tab(tab_id)


def test_open_edit_run_reset_and_reopen_restore_use_public_form_authority(
    qapp: QApplication,
) -> None:
    application = Fixture(headless=True)
    application.registry.register("trace", TraceAdapter)
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

        def terminal(payload: RunFinishedPayload) -> None:
            if payload.tab_id == tab_id:
                finished.append(payload)

        subscription = application.bus.subscribe(RunFinishedPayload, terminal)
        try:
            application.ctrl.start_run(tab_id)
            # The admitted trace adapter returns and publishes RunFinishedPayload.
            qt_until(lambda: bool(finished), label="form authority run terminal")
        finally:
            subscription.unsubscribe()
        assert finished, "Run must reach a terminal notification before timeout"
        assert not application.state.is_run_active()
        assert application.state.get_tab(tab_id).run.result == TraceCfg(amount=6.0)
        resource.reset()
        assert resource.result() == TraceCfg(amount=5.0)
        resource.write({("amount",): {"expr": "unfinished +"}})
        saved = application.ctrl.capture_persisted_state()
        assert saved.session.tabs[0].form_input == {
            "data": {"amount": {"expr": "unfinished +"}},
            "selections": [],
        }
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
    application.registry.register("trace", TraceAdapter)
    try:
        tab_id = application.ctrl.new_tab("trace")
        resource = application.ctrl.cfg_resources.lookup(tab_id)
        resource.write({("amount",): 7.0})
        saved = application.ctrl.capture_persisted_state()
        saved.session.tabs[0].form_input.update(
            json.loads(
                FormInput(
                    resource.input.model, {**resource.input.data, "retired": 9.0}
                ).to_json()
            )
        )
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
