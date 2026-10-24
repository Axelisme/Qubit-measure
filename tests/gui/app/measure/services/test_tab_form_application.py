"""Measure composition uses one fixed form authority through Run and restoration."""

from dataclasses import dataclass

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import ConfigBase, Expr, PulseCfg, Template
from zcu_tools.experiment import AxesSpec, ZSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
)
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload
from zcu_tools.gui.form import FormValue
from zcu_tools.gui.form_adapter import Form, FormContext

from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._record_support import require_record
from tests.gui.app.measure.remote._helpers import Fixture


class TraceCfg(ConfigBase):
    """One concrete cfg field is both displayed and executed."""

    amount: float = 1.0


@dataclass
class TraceResult:
    value: TraceCfg


def TraceAdapter_make_form(ctx: FormContext) -> Form[TraceCfg]:
    del ctx
    return Form(TraceCfg, {"amount": Expr("general.ext.amount")})


def TraceAdapter_run(cfg: TraceCfg, *, context: RunContext) -> TraceResult:
    return TraceResult(cfg)


TraceAdapter = Adapter(
    spec=AxesSpec(
        axes=(),
        z=ZSpec("value", "data", ""),
        result_type=TraceResult,
        cfg_type=TraceCfg,
        tag="test/TraceAdapter",
    ),
    make_form=TraceAdapter_make_form,
    run=TraceAdapter_run,
    guide=None,
    analysis=None,
    post_analysis=None,
)


class FactoryCfg(ConfigBase):
    """A synthetic optional module exposes template binding at tab creation."""

    pulse: PulseCfg | None = None
    amount: float = 1.0


@pytest.mark.parametrize(
    ("command", "failure"),
    [("reset", "template"), ("reset", "factory"), ("refresh_sources", "factory")],
)
def test_tab_factory_errors_surface_at_use_and_reset_keeps_live_form(
    tmp_path,
    qapp: QApplication,
    failure: str,
    command: str,
) -> None:
    broken = True

    @dataclass
    class FactoryResult:
        value: float = 0.0

    def FactoryAdapter_make_form(ctx: FormContext) -> Form[FactoryCfg]:
        del ctx
        if broken:
            if failure == "factory":
                raise RuntimeError("factory source failed")
            return Form(FactoryCfg, {"pulse": Template("missing")})
        return Form(FactoryCfg, {"amount": 3.0})

    FactoryAdapter = Adapter(
        spec=AxesSpec(
            axes=(),
            z=ZSpec("value", "data", ""),
            result_type=FactoryResult,
            cfg_type=FactoryCfg,
            tag="test/FactoryAdapter",
        ),
        make_form=FactoryAdapter_make_form,
        run=lambda cfg, *, context: FactoryResult(),
        guide=None,
        analysis=None,
        post_analysis=None,
    )

    application = Fixture(headless=True, project_root=str(tmp_path / "project"))
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


def test_open_edit_run_and_reset_use_public_form_authority(
    tmp_path,
    qapp: QApplication,
) -> None:
    application = Fixture(headless=True, project_root=str(tmp_path / "project"))
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
        assert require_record(
            application.state.get_tab(tab_id).run
        ).result == TraceResult(TraceCfg(amount=6.0))
        resource.reset()
        assert resource.result() == TraceCfg(amount=5.0)
        resource.write({("amount",): {"expr": "unfinished +"}})
    finally:
        application.stop()
        for tab_id in tuple(application.ctrl.list_tab_ids()):
            application.ctrl.close_tab(tab_id)
