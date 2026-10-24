"""Source-dependent native history for tab-owner contract and UI scenarios."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

from zcu_tools.cfg_model import PulseCfg, Template
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import component_registry
from zcu_tools.resources.entry.schema import ComponentSchema

from tests.experiment._saving_support import SavingCfg
from tests.gui.app.measure._native_save_support import SavingAdapter


class HistorySource(ComponentSchema):
    """A synthetic library namespace without calibration fields."""


class HistoryCfg(SavingCfg):
    """Allow template binding to fail after a successful factory invocation."""

    pulse: PulseCfg | None = None


class SourceDependentAdapter(SavingAdapter):
    """Require a captured library source, never the historical native cfg."""

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=True, requires_soc=False
    )

    @staticmethod
    def make_form(ctx: FormContext) -> Form[SavingCfg]:
        if ctx.has("$history.pulses.template"):
            return Form(HistoryCfg, {"pulse": Template("missing")})
        if not ctx.has("$history.pulses.history"):
            if ctx.has("$history.pulses.retry"):
                raise ValueError("factory: replacement source is still missing")
            raise ValueError("factory: history_source is missing")
        return Form(SavingCfg, {"gain": 0.25})


@contextmanager
def history_controller(
    root: Path, *, registry: Registry | None = None
) -> Generator[Controller]:
    """Compose real application owners with only the synthetic adapter catalog."""
    component_registry.register("history/source", HistorySource)
    try:
        registry = registry if registry is not None else Registry()
        registry.register("history", SourceDependentAdapter)
        ctrl = Controller(
            State(SessionEnv(parameters=None, soc=None, soccfg=None)),
            registry,
            None,
            BaseEventBus(),
            project_root=str(root),
        )
        try:
            yield ctrl
        finally:
            ctrl.close()
    finally:
        component_registry.unregister("history/source")
