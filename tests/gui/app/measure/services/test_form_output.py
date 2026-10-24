"""Guard and Run forward the adapter's opaque Form output unchanged."""

from typing import ClassVar

import pytest
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    ContextReadiness,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.services.guard import GuardService
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import Form, FormContext

from tests.gui.app.measure._cfg_fakes import parameter_capture
from tests.gui.app.measure._record_support import require_record

from ._run_support import make_plots, make_run_service


class ChoiceCfg(ConfigBase):
    """An adapter output with no experiment-specific base or runtime fields."""

    choice: str = "visible"


class ChoiceAdapter(BaseAdapter[ChoiceCfg, RunRecord[ConfigBase, str]]):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=False, analysis=AnalysisMode.NONE, load_data=False
    )

    def __init__(self) -> None:
        self.validated: list[ChoiceCfg] = []
        self.executed: list[ChoiceCfg] = []

    @staticmethod
    def make_form(ctx: FormContext) -> Form[ChoiceCfg]:
        del ctx
        return Form(ChoiceCfg, {})

    def validate_run_request(self, req: RunRequest, cfg: ChoiceCfg) -> None:
        del req
        self.validated.append(cfg)

    def run(
        self, req: RunRequest, cfg: ChoiceCfg, *, context: RunContext
    ) -> RunRecord[ConfigBase, str]:
        del req, context
        self.executed.append(cfg)
        return RunRecord(None, "complete")


@pytest.mark.parametrize("phase", ["guard", "run"])
def test_form_output_is_forwarded_without_rebuilding(phase: str) -> None:
    adapter = ChoiceAdapter()
    parameters = parameter_capture()
    captures: list[str] = []
    factories: list[str] = []

    def factory(ctx: FormContext) -> Form[ChoiceCfg]:
        factories.append("factory")
        return adapter.make_form(ctx)

    def capture(_aliases: object) -> TabFormSources:
        captures.append("capture")
        return TabFormSources(FormEnv(parameters=parameters))

    resource = TabFormResource(factory, capture=capture, mutation_allowed=lambda: True)
    state = State(
        SessionEnv(
            parameters=parameters,
            soc=None,
            soccfg=None,
            readiness=ContextReadiness.ACTIVE,
        )
    )
    state.add_tab(
        "choice", Session(adapter_name="choice", adapter=adapter, cfg=resource)
    )
    resource.write({("choice",): "edited"})
    inputs = GuardService(state).capture_run_inputs("choice")
    cfg = inputs.accepted_cfg.cfg
    if phase == "guard":
        assert adapter.validated == [cfg]
        assert adapter.validated[0] is cfg
    else:
        service, _gate, background, _handles = make_run_service(state)
        service.start_run(inputs, plots=make_plots())
        # Editing after admission must not replace the admitted cfg.
        resource.write({("choice",): "later"})
        background.run_work()
        assert adapter.executed == [cfg]
        assert adapter.executed[0] is cfg
        assert require_record(state.get_tab("choice").run).result == "complete"
        service.release_view_plots("choice")
    assert cfg.model_dump() == {"choice": "edited"}
    assert factories == ["factory"]
    assert captures == ["capture"]
