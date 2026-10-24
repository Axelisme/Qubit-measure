"""Run forwards the admitted Form output without rebuilding it."""

from dataclasses import dataclass
from unittest.mock import MagicMock

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AxesSpec, ZSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, ContextReadiness, SessionEnv
from zcu_tools.gui.app.measure.services.guard import GuardService
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import Form, FormContext

from tests.gui.app.measure._cfg_fakes import parameter_capture
from tests.gui.app.measure._record_support import require_record

from ._run_support import make_plots, make_run_service


class ChoiceCfg(ConfigBase):
    choice: str = "visible"


@dataclass
class ChoiceResult:
    value: str


def test_form_output_is_forwarded_without_rebuilding() -> None:
    executed: list[ChoiceCfg] = []
    parameters = parameter_capture()
    captures: list[str] = []
    factories: list[str] = []

    def run(
        cfg: ChoiceCfg, *, context: RunContext
    ) -> RunRecord[ChoiceCfg, ChoiceResult]:
        executed.append(cfg)
        return RunRecord(cfg=cfg, result=ChoiceResult("complete"))

    def factory(ctx: FormContext) -> Form[ChoiceCfg]:
        factories.append("factory")
        return Form(ChoiceCfg, {})

    adapter = Adapter(
        spec=AxesSpec(
            axes=(),
            z=ZSpec("value", "value", ""),
            result_type=ChoiceResult,
            cfg_type=ChoiceCfg,
            tag="test/choice",
        ),
        make_form=factory,
        run=run,
        guide=None,
    )

    def capture(_aliases: object) -> TabFormSources:
        captures.append("capture")
        return TabFormSources(FormEnv(parameters=parameters))

    resource = TabFormResource(
        adapter.make_form, capture=capture, mutation_allowed=lambda: True
    )
    state = State(
        SessionEnv(
            parameters=parameters,
            soc=MagicMock(),
            soccfg=MagicMock(),
            readiness=ContextReadiness.ACTIVE,
        )
    )
    state.add_tab(
        "choice", Session(adapter_name="choice", adapter=adapter, cfg=resource)
    )
    resource.write({("choice",): "edited"})
    inputs = GuardService(state).capture_run_inputs("choice")
    cfg = inputs.accepted_cfg.cfg
    service, _gate, background, _handles = make_run_service(state)
    service.start_run(inputs, plots=make_plots())
    resource.write({("choice",): "later"})
    background.run_work()
    assert executed == [cfg]
    assert executed[0] is cfg
    assert require_record(state.get_tab("choice").run).result == ChoiceResult(
        "complete"
    )
    service.release_view_plots("choice")
    assert cfg.model_dump() == {"choice": "edited"}
    assert factories == ["factory"]
    assert captures == ["capture"]
