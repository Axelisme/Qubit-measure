"""Adapter forwards its validated Form output directly to the run callback."""

from dataclasses import dataclass

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AxesSpec, ZSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.scope import Scope
from zcu_tools.gui.app.measure.adapter import Adapter
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import NonPresentingHost, Plots


class MeasurementCfg(ConfigBase):
    amount: float = 1.0


@dataclass
class MeasurementResult:
    cfg: MeasurementCfg


def recording_run(
    cfg: MeasurementCfg, *, context: RunContext
) -> RunRecord[MeasurementCfg, MeasurementResult]:
    return RunRecord(cfg=cfg, result=MeasurementResult(cfg))


def measurement_form(ctx: FormContext) -> Form[MeasurementCfg]:
    return Form(MeasurementCfg, {})


FormAdapter = Adapter(
    spec=AxesSpec(
        axes=(),
        z=ZSpec("cfg", "cfg", ""),
        result_type=MeasurementResult,
        cfg_type=MeasurementCfg,
        tag="test/measurement",
    ),
    make_form=measurement_form,
    run=recording_run,
    guide=None,
)


def measurement(cfg: ConfigBase) -> MeasurementCfg:
    assert isinstance(cfg, MeasurementCfg)
    return cfg


def test_base_run_preserves_form_output_identity() -> None:
    resource = TabFormResource(
        FormAdapter.make_form,
        capture=lambda _aliases: TabFormSources(FormEnv()),
        mutation_allowed=lambda: True,
    )
    resource.write({("amount",): 4.5})
    accepted = resource.accept()
    plots = Plots(NonPresentingHost())
    context = RunContext(
        soc=None, soccfg=None, plots=plots, devices={}, scope=Scope("run")
    )
    try:
        result = FormAdapter.run(measurement(accepted.cfg), context=context)
        assert result.result.cfg is accepted.cfg
        assert result.result.cfg.amount == 4.5
    finally:
        plots.finish()
        plots.release()
