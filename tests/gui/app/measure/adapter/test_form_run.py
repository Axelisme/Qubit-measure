"""BaseAdapter passes its validated Form output directly to the core."""

import pytest
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.services.tab_cfg import TabFormResource, TabFormSources
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import NonPresentingHost, Plots


class MeasurementCfg(ConfigBase):
    """A Form output need not inherit an experiment cfg base."""

    amount: float = 1.0


class RecordingCore:
    """Expose the exact cfg received through the synchronous core boundary."""

    def run(self, cfg: MeasurementCfg, *, context: RunContext) -> MeasurementCfg:
        del context
        return cfg


class FormAdapter(BaseAdapter[MeasurementCfg, MeasurementCfg]):
    exp_cls = RecordingCore
    capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE, load_data=False)

    @staticmethod
    def make_form(ctx: FormContext) -> Form[MeasurementCfg]:
        del ctx
        return Form(MeasurementCfg, {})

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        del ctx
        return "measurement"


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
    adapter = FormAdapter()
    req = RunRequest(soc=None, soccfg=None, cfg_context=None)
    plots = Plots(NonPresentingHost())
    context = RunContext(
        soc=None, soccfg=None, plots=plots, devices={}, cancel_signal=StopSignal()
    )
    try:
        try:
            result = adapter.run(req, measurement(accepted.cfg), context=context)
        except (AttributeError, NotImplementedError, TypeError) as exc:
            pytest.fail(f"Base Run must forward the validated Form output: {exc}")
        assert result is accepted.cfg
        assert result.amount == 4.5
    finally:
        plots.finish()
        plots.release()
