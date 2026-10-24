"""BaseAdapter passes its validated Form output directly to the core."""

from typing import Any

import pytest
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.base import Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.scope import Scope
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import NonPresentingHost, Plots


class MeasurementCfg(ConfigBase):
    """A Form output need not inherit an experiment cfg base."""

    amount: float = 1.0


class RecordingCore(Experiment[RunRecord[ConfigBase, MeasurementCfg], Any]):
    """Expose the exact cfg received through the synchronous core boundary."""

    def run(
        self, cfg: MeasurementCfg, *, context: RunContext
    ) -> RunRecord[ConfigBase, MeasurementCfg]:
        del context
        return RunRecord(None, cfg)


class FormAdapter(BaseAdapter[MeasurementCfg, RunRecord[ConfigBase, MeasurementCfg]]):
    exp_cls = RecordingCore
    capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE, load_data=False)

    @staticmethod
    def make_form(ctx: FormContext) -> Form[MeasurementCfg]:
        del ctx
        return Form(MeasurementCfg, {})


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
        soc=None, soccfg=None, plots=plots, devices={}, scope=Scope("run")
    )
    try:
        try:
            result = adapter.run(req, measurement(accepted.cfg), context=context)
        except (AttributeError, NotImplementedError, TypeError) as exc:
            pytest.fail(f"Base Run must forward the validated Form output: {exc}")
        assert result.result is accepted.cfg
        assert result.result.amount == 4.5
    finally:
        plots.finish()
        plots.release()
