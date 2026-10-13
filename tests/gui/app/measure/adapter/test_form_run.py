"""Accepted Form dumps cross the adapter's mapping seam without reassembly."""

import pytest
from pydantic import Field, ValidationError
from zcu_tools.experiment.cfg_model import ExpCfgModel
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
from zcu_tools.gui.form_adapter import Form
from zcu_tools.plotting.plots import NonPresentingHost, Plots


class MeasurementCfg(ExpCfgModel):
    """A core owns its setting validation, independently of GUI assembly."""

    amount: float = Field(default=1.0, gt=0)


class RecordingCore:
    """Return the core setting through the synchronous acquisition boundary."""

    def run(self, cfg: MeasurementCfg, *, context: RunContext) -> float:
        del context
        return cfg.amount


class FormAdapter(BaseAdapter[MeasurementCfg, float]):
    """A normal adapter declares only Form and core, not an assembly override."""

    form = Form(MeasurementCfg, {})
    exp_cls = RecordingCore
    capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE, load_data=False)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        del ctx
        return "measurement"


def test_accepted_form_dump_runs_with_core_validation() -> None:
    resource = TabFormResource(
        FormAdapter.form,
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
            result = adapter.run(req, accepted.values, context=context)
        except NotImplementedError as exc:
            pytest.fail(f"Accepted Form cfg must use its declared core type: {exc}")
        assert result == 4.5
        assert accepted.values["amount"] == 4.5
        assert adapter.build_exp_cfg(accepted.values, req) == MeasurementCfg(amount=4.5)
        with pytest.raises(ValidationError):
            adapter.build_exp_cfg({"amount": -1.0}, req)
    finally:
        plots.finish()
        plots.release()
