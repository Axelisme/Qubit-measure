"""Synthetic registered value shared by GUI registry and reload tests."""

from dataclasses import dataclass

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AxesSpec, ZSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots


@dataclass
class DummyResult:
    value: float = 0.0


@dataclass
class DummyAnalyzeResult:
    figure: None = None

    def to_summary_dict(self) -> dict[str, object]:
        return {}


@dataclass
class DummyAnalyzeParams:
    threshold: float = 0.0


def dummy_form(ctx: FormContext) -> Form[ConfigBase]:
    return Form(ConfigBase, {})


def dummy_run(cfg: ConfigBase, *, context: RunContext) -> DummyResult:
    return DummyResult()


def dummy_analyze(
    req: AnalyzeRequest[RunRecord[ConfigBase, DummyResult], DummyAnalyzeParams],
    *,
    plots: Plots,
) -> DummyAnalyzeResult:
    return DummyAnalyzeResult()


DummyAdapter = Adapter(
    spec=AxesSpec(
        axes=(),
        z=ZSpec("value", "value", ""),
        result_type=DummyResult,
        cfg_type=ConfigBase,
        tag="test/dummy",
    ),
    make_form=dummy_form,
    run=dummy_run,
    guide=None,
    analysis=Analysis(
        fn=dummy_analyze, options=lambda source, env: DummyAnalyzeParams()
    ),
)
