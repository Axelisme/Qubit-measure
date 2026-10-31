"""Measure attachment for joint readout with gg/ee herald diagnostics."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core_heralded import (
    JointReadoutHeraldedAnalysis,
    JointReadoutHeraldedCfg,
    JointReadoutHeraldedExp,
    JointReadoutHeraldedResult,
)
from .form import joint_readout_form

JointReadoutHeraldedRun: TypeAlias = RunRecord[
    JointReadoutHeraldedCfg, JointReadoutHeraldedResult
]


def make_form(ctx: FormContext) -> Form[JointReadoutHeraldedCfg]:
    """Capture full mux roles and seed inputs for the heralded experiment."""
    return joint_readout_form(ctx, JointReadoutHeraldedCfg)


def run(
    cfg: JointReadoutHeraldedCfg, *, context: RunContext
) -> JointReadoutHeraldedRun:
    """Run the heralded experiment and retain its actual configuration."""
    return RunRecord(
        cfg=cfg, result=JointReadoutHeraldedExp().run(cfg, context=context)
    )


def _analyze(
    req: AnalyzeRequest[JointReadoutHeraldedRun, None], *, plots: Plots
) -> JointReadoutHeraldedAnalysis:
    return JointReadoutHeraldedExp().analyze(req.run_result, None, plots=plots)


DEFINITION = Adapter(
    spec=JointReadoutHeraldedExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide_heralded.md"),
    analysis=Analysis(fn=_analyze),
)
