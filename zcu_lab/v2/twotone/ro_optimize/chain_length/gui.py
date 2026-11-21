from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TypeAlias

from zcu_tools.analysis.readout_chain import ThermalChainSums
from zcu_tools.cfg_model import TemplateDef
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.ro_optimize._chain_common import ChainAnalysis
from zcu_lab.v2.twotone.ro_optimize.chain_length.core import (
    ChainLengthCfg,
    ChainLengthExp,
)
from zcu_lab.v2.twotone.ro_optimize.chain_length.form import make_form

ChainLengthRunResult: TypeAlias = RunRecord[ChainLengthCfg, ThermalChainSums]


def _analyze(
    req: AnalyzeRequest[ChainLengthRunResult, None], *, plots: Plots
) -> ChainAnalysis:
    return ChainLengthExp().analyze(req.run_result, None, plots=plots)


def _run(cfg: ChainLengthCfg, *, context: RunContext) -> ChainLengthRunResult:
    """Acquire the moment sums and keep the experiment configuration with them."""
    return RunRecord(cfg=cfg, result=ChainLengthExp().run(cfg, context=context))


def build_adapter(
    templates: Mapping[str, TemplateDef],
) -> Adapter[ChainLengthCfg, ThermalChainSums, None, ChainAnalysis]:
    """Bind the registering Registry's templates for Open and Reset.

    Return an adapter without building a Form or inspecting the catalog.
    """
    # No writeback: choose the maximum score; QND and M remain diagnostics.
    return Adapter(
        spec=ChainLengthExp.AXES_SPEC,
        make_form=partial(make_form, templates=templates),
        run=_run,
        guide=Path(__file__).with_name("guide.md"),
        analysis=Analysis(fn=_analyze),
    )
