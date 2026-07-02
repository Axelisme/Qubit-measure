from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from zcu_tools.gui.app.main.adapter import AnalyzeRequest, NoAnalyzeParams, RunRequest
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def make_ml() -> MagicMock:
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    ml.make_cfg.return_value = MagicMock()
    return ml


def make_ctx(ml: MagicMock | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.ml = ml or make_ml()
    ctx.md = MetaDict()
    ctx.qub_name = "Q1"
    return ctx


def md_with_centers() -> MetaDict:
    md = MetaDict()
    md.g_center = -1.5 + 2.0j
    md.e_center = 1.2 - 0.7j
    md.ge_radius = 0.42
    return md


def run_req(md: MetaDict, ml: MagicMock) -> RunRequest:
    soc, soccfg = MagicMock(), MagicMock()
    return RunRequest(md=md, ml=cast(ModuleLibrary, ml), soc=soc, soccfg=soccfg)


def analyze_req(run_result: Any, md: MetaDict) -> AnalyzeRequest[Any, NoAnalyzeParams]:
    return AnalyzeRequest(
        run_result=run_result,
        analyze_params=NoAnalyzeParams(),
        md=md,
        ml=cast(ModuleLibrary, make_ml()),
        predictor=None,
    )
