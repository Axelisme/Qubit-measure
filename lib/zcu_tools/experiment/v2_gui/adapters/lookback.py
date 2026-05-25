from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Annotated, Sequence

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.lookback import LookbackCfg, LookbackExp, LookbackResult
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_module_ref_default,
    make_pulse_readout_ref_spec,
    make_pulse_ref_spec,
    make_reset_ref_spec,
    require_soc_handles,
)
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    AnalyzeRequest,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    RunRequest,
    SaveDataRequest,
    ScalarSpec,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.specs.readout import make_pulse_readout_spec
from zcu_tools.program.v2 import PulseReadoutCfg


@dataclass
class LookbackRunResult:
    result: LookbackResult
    cfg_snapshot: LookbackCfg


@dataclass
class LookbackAnalyzeParams:
    ratio: Annotated[float, ParamMeta(label="Threshold ratio", decimals=3)]
    smooth: Annotated[float, ParamMeta(label="Smooth sigma", decimals=3)]
    plot_fit: Annotated[bool, ParamMeta(label="Plot fit")]


@dataclass
class LookbackAnalyzeResult:
    predict_offset: float
    figure: Figure


def _pulse_readout_default(ctx: ExpContext):
    return make_module_ref_default(
        ml=ctx.ml,
        module_type=PulseReadoutCfg,
        preferred_names=["readout_rf", "readout", "res_readout"],
        fallback_key="<Custom:Pulse Readout>",
        fallback_spec_factory=make_pulse_readout_spec,
    )


class LookbackAdapter(
    AbsExpAdapter[LookbackRunResult, LookbackAnalyzeResult, LookbackAnalyzeParams]
):
    exp_cls = LookbackExp

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        root_spec = CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        "readout": make_pulse_readout_ref_spec(),
                        "init_pulse": make_pulse_ref_spec(optional=True),
                        "reset": make_reset_ref_spec(optional=True),
                    },
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "relax_delay": ScalarSpec(
                    label="Relax delay (us)", type=float, decimals=3
                ),
            }
        )
        root_val = CfgSectionValue(
            fields={
                "modules": CfgSectionValue(
                    fields={
                        "readout": _pulse_readout_default(ctx),
                    }
                ),
                "reps": DirectValue(1),
                "rounds": DirectValue(500),
                "relax_delay": DirectValue(0.0),
            }
        )
        return CfgSchema(spec=root_spec, value=root_val)

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> LookbackCfg:
        return req.ml.make_cfg(raw_cfg, LookbackCfg)

    def run(self, req: RunRequest, schema: CfgSchema) -> LookbackRunResult:
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema.to_raw_dict(req)
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = LookbackExp().run(soc, soccfg, cfg)
        return LookbackRunResult(result=result, cfg_snapshot=cfg)

    def get_analyze_params(
        self, result: LookbackRunResult, ctx: ExpContext
    ) -> LookbackAnalyzeParams:
        return LookbackAnalyzeParams(ratio=0.1, smooth=1.0, plot_fit=True)

    def analyze(
        self,
        req: AnalyzeRequest[LookbackRunResult, LookbackAnalyzeParams],
    ) -> LookbackAnalyzeResult:
        params = req.analyze_params
        run_result = req.run_result
        offset, figure = LookbackExp().analyze(
            run_result.result,
            ratio=params.ratio,
            smooth=params.smooth,
            plot_fit=params.plot_fit,
        )
        return LookbackAnalyzeResult(predict_offset=offset, figure=figure)

    def get_writeback_items(
        self, req: WritebackRequest[LookbackRunResult, LookbackAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return [
            MetaDictWriteback(
                key="timeFly",
                description="Readout trigger offset prediction (us)",
                current_value=getattr(req.ctx.md, "timeFly", None),
                md_key="timeFly",
                proposed_value=round(req.analyze_result.predict_offset, 6),
            )
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"lookback_{time.strftime('%H%M')}"

    def save(self, req: SaveDataRequest[LookbackRunResult]) -> None:
        run_result = req.run_result
        LookbackExp().save(
            filepath=req.data_path,
            result=run_result.result,
        )
