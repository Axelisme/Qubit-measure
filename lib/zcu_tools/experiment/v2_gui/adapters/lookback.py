from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Any, ClassVar, Sequence, TypeAlias

from zcu_tools.experiment.v2.lookback import LookbackCfg, LookbackExp, LookbackResult
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_readout_default,
    make_reset_module_spec,
    make_trig_offset,
)
from zcu_tools.gui.adapter import (
    AdapterCapabilities,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    ScalarSpec,
    WritebackItem,
    WritebackRequest,
)

LookbackRunResult: TypeAlias = LookbackResult


@dataclass
class LookbackAnalyzeParams:
    ratio: Annotated[float, ParamMeta(label="Threshold ratio", decimals=3)]
    smooth: Annotated[float, ParamMeta(label="Smooth sigma", decimals=3)]
    plot_fit: Annotated[bool, ParamMeta(label="Plot fit")]


@dataclass
class LookbackAnalyzeResult(AnalyzeResultBase):
    predict_offset: float
    figure: Figure


class LookbackAdapter(
    BaseAdapter[
        LookbackCfg,
        LookbackRunResult,
        LookbackAnalyzeResult,
        LookbackAnalyzeParams,
    ]
):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=False
    )
    exp_cls = LookbackExp
    ExpCfg_cls: ClassVar[Any] = LookbackCfg

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        "readout": make_pulse_readout_module_spec(),
                        "init_pulse": make_pulse_module_spec(optional=True),
                        "reset": make_reset_module_spec(optional=True),
                    },
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "relax_delay": ScalarSpec(
                    label="Relax delay (us)", type=float, decimals=3
                ),
            }
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        root_val = CfgSectionValue(
            fields={
                "modules": CfgSectionValue(
                    fields={
                        "readout": make_readout_default(ctx)
                        .with_field("pulse_cfg.gain", 1.0)
                        .with_field("ro_cfg.ro_length", 1.4)
                        .with_field(
                            "ro_cfg.trig_offset",
                            make_trig_offset(
                                ctx,
                                trig_expr="timeFly - 0.1",
                                trig_fallback=0.4,
                            ),
                        ),
                    }
                ),
                "reps": DirectValue(1),
                "rounds": DirectValue(500),
                "relax_delay": DirectValue(0.0),
            }
        )
        return root_val

    def get_analyze_params(
        self, result: LookbackRunResult, ctx: ExpContext
    ) -> LookbackAnalyzeParams:
        return LookbackAnalyzeParams(ratio=0.1, smooth=1.0, plot_fit=True)

    def analyze(
        self,
        req: AnalyzeRequest[LookbackRunResult, LookbackAnalyzeParams],
    ) -> LookbackAnalyzeResult:
        params = req.analyze_params
        offset, figure = LookbackExp().analyze(
            req.run_result,
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
                target_name="timeFly",
                description="Readout trigger offset prediction (us)",
                proposed_value=req.analyze_result.predict_offset,
            )
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"lookback_{time.strftime('%H%M')}"
