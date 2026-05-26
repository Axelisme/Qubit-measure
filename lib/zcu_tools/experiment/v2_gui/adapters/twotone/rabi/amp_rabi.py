from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.rabi.amp_rabi import (
    AmpRabiCfg,
    AmpRabiExp,
    AmpRabiResult,
)
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_module_spec,
    make_pulse_ref_default,
    make_readout_module_spec,
    make_readout_ref_default,
    make_reset_module_spec,
    make_reset_ref_default,
    md_get_float,
)
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    AnalyzeRequest,
    CfgNodeValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    RunRequest,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WritebackItem,
    WritebackRequest,
)

AmpRabiRunResult: TypeAlias = AmpRabiResult


@dataclass
class AmpRabiAnalyzeParams:
    skip: Annotated[int, ParamMeta(label="Skip points")]


@dataclass
class AmpRabiAnalyzeResult:
    pi_amp: float
    pi2_amp: float
    figure: Figure


class AmpRabiAdapter(
    AbsExpAdapter[AmpRabiRunResult, AmpRabiAnalyzeResult, AmpRabiAnalyzeParams]
):
    exp_cls = AmpRabiExp

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        pi_amp = md_get_float(ctx, "pi_amp", 0.5)
        root_spec = CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        "reset": make_reset_module_spec(optional=True),
                        "qub_pulse": make_pulse_module_spec(),
                        "readout": make_readout_module_spec(),
                    },
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "relax_delay": ScalarSpec(
                    label="Relax delay (us)", type=float, decimals=3
                ),
                "sweep": CfgSectionSpec(
                    label="Sweep",
                    fields={"gain": SweepSpec(label="Gain (a.u.)")},
                ),
            }
        )
        _module_fields: dict[str, CfgNodeValue] = {
            "qub_pulse": make_pulse_ref_default(ctx),
            "readout": make_readout_ref_default(ctx),
        }
        _reset = make_reset_ref_default(ctx, optional=True)
        if _reset is not None:
            _module_fields["reset"] = _reset
        root_val = CfgSectionValue(
            fields={
                "modules": CfgSectionValue(fields=_module_fields),
                "reps": DirectValue(100),
                "rounds": DirectValue(100),
                "relax_delay": DirectValue(10.5),
                "sweep": CfgSectionValue(
                    fields={
                        "gain": SweepValue(start=-0.3, stop=pi_amp * 2, expts=51),
                    }
                ),
            }
        )
        return CfgSchema(spec=root_spec, value=root_val)

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> AmpRabiCfg:
        return req.ml.make_cfg(raw_cfg, AmpRabiCfg)

    def get_analyze_params(
        self, result: AmpRabiRunResult, ctx: ExpContext
    ) -> AmpRabiAnalyzeParams:
        return AmpRabiAnalyzeParams(skip=0)

    def analyze(
        self, req: AnalyzeRequest[AmpRabiRunResult, AmpRabiAnalyzeParams]
    ) -> AmpRabiAnalyzeResult:
        params = req.analyze_params
        pi_amp, pi2_amp, fig = AmpRabiExp().analyze(
            req.run_result,
            skip=params.skip,
        )
        return AmpRabiAnalyzeResult(pi_amp=pi_amp, pi2_amp=pi2_amp, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[AmpRabiRunResult, AmpRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        ctx = req.ctx
        return [
            MetaDictWriteback(
                key="pi_amp",
                description="Pi pulse gain (a.u.)",
                current_value=ctx.md.get("pi_amp"),
                md_key="pi_amp",
                proposed_value=round(result.pi_amp, 5),
            ),
            MetaDictWriteback(
                key="pi2_amp",
                description="Pi/2 pulse gain (a.u.)",
                current_value=ctx.md.get("pi2_amp"),
                md_key="pi2_amp",
                proposed_value=round(result.pi2_amp, 5),
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_amp_rabi_{time.strftime('%m%d')}"
