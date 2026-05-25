from __future__ import annotations

import time

from typing_extensions import TypeAlias

from zcu_tools.experiment.v2.twotone.power_dep import PowerCfg, PowerExp, PowerResult
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
    CfgNodeValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ExpContext,
    NoAnalysisResult,
    NoAnalyzeParams,
    RunRequest,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)

PowerDepRunResult: TypeAlias = PowerResult


class PowerDepAdapter(
    AbsExpAdapter[PowerDepRunResult, NoAnalysisResult, NoAnalyzeParams]
):
    exp_cls = PowerExp

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        q_f = md_get_float(ctx, "q_f", 4000.0)
        qf_w = md_get_float(ctx, "qf_w", 20.0)
        half_span = 1.5 * qf_w if qf_w > 0 else 30.0
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
                    fields={
                        "gain": SweepSpec(label="Gain (a.u.)"),
                        "freq": SweepSpec(label="Freq (MHz)"),
                    },
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
                "relax_delay": DirectValue(1.0),
                "sweep": CfgSectionValue(
                    fields={
                        "gain": SweepValue(start=0.001, stop=0.5, expts=101),
                        "freq": SweepValue(
                            start=q_f - half_span,
                            stop=q_f + half_span,
                            expts=201,
                        ),
                    }
                ),
            }
        )
        return CfgSchema(spec=root_spec, value=root_val)

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> PowerCfg:
        return req.ml.make_cfg(raw_cfg, PowerCfg)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_qubit_power_{time.strftime('%H%M')}"
