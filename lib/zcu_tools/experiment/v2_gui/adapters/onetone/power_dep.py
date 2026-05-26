from __future__ import annotations

import time

from typing_extensions import Optional, TypeAlias

from zcu_tools.experiment.v2.onetone.power_dep import (
    PowerDepCfg,
    PowerDepExp,
    PowerDepResult,
)
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_readout_default,
    make_pulse_readout_module_spec,
    make_reset_module_spec,
    md_get_float,
    md_has_key,
)
from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ExpContext,
    NoAnalysisAdapterMixin,
    RunRequest,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    require_soc_handles,
)

OneTonePowerDepRunResult: TypeAlias = PowerDepResult


class OneTonePowerDepAdapter(
    NoAnalysisAdapterMixin[PowerDepCfg, OneTonePowerDepRunResult]
):
    exp_cls = PowerDepExp

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        r_f = md_get_float(ctx, "r_f", 6000.0)
        rf_w = md_get_float(ctx, "rf_w", 20.0)
        half_span = 1.5 * rf_w if rf_w > 0 else 30.0
        root_spec = CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        "reset": make_reset_module_spec(optional=True),
                        "readout": make_pulse_readout_module_spec(),
                    },
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "relax_delay": ScalarSpec(
                    label="Relax delay (us)", type=float, decimals=3
                ),
                "earlystop_snr": ScalarSpec(
                    label="Early-stop SNR (0 disables)", type=float, decimals=3
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
        root_val = CfgSectionValue(
            fields={
                "modules": CfgSectionValue(
                    fields={
                        "readout": make_pulse_readout_default(ctx),
                    }
                ),
                "reps": DirectValue(100),
                "rounds": DirectValue(10),
                "relax_delay": DirectValue(10.0),
                "earlystop_snr": DirectValue(0.0),
                "sweep": CfgSectionValue(
                    fields={
                        "gain": SweepValue(start=0.001, stop=0.5, expts=101),
                        "freq": SweepValue(
                            start=(
                                EvalValue(
                                    expr="r_f - 1.5 * rf_w",
                                    resolved=r_f - half_span,
                                    error=None,
                                )
                                if (md_has_key(ctx, "r_f") and md_has_key(ctx, "rf_w"))
                                else (r_f - half_span)
                            ),
                            stop=(
                                EvalValue(
                                    expr="r_f + 1.5 * rf_w",
                                    resolved=r_f + half_span,
                                    error=None,
                                )
                                if (md_has_key(ctx, "r_f") and md_has_key(ctx, "rf_w"))
                                else (r_f + half_span)
                            ),
                            expts=201,
                        ),
                    }
                ),
            }
        )
        return CfgSchema(spec=root_spec, value=root_val)

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> PowerDepCfg:
        cfg_raw = dict(raw_cfg)
        cfg_raw.pop("earlystop_snr", None)
        return req.ml.make_cfg(cfg_raw, PowerDepCfg)

    def _earlystop_snr(self, raw_cfg: dict[str, object]) -> Optional[float]:
        value = raw_cfg.get("earlystop_snr")
        if not isinstance(value, (int, float)):
            return None
        snr = float(value)
        if snr <= 0:
            return None
        return snr

    def run(self, req: RunRequest, schema: CfgSchema) -> OneTonePowerDepRunResult:
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema.to_raw_dict(req)
        cfg = self.build_exp_cfg(raw_cfg, req)
        earlystop_snr = self._earlystop_snr(raw_cfg)
        return PowerDepExp().run(soc, soccfg, cfg, earlystop_snr=earlystop_snr)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.res_name}_gain_{time.strftime('%H%M')}"
