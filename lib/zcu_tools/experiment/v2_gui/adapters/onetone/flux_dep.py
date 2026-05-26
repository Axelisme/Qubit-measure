from __future__ import annotations

from typing_extensions import TypeAlias, Union

from zcu_tools.experiment.v2.onetone.flux_dep import (
    FluxDepCfg,
    FluxDepExp,
    FluxDepResult,
)
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_readout_default,
    make_pulse_readout_module_spec,
    make_reset_module_spec,
    md_get_float,
    md_has_key,
)
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    DirectValue,
    EvalValue,
    ExpContext,
    NoAnalysisResult,
    NoAnalyzeParams,
    RunRequest,
    ScalarSpec,
    ScalarValue,
    SweepSpec,
    SweepValue,
)

OneToneFluxDepRunResult: TypeAlias = FluxDepResult


class OneToneFluxDepAdapter(
    AbsExpAdapter[OneToneFluxDepRunResult, NoAnalysisResult, NoAnalyzeParams]
):
    exp_cls = FluxDepExp

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        r_f = md_get_float(ctx, "r_f", 6000.0)
        rf_w = md_get_float(ctx, "rf_w", 20.0)
        half_span = rf_w if rf_w > 0 else 20.0
        probe_len = md_get_float(ctx, "res_probe_len", 1.0)
        ro_length: Union[float, ScalarValue] = (
            EvalValue(
                expr="res_probe_len - 0.1",
                resolved=probe_len - 0.1,
                error=None,
            )
            if md_has_key(ctx, "res_probe_len")
            else probe_len - 0.1
        )
        root_spec = CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        "reset": make_reset_module_spec(optional=True),
                        "readout": make_pulse_readout_module_spec(),
                    },
                ),
                "dev": CfgSectionSpec(
                    label="Flux Device",
                    fields={
                        "flux_dev": DeviceRefSpec(label="Flux Device"),
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
                        "flux": SweepSpec(label="Flux device value", decimals=6),
                        "freq": SweepSpec(label="Freq (MHz)"),
                    },
                ),
            }
        )
        root_val = CfgSectionValue(
            fields={
                "modules": CfgSectionValue(
                    fields={
                        "readout": make_pulse_readout_default(
                            ctx, gain=0.005, ro_length=ro_length
                        ),
                    }
                ),
                "dev": CfgSectionValue(
                    fields={
                        "flux_dev": DirectValue("flux_yoko"),
                    }
                ),
                "reps": DirectValue(1000),
                "rounds": DirectValue(1),
                "relax_delay": DirectValue(1.0),
                "sweep": CfgSectionValue(
                    fields={
                        "flux": SweepValue(start=3.57e-3, stop=3.61e-3, expts=101),
                        "freq": SweepValue(
                            start=(
                                EvalValue(
                                    expr="r_f - rf_w",
                                    resolved=r_f - half_span,
                                    error=None,
                                )
                                if (md_has_key(ctx, "r_f") and md_has_key(ctx, "rf_w"))
                                else (r_f - half_span)
                            ),
                            stop=(
                                EvalValue(
                                    expr="r_f + rf_w",
                                    resolved=r_f + half_span,
                                    error=None,
                                )
                                if (md_has_key(ctx, "r_f") and md_has_key(ctx, "rf_w"))
                                else (r_f + half_span)
                            ),
                            expts=101,
                        ),
                    }
                ),
            }
        )
        return CfgSchema(spec=root_spec, value=root_val)

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> FluxDepCfg:
        cfg_raw = dict(raw_cfg)
        dev_raw = cfg_raw.pop("dev")
        if not isinstance(dev_raw, dict):
            raise RuntimeError("FluxDep dev section must lower to a dict")
        # dev_raw = {"flux_dev": "flux_yoko", ...}
        # convert to make_cfg patch format: {"flux_yoko": {"label": "flux_dev"}}
        dev_patch: dict[str, dict] = {}
        for label_key, device_name in dev_raw.items():
            if not isinstance(device_name, str) or not device_name:
                raise RuntimeError(
                    f"FluxDep dev.{label_key} must be a non-empty device name"
                )
            dev_patch[device_name] = {"label": label_key}
        cfg_raw["dev"] = dev_patch
        return req.ml.make_cfg(cfg_raw, FluxDepCfg)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.res_name}_flux"
