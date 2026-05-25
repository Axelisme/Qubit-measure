from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.onetone.flux_dep import (
    FluxDepCfg,
    FluxDepExp,
    FluxDepResult,
)
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_module_ref_default,
    make_pulse_readout_ref_spec,
    make_reset_ref_spec,
    require_soc_handles,
)
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    AnalyzeRequest,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    DirectValue,
    ExpContext,
    RunRequest,
    SaveDataRequest,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.specs.readout import make_pulse_readout_spec
from zcu_tools.program.v2 import PulseReadoutCfg


@dataclass
class OneToneFluxDepRunResult:
    result: FluxDepResult
    cfg_snapshot: FluxDepCfg


@dataclass
class NoAnalyzeParams:
    pass


@dataclass
class NoAnalysisResult:
    figure: Optional[Figure] = None


def _md_float(ctx: ExpContext, key: str, default: float) -> float:
    value = getattr(ctx.md, key, None)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _pulse_readout_default(ctx: ExpContext):
    return make_module_ref_default(
        ml=ctx.ml,
        module_type=PulseReadoutCfg,
        preferred_names=["readout_rf", "readout", "res_readout"],
        fallback_key="<Custom:Pulse Readout>",
        fallback_spec_factory=make_pulse_readout_spec,
    )


class OneToneFluxDepAdapter(
    AbsExpAdapter[OneToneFluxDepRunResult, NoAnalysisResult, NoAnalyzeParams]
):
    exp_cls = FluxDepExp

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        r_f = _md_float(ctx, "r_f", 6000.0)
        rf_w = _md_float(ctx, "rf_w", 20.0)
        half_span = rf_w if rf_w > 0 else 20.0
        root_spec = CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        "readout": make_pulse_readout_ref_spec(),
                        "reset": make_reset_ref_spec(optional=True),
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
                        "flux": SweepSpec(label="Flux device value"),
                        "freq": SweepSpec(label="Freq (MHz)"),
                    },
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
                "dev": CfgSectionValue(
                    fields={
                        "flux_dev": DirectValue("flux_yoko"),
                    }
                ),
                "reps": DirectValue(100),
                "rounds": DirectValue(100),
                "relax_delay": DirectValue(1.0),
                "sweep": CfgSectionValue(
                    fields={
                        "flux": SweepValue(start=3.57e-3, stop=3.61e-3, expts=101),
                        "freq": SweepValue(
                            start=r_f - half_span,
                            stop=r_f + half_span,
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

    def run(self, req: RunRequest, schema: CfgSchema) -> OneToneFluxDepRunResult:
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema.to_raw_dict(req)
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = FluxDepExp().run(soc, soccfg, cfg)
        return OneToneFluxDepRunResult(result=result, cfg_snapshot=cfg)

    def get_analyze_params(
        self, result: OneToneFluxDepRunResult, ctx: ExpContext
    ) -> NoAnalyzeParams:
        return NoAnalyzeParams()

    def analyze(
        self, req: AnalyzeRequest[OneToneFluxDepRunResult, NoAnalyzeParams]
    ) -> NoAnalysisResult:
        return NoAnalysisResult()

    def get_writeback_items(
        self, req: WritebackRequest[OneToneFluxDepRunResult, NoAnalysisResult]
    ) -> Sequence[WritebackItem]:
        return []

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.res_name}_flux"

    def save(self, req: SaveDataRequest[OneToneFluxDepRunResult]) -> None:
        run_result = req.run_result
        FluxDepExp().save(
            filepath=req.data_path,
            result=run_result.result,
        )
