from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.experiment.v2.onetone.flux_dep import FluxDepCfg, FluxDepExp
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_module_ref_default,
    make_pulse_readout_ref_spec,
    make_reset_ref_spec,
    require_soc_handles,
    save_with_last_state,
)
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    AnalyzeRequest,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
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
    fluxes: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
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
                        "name": ScalarSpec(label="Device name", type=str),
                        "label": ScalarSpec(label="Device label", type=str),
                        "mode": ScalarSpec(
                            label="Mode", type=str, choices=["current", "voltage"]
                        ),
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
                        "name": DirectValue("flux_yoko"),
                        "label": DirectValue("flux_dev"),
                        "mode": DirectValue("current"),
                    }
                ),
                "reps": DirectValue(1000),
                "rounds": DirectValue(1),
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
        name = dev_raw.get("name")
        label = dev_raw.get("label")
        mode = dev_raw.get("mode")
        if not isinstance(name, str) or not name:
            raise RuntimeError("FluxDep dev.name must be a non-empty string")
        if not isinstance(label, str) or not label:
            raise RuntimeError("FluxDep dev.label must be a non-empty string")
        if not isinstance(mode, str) or not mode:
            raise RuntimeError("FluxDep dev.mode must be a non-empty string")
        cfg_raw["dev"] = {name: {"label": label, "mode": mode}}
        return req.ml.make_cfg(cfg_raw, FluxDepCfg)

    def run(self, req: RunRequest, schema: CfgSchema) -> OneToneFluxDepRunResult:
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema.to_raw_dict(req)
        cfg = self.build_exp_cfg(raw_cfg, req)
        fluxes, freqs, signals = FluxDepExp().run(soc, soccfg, cfg)
        return OneToneFluxDepRunResult(
            fluxes=fluxes,
            freqs=freqs,
            signals=signals,
            cfg_snapshot=cfg,
        )

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
        result = req.run_result
        save_with_last_state(
            exp_cls=FluxDepExp,
            cfg=result.cfg_snapshot,
            result=(result.fluxes, result.freqs, result.signals),
            filepath=req.data_path,
        )
