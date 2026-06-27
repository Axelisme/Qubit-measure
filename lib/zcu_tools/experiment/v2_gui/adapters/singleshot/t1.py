from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.singleshot.t1.t1 import T1Cfg, T1Exp, T1Result
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    FigureOnlyAnalyzeResult,
    Init,
    build_exp_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    md_eval_scaled,
    proper_relax,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    NoAnalyzeParams,
    RunRequest,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    require_soc_handles,
)

from ._shared import read_ge_centers

# Domain T1Exp.analyze returns only a Figure (T1 in suptitle, no numeric return).
# This adapter is therefore figure-only with no writeback. If you need the T1
# value written to the MetaDict, use 'singleshot/t1_tone' whose domain analyze
# returns (t1, t1_b, fig).
SsT1RunResult: TypeAlias = T1Result


@dataclass
class SsT1AnalyzeResult(FigureOnlyAnalyzeResult):
    # Dual-transition-rate fit result in suptitle only; no numeric writeback.
    pass


class SsT1Adapter(
    BaseAdapter[T1Cfg, SsT1RunResult, SsT1AnalyzeResult, NoAnalyzeParams]
):
    exp_cls = T1Exp
    ExpCfg_cls: ClassVar[Any] = T1Cfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot T1: applies a π pulse, waits for a variable delay, "
            "reads out, and classifies each shot in-program against the |g>/|e> "
            "IQ-cluster centres. Repeats from both |g> and |e> initial states "
            "(Branch). Plots the ground / excited / other populations versus "
            "delay time with dual-transition-rate fits for T1 and T1_b. "
            "Runs on real hardware."
        ),
        expects_md=(
            "REQUIRES the single-shot discrimination calibration in the "
            "MetaDict — run 'singleshot/ge' first and apply its writeback so "
            "'g_center' / 'e_center' / 'ge_radius' are present; run "
            "fast-fails if any is missing. "
            "Optionally reads 'confusion_matrix' to readout-correct populations "
            "at analyze time; 't1' to seed the sweep stop (default 5*t1, "
            "fallback 100 us); 'q_f' / 'qub_ch' for the pi pulse; 'r_f' / "
            "'res_ch' / 'ro_ch' / 'timeFly' for readout."
        ),
        expects_ml=(
            "Needs a qubit pi-pulse module and a readout module. "
            "Optional reset (disabled when no library entry exists)."
        ),
        typical_writeback=(
            "No writeback — T1 is shown in the plot title only. "
            "Use 'singleshot/t1_tone' if you need the T1 scalar in the "
            "MetaDict."
        ),
        recommended=(
            "Run after 'singleshot/ge'. A delay sweep reaching ~5*T1 lets the "
            "decay flatten; with no prior 't1', the sweep spans 0–100 us. "
            "Set 'uniform=True' to sweep linearly (slower, evenly spaced); "
            "leave False for a log-spaced geomspace (better for wide T1 range)."
        ),
    )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        # ``uniform`` controls the sweep spacing inside the domain run: False →
        # log-spaced geomspace; True → linear uniform. Exposed via extra because it
        # is a run-only flag (not lowered into T1Cfg).
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "pi_pulse": make_pulse_module_spec(),
                "readout": make_readout_module_spec(),
            },
            sweep={"length": SweepSpec(label="Delay (us)")},
            extra={"uniform": ScalarSpec(label="Uniform (linear) sweep", type=bool)},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        sweep_stop = md_eval_scaled(ctx, "t1", factor=5.0, fallback=100.0)
        relax_delay = proper_relax(ctx)
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(
                reps=1000,
                rounds=10,
                relax_delay=relax_delay,
                uniform=False,  # domain default: log-spaced
            )
            .role("modules.pi_pulse", "pi_pulse")
            .role("modules.readout", "readout")
            .role("modules.reset", "reset", Init.DISABLED)
            .set_sweep(
                "sweep.length", SweepValue(start=0.0, stop=sweep_stop, expts=101)
            )
            .build()
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> T1Cfg:
        # Pop ``uniform`` before lowering — it is not part of T1Cfg.
        cfg_raw = dict(raw_cfg)
        cfg_raw.pop("uniform", None)
        return req.ml.make_cfg(cfg_raw, T1Cfg)

    def _uniform(self, raw_cfg: dict[str, object]) -> bool:
        value = raw_cfg.get("uniform", False)
        if not isinstance(value, bool):
            raise ValueError(f"'uniform' must be a bool, got {type(value).__name__}")
        return value

    def run(self, req: RunRequest, schema: CfgSchema) -> SsT1RunResult:
        # Override standard run: domain run needs GE centres + uniform kwarg.
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema.to_raw_dict(req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        g_center, e_center, radius = read_ge_centers(req.md)
        uniform = self._uniform(raw_cfg)
        return T1Exp().run(
            soc, soccfg, cfg, g_center, e_center, radius, uniform=uniform
        )

    def analyze(
        self, req: AnalyzeRequest[SsT1RunResult, NoAnalyzeParams]
    ) -> SsT1AnalyzeResult:
        # ``confusion_matrix`` is the GE 3×3 readout-correction matrix from md.
        # ``skip`` is not exposed as a user knob — users can re-run with a shorter
        # sweep instead.
        confusion = req.md.get("confusion_matrix")
        fig = T1Exp().analyze(req.run_result, confusion_matrix=confusion)
        return SsT1AnalyzeResult(figure=fig)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_ss_t1_{time.strftime('%m%d')}"
