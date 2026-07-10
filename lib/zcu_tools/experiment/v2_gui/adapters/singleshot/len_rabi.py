from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.v2.singleshot.len_rabi import (
    LenRabiCfg,
    LenRabiExp,
    LenRabiResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    FigureOnlyAnalyzeResult,
    Init,
    build_exp_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    md_eval_scaled_or_value,
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
    SweepSpec,
    SweepValue,
    require_soc_handles,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict

from ._shared import read_ge_centers

# ``LenRabiExp`` from ``singleshot`` — sweeps the qubit-drive pulse *length* and
# classifies each shot in-program against |g>/|e> centres. The domain ``run``
# result is already populations (not raw IQ), so analyze only draws curves.
SsLenRabiRunResult: TypeAlias = LenRabiResult


@dataclass
class SsLenRabiAnalyzeResult(FigureOnlyAnalyzeResult):
    # Population-vs-length three-line curves; domain analyze returns only Figure.
    # No writeback: read the calibrated pulse length off the Rabi oscillation by
    # eye, then use twotone/rabi/len_rabi for numeric pi-length extraction.
    pass


class SsLenRabiAdapter(
    BaseAdapter[LenRabiCfg, SsLenRabiRunResult, SsLenRabiAnalyzeResult, NoAnalyzeParams]
):
    exp_cls = LenRabiExp
    ExpCfg_cls: ClassVar[Any] = LenRabiCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot Length Rabi: sweeps the qubit-drive pulse length and "
            "classifies each raw shot in-program against the |g>/|e> IQ-cluster "
            "centres, plotting the ground / excited / other populations versus "
            "pulse length. The result is already populations — no per-point "
            "fit is performed during analyze. Runs on real hardware."
        ),
        expects_md=(
            "REQUIRES the single-shot discrimination calibration in the "
            "MetaDict — run 'singleshot/ge' first and apply its writeback so "
            "'g_center' / 'e_center' / 'ge_radius' are present; run "
            "fast-fails if any is missing. "
            "Optionally reads 'confusion_matrix' to readout-correct the "
            "populations at analyze time; 'pi_len' to seed the sweep stop "
            "(4*pi_len when calibrated; fallback sweep 0.03–0.2 us); "
            "'q_f' / 'qub_ch' to seed the qubit-drive defaults."
        ),
        expects_ml=(
            "Needs a qubit drive-pulse module (qub_pulse) and a readout module. "
            "Optional reset (disabled when no library entry exists)."
        ),
        typical_writeback=(
            "No writeback — read the pi-pulse length off the Rabi plot by eye."
        ),
        recommended=(
            "Run after 'singleshot/ge'. A sweep spanning a few pi lengths "
            "captures a full oscillation. The three-line population plot lets "
            "you check leakage (other population)."
        ),
    )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        # cfg mirrors TwoToneModuleCfg (qub_pulse / readout / reset) + sweep.length.
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "qub_pulse": make_pulse_module_spec(),
                "readout": make_readout_module_spec(),
            },
            sweep={"length": SweepSpec(label="Length (us)")},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        sweep_stop = md_eval_scaled_or_value(ctx, "pi_len", factor=4.0, fallback=0.2)
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=1000, rounds=100, relax_delay=50.5)
            .role("modules.qub_pulse", "qub_probe", Init.INLINE)
            .set("modules.qub_pulse.gain", 1.0)
            .role("modules.readout", "readout")
            .role("modules.reset", "reset", Init.DISABLED)
            .set_sweep(
                "sweep.length", SweepValue(start=0.03, stop=sweep_stop, expts=51)
            )
            .build()
        )

    def run(self, req: RunRequest, schema: CfgSchema) -> SsLenRabiRunResult:
        # Override standard run: domain run needs the GE classification trio.
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema_to_raw_dict(schema, req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        g_center, e_center, radius = read_ge_centers(req.md)
        return LenRabiExp().run(soc, soccfg, cfg, g_center, e_center, radius)

    def analyze(
        self, req: AnalyzeRequest[SsLenRabiRunResult, NoAnalyzeParams]
    ) -> SsLenRabiAnalyzeResult:
        # ``confusion_matrix`` is an analyze input read from md (the GE 3×3
        # correction matrix); absent → None, which skips correction (domain
        # default).
        confusion = req.md.get("confusion_matrix")
        fig = LenRabiExp().analyze(req.run_result, confusion_matrix=confusion)
        return SsLenRabiAnalyzeResult(figure=fig)

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_ss_len_rabi_{time.strftime('%m%d')}"
