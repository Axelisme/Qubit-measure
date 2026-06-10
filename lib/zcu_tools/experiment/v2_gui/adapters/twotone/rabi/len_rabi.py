from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Any, ClassVar, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.rabi.len_rabi import (
    LenRabiCfg,
    LenRabiExp,
    LenRabiResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    build_exp_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    md_eval_scaled,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    SweepSpec,
    SweepValue,
    WritebackItem,
    WritebackRequest,
)

LenRabiRunResult: TypeAlias = LenRabiResult


@dataclass
class LenRabiAnalyzeParams:
    decay: Annotated[bool, ParamMeta(label="Fit decay envelope")]


@dataclass
class LenRabiAnalyzeResult(AnalyzeResultBase):
    pi_len: float
    pi2_len: float
    figure: Figure


class LenRabiAdapter(
    BaseAdapter[
        LenRabiCfg,
        LenRabiRunResult,
        LenRabiAnalyzeResult,
        LenRabiAnalyzeParams,
    ]
):
    exp_cls = LenRabiExp
    ExpCfg_cls: ClassVar[Any] = LenRabiCfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "Length Rabi: drives the qubit at its known frequency and sweeps "
                "the drive-pulse length, fitting the Rabi oscillation to find the "
                "pi and pi/2 pulse lengths. Runs on real hardware. Run once you "
                "know the qubit frequency, to calibrate a pi pulse by duration "
                "(the pulse gain stays fixed)."
            ),
            expects_md=(
                "Reads from the MetaDict (all optional, seeding defaults): 'q_f' — "
                "qubit frequency feeding the drive pulse (~2000–6000 MHz); "
                "'qub_ch' — qubit-drive channel; 'pi_len' — prior pi-pulse length, "
                "the length sweep spans up to 4*pi_len (fallback ~0.1 us). Readout "
                "defaults pull 'r_f' (~4000–8000 MHz), 'res_ch' / 'ro_ch', and "
                "'timeFly' for the readout trigger offset (~0–1 us)."
            ),
            expects_ml=(
                "Needs a qubit drive-pulse module (defaults to a blank inline "
                "pulse) and a readout module — references a calibrated library "
                "readout ('readout_dpm' / 'readout_rf' / 'readout' / "
                "'res_readout') when present, else a blank pulse-readout that "
                "references the 'ro_waveform' waveform when one exists. Optional "
                "reset references a library reset ('reset_bath' / 'reset_10' / "
                "'reset_120') when present, else stays disabled."
            ),
            typical_writeback=(
                "Proposes the fitted pi-pulse length into MetaDict 'pi_len' and "
                "the pi/2-pulse length into 'pi2_len'. No ModuleLibrary writeback."
            ),
            recommended=(
                "Analysis defaults to fitting a decay envelope on the oscillation; "
                "keep it on when the Rabi oscillation visibly damps over the "
                "sweep, turn it off for a pure undamped cosine fit. A length sweep "
                "spanning a few pi lengths (up to 4*pi_len) captures a full "
                "oscillation; widen it if no full period is visible."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "qub_pulse": make_pulse_module_spec(),
                "readout": make_readout_module_spec(),
            },
            sweep={"length": SweepSpec(label="Length (us)")},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        sweep_stop = md_eval_scaled(ctx, "pi_len", factor=4.0, fallback=0.1)
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=100, rounds=100, relax_delay=10.5)
            .role("modules.qub_pulse", "qub_probe", prefer_blank=True)
            .role("modules.readout", "readout")
            # optional → None (disabled) when no library reset (ADR-0010)
            .role("modules.reset", "reset", optional=True)
            .set_sweep(
                "sweep.length", SweepValue(start=0.03, stop=sweep_stop, expts=101)
            )
            .build()
        )

    def get_analyze_params(
        self, result: LenRabiRunResult, ctx: ExpContext
    ) -> LenRabiAnalyzeParams:
        return LenRabiAnalyzeParams(decay=True)

    def analyze(
        self, req: AnalyzeRequest[LenRabiRunResult, LenRabiAnalyzeParams]
    ) -> LenRabiAnalyzeResult:
        params = req.analyze_params
        pi_len, pi2_len, _, fig = LenRabiExp().analyze(
            req.run_result,
            decay=params.decay,
        )
        return LenRabiAnalyzeResult(pi_len=pi_len, pi2_len=pi2_len, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[LenRabiRunResult, LenRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            MetaDictWriteback(
                target_name="pi_len",
                description="Pi pulse length (us)",
                proposed_value=result.pi_len,
            ),
            MetaDictWriteback(
                target_name="pi2_len",
                description="Pi/2 pulse length (us)",
                proposed_value=result.pi2_len,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_len_rabi_{time.strftime('%m%d')}"
