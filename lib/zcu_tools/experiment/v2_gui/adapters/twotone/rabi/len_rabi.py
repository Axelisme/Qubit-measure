from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

from matplotlib.figure import Figure

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
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    MetaDictWriteback,
    ModuleWriteback,
    ParamMeta,
    SweepSpec,
    SweepValue,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.main.cfg_schemas import module_cfg_to_value

LenRabiRunResult: TypeAlias = LenRabiResult


@dataclass
class LenRabiAnalyzeParams:
    decay: Annotated[bool, ParamMeta(label="Fit decay envelope")]


@dataclass
class LenRabiAnalyzeResult(AnalyzeResultBase):
    pi_len: float
    pi2_len: float
    # Rabi oscillation frequency in MHz (1/us), captured from the fit but
    # previously discarded; now preserved for writeback as 'rabi_f'.
    rabi_f: float
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
                "Proposes MetaDict scalars 'pi_len', 'pi2_len' (fitted lengths), "
                "and 'rabi_f' (Rabi oscillation frequency in MHz). Also proposes "
                "ModuleLibrary modules 'pi_len' and 'pi2_len' — length-calibrated "
                "pi-pulse modules produced by this adapter — with waveform length "
                "overridden to the fitted pi / pi/2 value. amp_rabi produces the "
                "separate 'pi_amp'/'pi2_amp' gain-calibrated modules and seeds its "
                "gain sweep from the 'pi_len' scalar written here. Module items are "
                "skipped when no cfg_snapshot is available (e.g. loaded from file)."
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
        pi_len, pi2_len, rabi_f, fig = LenRabiExp().analyze(
            req.run_result,
            decay=params.decay,
        )
        return LenRabiAnalyzeResult(
            pi_len=pi_len, pi2_len=pi2_len, rabi_f=rabi_f, figure=fig
        )

    def get_writeback_items(
        self, req: WritebackRequest[LenRabiRunResult, LenRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
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
            MetaDictWriteback(
                target_name="rabi_f",
                description="Rabi oscillation frequency (MHz)",
                proposed_value=result.rabi_f,
            ),
        ]

        # Emit module writeback items when the run captured a cfg_snapshot.
        # Each item is a copy of qub_pulse with waveform.length overridden to
        # the fitted pi / pi/2 value, registered as a library module for
        # subsequent experiments that reference the calibrated pulse by name.
        # These are the LENGTH-calibrated modules 'pi_len'/'pi2_len' produced
        # by this adapter (single_qubit.md); amp_rabi produces the separate
        # gain-calibrated 'pi_amp'/'pi2_amp' modules and reads the 'pi_len'
        # SCALAR (also written above) as its gain-sweep default-value seed.
        snapshot = req.run_result.cfg_snapshot
        if snapshot is not None:
            qub_pulse_cfg = snapshot.modules.qub_pulse
            for target, length, desc in [
                ("pi_len", result.pi_len, "len pi pulse"),
                ("pi2_len", result.pi2_len, "len pi/2 pulse"),
            ]:
                spec, value = module_cfg_to_value(qub_pulse_cfg)
                value.with_field("waveform.length", length)
                items.append(
                    ModuleWriteback(
                        target_name=target,
                        description=desc,
                        edit_schema=CfgSchema(spec=spec, value=value),
                    )
                )

        return items

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_len_rabi_{time.strftime('%m%d')}"
