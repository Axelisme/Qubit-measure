from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Any, ClassVar, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.rabi.amp_rabi import (
    AmpRabiCfg,
    AmpRabiExp,
    AmpRabiResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_module_spec,
    make_qub_probe_default,
    make_readout_module_spec,
    make_readout_ref_default,
    make_reset_module_spec,
    make_reset_ref_default,
    md_eval_scaled,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgNodeValue,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
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
class AmpRabiAnalyzeResult(AnalyzeResultBase):
    pi_amp: float
    pi2_amp: float
    figure: Figure


class AmpRabiAdapter(
    BaseAdapter[
        AmpRabiCfg,
        AmpRabiRunResult,
        AmpRabiAnalyzeResult,
        AmpRabiAnalyzeParams,
    ]
):
    exp_cls = AmpRabiExp
    ExpCfg_cls: ClassVar[Any] = AmpRabiCfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "Amplitude Rabi: drives the qubit at its known frequency and "
                "sweeps the drive-pulse gain, fitting the resulting Rabi "
                "oscillation to find the pi and pi/2 pulse amplitudes. Runs on "
                "real hardware. Run once you know the qubit frequency, to "
                "calibrate a pi pulse by amplitude (the pulse length stays fixed)."
            ),
            expects_md=(
                "Reads from the MetaDict (all optional, seeding defaults): 'q_f' — "
                "qubit frequency feeding the drive pulse (~2000–6000 MHz); "
                "'qub_ch' — qubit-drive channel; 'pi_amp' — prior pi-pulse gain, "
                "the gain sweep spans up to 2*pi_amp (fallback ~0.5). Readout "
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
                "Proposes the fitted pi-pulse gain into MetaDict 'pi_amp' and the "
                "pi/2-pulse gain into 'pi2_amp'. No ModuleLibrary writeback."
            ),
            recommended=(
                "Analysis takes a 'Skip points' count (default 0) to drop leading "
                "low-gain sweep points before the cosine fit; raise it only if the "
                "first few points are distorted. A gain sweep spanning roughly two "
                "pi amplitudes captures at least a full oscillation; widen it if "
                "no full period is visible."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return CfgSectionSpec(
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

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        sweep_stop = md_eval_scaled(ctx, "pi_amp", factor=2.0, fallback=0.5)
        root_val = CfgSectionValue(
            fields={
                "modules": CfgSectionValue(
                    fields={
                        "qub_pulse": make_qub_probe_default(ctx),
                        "readout": make_readout_ref_default(ctx),
                        # optional → DisabledRefValue when no library reset (ADR-0012)
                        "reset": make_reset_ref_default(ctx, optional=True),
                    }
                ),
                "reps": DirectValue(100),
                "rounds": DirectValue(100),
                "relax_delay": DirectValue(10.5),
                "sweep": CfgSectionValue(
                    fields={
                        "gain": SweepValue(start=-0.3, stop=sweep_stop, expts=51),
                    }
                ),
            }
        )
        return root_val

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
        return [
            MetaDictWriteback(
                target_name="pi_amp",
                description="Pi pulse gain (a.u.)",
                proposed_value=result.pi_amp,
            ),
            MetaDictWriteback(
                target_name="pi2_amp",
                description="Pi/2 pulse gain (a.u.)",
                proposed_value=result.pi2_amp,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_amp_rabi_{time.strftime('%m%d')}"
