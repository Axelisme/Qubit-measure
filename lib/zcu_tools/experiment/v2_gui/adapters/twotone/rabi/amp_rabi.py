from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.rabi.amp_rabi import (
    AmpRabiCfg,
    AmpRabiExp,
    AmpRabiResult,
)
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    SweepDefault,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
    ModuleWriteback,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.main.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.cfg import (
    CfgSchema,
)

AmpRabiRunResult: TypeAlias = AmpRabiResult


@dataclass
class AmpRabiAnalyzeParams:
    skip: Annotated[int, ParamMeta(label="Skip points")] = 0


@dataclass
class AmpRabiAnalyzeResult(AnalyzeResultBase):
    # Summary keys are the GAIN scalars (pi_gain / pi2_gain) to match the glossary
    # (single_qubit.md) and the MetaDict writeback target; the 'pi_amp'/'pi2_amp'
    # names belong to the pi-pulse MODULES, not these scalars.
    pi_gain: float
    pi_gain_err: float
    pi2_gain: float
    pi2_gain_err: float
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

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
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
            "'qub_ch' — qubit-drive channel; 'pi_gain' — prior pi-pulse gain, "
            "the gain sweep spans up to 1.2*pi_gain (fallback 0.6). 'pi_len' "
            "seeds the qubit pulse length as 1.01*pi_len when present. Readout "
            "defaults pull 'r_f' (~4000–8000 MHz), 'res_ch' / 'ro_ch', and "
            "'timeFly' for the readout trigger offset (~0–1 us)."
        ),
        expects_ml=(
            "Needs a qubit drive-pulse module — prefers calibrated library "
            "pi pulses ('pi_len' / 'pi_amp') and falls back to a blank inline "
            "pulse — and a readout module that references a calibrated library "
            "readout ('readout_dpm' / 'readout_rf' / 'readout' / "
            "'res_readout') when present, else a blank pulse-readout that "
            "references the 'ro_waveform' waveform when one exists. Optional "
            "reset references a library reset ('reset_bath' / 'reset_10' / "
            "'reset_120') when present, else stays disabled."
        ),
        typical_writeback=(
            "Proposes MetaDict scalars 'pi_gain' and 'pi2_gain' (fitted "
            "gains). Also proposes ModuleLibrary modules 'pi_amp' and "
            "'pi2_amp' — copies of the qubit drive module with gain "
            "overridden to the fitted pi / pi/2 value. Module items are "
            "skipped when no cfg_snapshot is available (e.g. loaded from "
            "file)."
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
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse(
                "qub_pulse",
                role_id="rabi_pulse",
                label="Rabi Pulse",
                locked={"gain": 0.0},
                blank_overrides={
                    "waveform.length": scaled_md(
                        "pi_len",
                        factor=1.01,
                        fallback_value=1.1,
                    )
                },
            )
            .readout()
            .relax_delay(
                scaled_md("t1", factor=5.0, fallback_value=30.5),
            )
            .sweep(
                "gain",
                label="Gain (a.u.)",
                default=SweepDefault(
                    start=-0.3,
                    stop=scaled_md("pi_gain", factor=1.2, fallback_value=0.6),
                    expts=51,
                ),
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    def analyze(
        self, req: AnalyzeRequest[AmpRabiRunResult, AmpRabiAnalyzeParams]
    ) -> AmpRabiAnalyzeResult:
        params = req.analyze_params
        pi_amp, pi_amp_err, pi2_amp, pi2_amp_err, fig = AmpRabiExp().analyze(
            req.run_result,
            skip=params.skip,
        )
        return AmpRabiAnalyzeResult(
            pi_gain=pi_amp,
            pi_gain_err=pi_amp_err,
            pi2_gain=pi2_amp,
            pi2_gain_err=pi2_amp_err,
            figure=fig,
        )

    def get_writeback_items(
        self, req: WritebackRequest[AmpRabiRunResult, AmpRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            # Scalar gains write back as 'pi_gain'/'pi2_gain' per the naming
            # convention (single_qubit.md); the 'pi_amp'/'pi2_amp' names below
            # belong to the pi-pulse MODULES, not these scalars.
            MetaDictWriteback(
                target_name="pi_gain",
                description="Pi pulse gain (a.u.)",
                proposed_value=result.pi_gain,
            ),
            MetaDictWriteback(
                target_name="pi2_gain",
                description="Pi/2 pulse gain (a.u.)",
                proposed_value=result.pi2_gain,
            ),
        ]

        # Emit module writeback items when the run captured a cfg_snapshot.
        # Each item is a copy of qub_pulse with gain overridden to the fitted
        # pi / pi/2 value, registered as a library module for subsequent
        # experiments that reference the calibrated pulse by name.
        snapshot = req.run_result.cfg_snapshot
        if snapshot is not None:
            qub_pulse_cfg = snapshot.modules.qub_pulse
            # Module targets keep the 'pi_amp'/'pi2_amp' names; the gain values come
            # from the renamed pi_gain / pi2_gain scalar fields.
            for target, gain, desc in [
                ("pi_amp", result.pi_gain, "amp pi pulse"),
                ("pi2_amp", result.pi2_gain, "amp pi/2 pulse"),
            ]:
                spec, value = module_cfg_to_value(qub_pulse_cfg)
                value.with_field("gain", gain)
                items.append(
                    ModuleWriteback(
                        target_name=target,
                        description=desc,
                        edit_schema=CfgSchema(spec=spec, value=value),
                    )
                )

        return items

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_amp_rabi_{time.strftime('%m%d')}"
