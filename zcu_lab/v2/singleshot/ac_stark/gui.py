from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    ParameterWriteback,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import (
    EvalValue,
    ScalarSpec,
    SweepValue,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
)
from zcu_lab.v2._support.measure.seeds import custom
from zcu_lab.v2._support.measure.singleshot_shared import (
    read_analysis_float,
    read_chi_kappa,
    read_confusion_matrix,
    readout_probe_freq,
)
from zcu_lab.v2.singleshot.ac_stark.core import (
    AcStarkAnalyzeOptions,
    AcStarkCfg,
    AcStarkExp,
    AcStarkResult,
)

# Domain analysis returns a numeric coefficient and publishes its named fit. The AC-Stark
# coefficient is written back to the parameter context (key ``ac_stark_coeff``, matching
# single_qubit.md:3329) — it is the photon-number-per-gain² calibration the
# downstream MIST experiments read as ``ac_coeff``.
SsAcStarkRunResult: TypeAlias = RunRecord[AcStarkCfg, AcStarkResult]


def _probe_length(_ctx: SessionEnv) -> EvalValue:
    return EvalValue("$qubit.module.x180.waveform.length")


def _qubit_frequency(_ctx: SessionEnv) -> EvalValue:
    return EvalValue("$qubit.freq")


def _qubit_channel(_ctx: SessionEnv) -> EvalValue:
    return EvalValue("$qubit.wiring.L01.ch")


def _qubit_mixer_frequency(_ctx: SessionEnv) -> EvalValue:
    return EvalValue("$qubit.module.x180.mixer_freq")


def _rf_width_timing(_ctx: SessionEnv, *, numerator: float) -> EvalValue:
    coefficient = numerator / (2 * math.pi)
    return EvalValue(f"{coefficient} / $resonator.kappa")


def _cavity_tone_length(_ctx: SessionEnv) -> EvalValue:
    coefficient = 5.1 / (2 * math.pi)
    return EvalValue(
        f"$qubit.module.x180.waveform.length + {coefficient} / $resonator.kappa"
    )


def _qubit_pre_delay(ctx: SessionEnv) -> EvalValue:
    return _rf_width_timing(ctx, numerator=5.0)


def _qubit_post_delay(ctx: SessionEnv) -> EvalValue:
    return _rf_width_timing(ctx, numerator=3.1)


def _freq_sweep_default(_ctx: SessionEnv) -> SweepValue:
    return SweepValue(
        start=EvalValue(expr="$qubit.freq - 700.0"),
        stop=EvalValue(expr="$qubit.freq + 100.0"),
        expts=801,
    )


@dataclass
class SsAcStarkAnalyzeResult(AnalyzeResultBase):
    ac_stark_coeff: float


class SsAcStarkAdapter(
    BaseAdapter[
        AcStarkCfg,
        SsAcStarkRunResult,
        SsAcStarkAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    exp_cls = AcStarkExp
    ExpCfg_cls: ClassVar[Any] = AcStarkCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot AC-Stark calibration: drives a cavity Stark tone and a "
            "qubit probe tone while sweeping cavity gain and qubit frequency (2D), "
            "classifying each shot in-program against the |g>/|e> IQ-cluster "
            "centres. Fits the Stark-shifted resonance versus gain² to extract "
            "the AC-Stark coefficient (photon number per gain²). Runs on real "
            "hardware."
        ),
        expects_md=(
            "Run classification defaults reference $qubit.ext.readout_cal.g_center, "
            "$qubit.ext.readout_cal.e_center and $qubit.ext.readout_cal.radius. "
            "Probe frequency/channel use $qubit.freq and $qubit.wiring.L01.ch; "
            "probe length/mixer use $qubit.module.x180.waveform.length and "
            "$qubit.module.x180.mixer_freq. Cavity timing uses $resonator.kappa. "
            "Missing aliases or calibration leave cfg Invalid. Analyze calibration "
            "sources are not yet migrated to the historical Run contract."
        ),
        expects_ml=(
            "Uses $qubit.module.readout for SMART readout. Cavity and qubit probe "
            "tones are inline with canonical scalar calibration references; reset "
            "and init pulse start disabled."
        ),
        typical_writeback=(
            "Proposes the fitted AC-Stark coefficient into parameter context "
            "'ac_stark_coeff' (photon number per gain²); the MIST experiments "
            "read it to rescale their x-axis to photon number."
        ),
        recommended=(
            "Set calibration cfg directly or seed it with 'singleshot/ge'; run "
            "after the dispersive-shift "
            "experiment has set 'chi' / 'rf_w'. Sweep the Stark gain across the "
            "onset and the probe frequency around the qubit line."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True, init=ModuleInit.DISABLED)
            .pulse(
                "init_pulse",
                role_id="pi_pulse",
                optional=True,
                init=ModuleInit.DISABLED,
            )
            .pulse(
                "stark_pulse1",
                role_id="res_probe",
                label="Cavity Stark Tone (gain sweep)",
                init=ModuleInit.INLINE,
                overrides={
                    "freq": custom(
                        readout_probe_freq,
                        description="cavity Stark tone frequency",
                    ),
                    "gain": 0.0,
                    "waveform.length": custom(
                        _cavity_tone_length,
                        description="cavity Stark tone length",
                    ),
                },
            )
            .pulse(
                "stark_pulse2",
                role_id="pi_pulse",
                label="Qubit Probe Tone (freq sweep)",
                init=ModuleInit.INLINE,
                overrides={
                    "waveform.length": custom(
                        _probe_length, description="calibrated probe length"
                    ),
                    "freq": custom(
                        _qubit_frequency,
                        description="qubit probe frequency",
                    ),
                    "ch": custom(
                        _qubit_channel,
                        description="qubit probe wiring channel",
                    ),
                    "mixer_freq": custom(
                        _qubit_mixer_frequency,
                        description="calibrated qubit probe mixer frequency",
                    ),
                    "pre_delay": custom(
                        _qubit_pre_delay,
                        description="qubit probe pre-delay",
                    ),
                    "post_delay": custom(
                        _qubit_post_delay,
                        description="qubit probe post-delay",
                    ),
                },
            )
            .readout()
            .relax_delay(5.5)
            .sweep(
                "gain",
                label="Cavity Stark gain (a.u.)",
                default=SweepValue(start=0.0, stop=0.22, expts=301),
            )
            .sweep(
                "freq",
                label="Qubit probe frequency (MHz)",
                default=custom(
                    _freq_sweep_default,
                    description="AC-Stark probe frequency range",
                ),
            )
            .field(
                "g_center",
                spec=ScalarSpec("Ground center", complex),
                default=EvalValue("$qubit.ext.readout_cal.g_center"),
            )
            .field(
                "e_center",
                spec=ScalarSpec("Excited center", complex),
                default=EvalValue("$qubit.ext.readout_cal.e_center"),
            )
            .field(
                "radius",
                spec=ScalarSpec("Classification radius", float),
                default=EvalValue("$qubit.ext.readout_cal.radius"),
            )
            .reps(1000)
            .rounds(2)
            .build()
        )

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> SsAcStarkRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = AcStarkExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[SsAcStarkRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> SsAcStarkAnalyzeResult:
        # Required chi/kappa and optional readout correction/cutoff use this
        # Run's fixed capture. The core derives eta and owns fit defaults.
        chi, kappa = read_chi_kappa(req.parameters)
        confusion = read_confusion_matrix(req.parameters)
        cutoff = read_analysis_float(req.parameters, "$qubit.ext.cutoff")
        analysis = AcStarkExp().analyze(
            req.run_result,
            AcStarkAnalyzeOptions(
                chi=chi, kappa=kappa, confusion_matrix=confusion, cutoff=cutoff
            ),
            plots=plots,
        )
        return SsAcStarkAnalyzeResult(ac_stark_coeff=analysis.ac_stark_coeff)

    def get_writeback_items(
        self,
        req: WritebackRequest[SsAcStarkRunResult, SsAcStarkAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        # Key ``ac_stark_coeff`` per single_qubit.md:3329.
        return [
            ParameterWriteback(
                target_name="ac_stark_coeff",
                description="AC-Stark coefficient (photon number per gain²)",
                proposed_value=req.analyze_result.ac_stark_coeff,
            ),
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_sh_ac_stark_{time.strftime('%m%d')}"
