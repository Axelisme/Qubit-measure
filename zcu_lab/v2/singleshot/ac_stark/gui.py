from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

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
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import (
    read_analysis_float,
    read_chi_kappa,
    read_confusion_matrix,
)
from zcu_lab.v2.singleshot.ac_stark.core import (
    AcStarkAnalyzeOptions,
    AcStarkCfg,
    AcStarkExp,
    AcStarkResult,
)
from zcu_lab.v2.singleshot.ac_stark.form import FORM

# Domain analysis returns a numeric coefficient and publishes its named fit. The AC-Stark
# coefficient is written back to the parameter context (key ``ac_stark_coeff``, matching
# single_qubit.md:3329) — it is the photon-number-per-gain² calibration the
# downstream MIST experiments read as ``ac_coeff``.
SsAcStarkRunResult: TypeAlias = RunRecord[AcStarkCfg, AcStarkResult]


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
    experiment_tag: ClassVar[str | None] = AcStarkExp.persistence_tag()
    form = FORM
    exp_cls = AcStarkExp
    ExpCfg_cls = AcStarkCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot AC-Stark calibration: drives a cavity Stark tone and a qubit "
            "probe tone while sweeping cavity gain and qubit frequency (2D), classifying "
            "each shot in-program against the |g>/|e> IQ-cluster centres. Fits the "
            "Stark-shifted resonance versus gain² to extract the AC-Stark coefficient "
            "(photon number per gain²). Runs on real hardware."
        ),
        expects_md=(
            "Open and Reset capture the current aliases, parameters and library. Relax "
            "delay starts at 5.5 us. Qubit sweep is $qubit.freq - 700 to $qubit.freq + "
            "100 MHz. Qubit probe length/mixer use "
            "$qubit.module.x180.waveform.length and $qubit.module.x180.mixer_freq. "
            "Pre/post delay are "
            "1.0/$resonator.kappa and 0.5/$resonator.kappa; cavity tone length adds "
            "1.0/$resonator.kappa to the pi length. Classification seeds "
            "$qubit.ext.readout_cal.g_center/e_center/radius; missing calibration stays "
            "Invalid. Inline templates read $qubit.freq, $qubit.wiring.L01.ch and "
            "$resonator.freq/wiring.ch/wiring.ro_ch; readout trigger uses "
            "$resonator.ext.time_of_fly + 0.05 us. Missing required sources remain "
            "Invalid; no numeric fallback is supplied."
        ),
        expects_ml=(
            "Reset and init start disabled. Cavity/qubit tones start from "
            "res_probe/pi_pulse with calibrated timing. Readout uses "
            "$qubit.module.readout or readout template. Sweep-overwritten pulse values "
            "remain editable and library keys stay selectable. Run uses this fixed "
            "capture."
        ),
        typical_writeback=(
            "Proposes the fitted AC-Stark coefficient into parameter context "
            "'$qubit.ext.ac_stark_coeff' (photon number per gain²); MIST reads it to "
            "rescale their x-axis to photon number."
        ),
        recommended=(
            "Set calibration cfg directly or seed it with 'singleshot/ge'; run after the "
            "dispersive-shift experiment has set 'chi' / 'rf_w'. Sweep the Stark gain "
            "across the onset and the probe frequency around the qubit line."
        ),
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
                target_name="$qubit.ext.ac_stark_coeff",
                description="AC-Stark coefficient (photon number per gain²)",
                proposed_value=req.analyze_result.ac_stark_coeff,
            ),
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_sh_ac_stark_{time.strftime('%m%d')}"
