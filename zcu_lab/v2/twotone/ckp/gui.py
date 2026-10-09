"""CKP GUI adapter for ground/excited-resolved two-tone spectroscopy."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    MetaDictWriteback,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import EvalValue, SweepValue
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
)
from zcu_lab.v2._support.measure.seeds import custom, res_freq_range
from zcu_lab.v2.twotone.ckp.core import CKP_Cfg, CKP_Exp, CKP_Result

_CKP_SWEEP_EXPTS = 101
_CKP_QUB_PULSE_LENGTH_US = 1.5


def _rf_width_timing_seed(
    _ctx: SessionEnv, *, numerator: float, offset: float = 0.0
) -> EvalValue:
    coefficient = numerator / (2 * math.pi)
    expression = f"{coefficient} / $resonator.kappa"
    if offset:
        expression = f"{offset} + {expression}"
    return EvalValue(expr=expression)


def _res_pulse_length_seed(ctx: SessionEnv) -> float | EvalValue:
    return _rf_width_timing_seed(
        ctx,
        numerator=5.1,
        offset=_CKP_QUB_PULSE_LENGTH_US,
    )


def _qub_pre_delay_seed(ctx: SessionEnv) -> float | EvalValue:
    return _rf_width_timing_seed(ctx, numerator=5.0)


def _qub_post_delay_seed(ctx: SessionEnv) -> float | EvalValue:
    return _rf_width_timing_seed(ctx, numerator=3.1)


def _qub_freq_sweep_seed(_ctx: SessionEnv) -> SweepValue:
    return SweepValue(
        start=EvalValue("$qubit.freq - 10"),
        stop=EvalValue("$qubit.freq + 5"),
        expts=_CKP_SWEEP_EXPTS,
    )


CKPRunResult: TypeAlias = RunRecord[CKP_Cfg, CKP_Result]


@dataclass
class CKPAnalyzeResult(AnalyzeResultBase):
    chi: float
    kappa: float
    res_freq: float


class CKPAdapter(BaseAdapter[CKP_Cfg, CKPRunResult, CKPAnalyzeResult, NoAnalyzeParams]):
    exp_cls = CKP_Exp
    ExpCfg_cls: ClassVar[Any] = CKP_Cfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "CKP ground/excited-resolved spectroscopy: prepares the qubit in g "
            "and e, drives resonator and qubit probe tones concurrently over a "
            "two-dimensional frequency sweep, and fits the two resonator branches "
            "to extract dispersive shift chi, linewidth kappa, and the readout "
            "frequency. Runs on real hardware."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional, seeding fresh cfg defaults): "
            "'r_f' and 'rf_w' set the resonator sweep and pulse timing; 'q_f' sets "
            "the qubit-probe sweep; 'qub_ch' / 'qub_1_4_ch' / 'qub_4_5_ch' select "
            "the qubit drive channel; 'res_ch' / 'ro_ch' select readout channels; "
            "'timeFly' seeds the readout trigger offset."
        ),
        expects_ml=(
            "Needs a calibrated qubit pi pulse (typically 'pi_amp' or 'pi_len'), "
            "an inline resonator-drive pulse, an inline qubit-probe pulse, and a "
            "pulse-readout module (typically 'readout_dpm' or 'readout_rf'). "
            "Optionally references a calibrated reset module."
        ),
        typical_writeback=(
            "Proposes fitted chi into MetaDict 'chi', fitted kappa into 'rf_w', "
            "and fitted readout frequency into 'readout_f'. No ModuleLibrary "
            "writeback."
        ),
        recommended=(
            "Run only after qubit frequency, pi pulse, resonator frequency, and "
            "resonator linewidth are calibrated. Review both g/e maps and fitted "
            "branches before applying writeback; a branch-selection or fit failure "
            "can produce plausible-looking scalar values."
        ),
    )

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, context: RunContext
    ) -> CKPRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = CKP_Exp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("pi_pulse", role_id="pi_pulse", label="Pi Pulse")
            .pulse(
                "res_pulse",
                role_id="res_probe",
                label="Resonator Drive Pulse",
                init=ModuleInit.INLINE,
                overrides={
                    "waveform.length": custom(
                        _res_pulse_length_seed,
                        description="CKP resonator-drive pulse length",
                    ),
                    "gain": 0.015,
                },
                locked={"freq": 0.0},
            )
            .pulse(
                "qub_pulse",
                role_id="qub_probe",
                label="Qubit Probe Pulse",
                init=ModuleInit.INLINE,
                overrides={
                    "waveform.length": _CKP_QUB_PULSE_LENGTH_US,
                    "gain": 0.01,
                    "pre_delay": custom(
                        _qub_pre_delay_seed,
                        description="CKP qubit-probe pre-delay",
                    ),
                    "post_delay": custom(
                        _qub_post_delay_seed,
                        description="CKP qubit-probe post-delay",
                    ),
                },
                locked={"freq": 0.0},
            )
            .readout()
            .relax_delay(10.1)
            .sweep(
                "res_freq",
                label="Resonator drive freq (MHz)",
                default=res_freq_range(expts=_CKP_SWEEP_EXPTS),
            )
            .sweep(
                "qub_freq",
                label="Qubit probe freq (MHz)",
                default=custom(
                    _qub_freq_sweep_seed,
                    description="CKP qubit-probe frequency range",
                ),
            )
            .reps(100)
            .rounds(100)
            .build()
        )

    def analyze(
        self, req: AnalyzeRequest[CKPRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> CKPAnalyzeResult:
        analysis = CKP_Exp().analyze(req.run_result, None, plots=plots)
        return CKPAnalyzeResult(
            chi=analysis.chi,
            kappa=analysis.kappa,
            res_freq=analysis.res_freq,
        )

    def get_writeback_items(
        self, req: WritebackRequest[CKPRunResult, CKPAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            MetaDictWriteback(
                target_name="chi",
                description="Dispersive shift chi (MHz)",
                proposed_value=result.chi,
            ),
            MetaDictWriteback(
                target_name="$resonator.kappa",
                description="Resonator linewidth kappa (MHz)",
                proposed_value=result.kappa,
            ),
            MetaDictWriteback(
                target_name="readout_f",
                description="Readout frequency (MHz)",
                proposed_value=result.res_freq,
            ),
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ckp_{time.strftime('%m%d')}"
