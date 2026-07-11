from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.reset.single_tone.freq import (
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    md,
    reset_freq_range,
    reset_module_writeback_items,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
    NoAnalyzeParams,
    WritebackItem,
    WritebackRequest,
)

SingleToneFreqRunResult: TypeAlias = FreqResult


@dataclass
class SingleToneFreqAnalyzeResult(AnalyzeResultBase):
    freq: float
    fwhm: float
    figure: Figure


class SingleToneFreqAdapter(
    BaseAdapter[
        FreqCfg,
        SingleToneFreqRunResult,
        SingleToneFreqAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    exp_cls = FreqExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-tone sideband-reset frequency sweep: drives the tested "
            "pulse-reset tone and sweeps its frequency, fitting the qubit "
            "response (Lorentzian) to find the reset sideband frequency and "
            "its linewidth. Runs on real hardware. Typically run after an "
            "init pulse excites the qubit, to locate the f01+f12 (r_f - q_f) "
            "sideband at which the reset pulse dumps the excitation."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'reset_f' — the sideband "
            "reset frequency (= r_f - q_f), the sweep centre; 'resetf_w' — its "
            "linewidth, setting the half-span as 1.5*resetf_w. Absent "
            "'resetf_w' → a fixed ±50 MHz span around 'reset_f'; absent "
            "'reset_f' → centred on 0 MHz. The tested reset and init pulse "
            "pull 'q_f' / 'qub_ch' for their drive defaults."
        ),
        expects_ml=(
            "Needs a pulse-reset module (the tested reset) and a readout "
            "module. Optionally references a calibrated upstream reset and an "
            "init pulse (a library pi pulse when present) — both disabled when "
            "no library entry exists."
        ),
        typical_writeback=(
            "Proposes the fitted sideband frequency into MetaDict 'reset_f' "
            "and the fitted linewidth (FWHM) into 'resetf_w'. No ModuleLibrary "
            "writeback — registering the calibrated reset module is left to "
            "the user once both the frequency and the length are dialled in."
        ),
        recommended=(
            "A sweep of ~201 points spanning a couple of linewidths around "
            "the expected sideband captures the dip cleanly; widen the span "
            "if 'reset_f' is only a rough r_f - q_f estimate. The reset-pulse "
            "length is held fixed here — sweep it separately afterwards."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("init_pulse", role_id="pi_pulse", optional=True)
            .reset(
                "tested_reset",
                role_id="reset",
                label="Tested Reset",
                shape="pulse",
                init=ModuleInit.INLINE,
                locked={"pulse_cfg.freq": 0.0},
                overrides={
                    "pulse_cfg.gain": 0.3,
                    "pulse_cfg.waveform.length": 5.0,
                    "pulse_cfg.post_delay": md(
                        "rf_w",
                        expr="5.0 / (2 * 3.141592653589793 * rf_w)",
                        fallback=0.8,
                    ),
                },
            )
            .readout()
            .relax_delay(md("t1", expr="1.0 * t1", fallback=30.5))
            .sweep(
                "freq",
                label="Freq (MHz)",
                default=reset_freq_range(expts=201),
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    # No get_analyze_params override: NoAnalyzeParams (the 4th generic arg) makes
    # BaseAdapter return the empty params instance and reflect the type.

    def analyze(
        self, req: AnalyzeRequest[SingleToneFreqRunResult, NoAnalyzeParams]
    ) -> SingleToneFreqAnalyzeResult:
        freq, fwhm, fig = FreqExp().analyze(req.run_result)
        return SingleToneFreqAnalyzeResult(freq=freq, fwhm=fwhm, figure=fig)

    def get_writeback_items(
        self,
        req: WritebackRequest[SingleToneFreqRunResult, SingleToneFreqAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            MetaDictWriteback(
                target_name="reset_f",
                description="Sideband reset frequency (MHz)",
                proposed_value=result.freq,
            ),
            MetaDictWriteback(
                target_name="resetf_w",
                description="Reset sideband linewidth FWHM (MHz)",
                proposed_value=result.fwhm,
            ),
        ]
        # Gated per-experiment 'reset_10' proposal: when md carries the calibrated
        # sideband freq, register the calibrated single-pulse reset built from this
        # run's tested_reset template (md overwrites its freq).
        items.extend(
            reset_module_writeback_items(
                req.ctx,
                req.run_result.cfg_snapshot,
                target="reset_10",
                field_md_map=[("pulse_cfg.freq", "reset_f")],
                desc="Reset with one pulse from 1 to 0",
            )
        )
        return items

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_sidereset_freq_{time.strftime('%m%d')}"
