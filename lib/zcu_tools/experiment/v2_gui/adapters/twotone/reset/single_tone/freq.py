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
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    build_exp_spec,
    make_pulse_module_spec,
    make_pulse_reset_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    proper_reset_freq_range,
    reset_module_writeback_items,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    MetaDictWriteback,
    NoAnalyzeParams,
    SweepSpec,
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

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
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
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "init_pulse": make_pulse_module_spec(optional=True),
                # The sweep axis owns the tested-reset frequency
                # (set_param("freq") at run); lock it so the form does not show a
                # field the sweep silently overwrites (notebook: freq=0.0).
                "tested_reset": make_pulse_reset_module_spec().lock_literal(
                    "pulse_cfg.freq", 0.0
                ),
                "readout": make_readout_module_spec(),
            },
            sweep={"freq": SweepSpec(label="Freq (MHz)")},
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=100, rounds=100, relax_delay=1.0)
            .role("modules.tested_reset", "pulse_reset")
            .role("modules.readout", "readout", prefer_blank=True)
            # optional → None (disabled) when no library entry (ADR-0010)
            .role("modules.reset", "reset", optional=True)
            .role("modules.init_pulse", "pi_pulse", optional=True)
            .set_sweep("sweep.freq", proper_reset_freq_range(ctx, 201))
            .build()
        )

    def get_analyze_params(
        self, result: SingleToneFreqRunResult, ctx: ExpContext
    ) -> NoAnalyzeParams:
        return NoAnalyzeParams()

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
