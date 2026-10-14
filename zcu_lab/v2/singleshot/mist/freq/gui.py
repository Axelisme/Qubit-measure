from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import (
    read_confusion_matrix,
)
from zcu_lab.v2.singleshot.mist.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqDepExp,
    FreqResult,
)
from zcu_lab.v2.singleshot.mist.freq.form import make_form

MistFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class MistFreqAnalyzeResult(AnalyzeResultBase):
    pass


class MistFreqAdapter(
    BaseAdapter[FreqCfg, MistFreqRunResult, MistFreqAnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = FreqDepExp.persistence_tag()

    @staticmethod
    def make_form(ctx: FormContext) -> Form[FreqCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = FreqDepExp

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "MIST probe-frequency sweep: drives a probe pulse whose frequency is swept "
            "and classifies each shot in-program against the |g>/|e> single-shot centres, "
            "plotting the ground/excited/other populations versus probe frequency. Runs "
            "on real hardware; the result is already populations (no per-point fit)."
        ),
        expects_md=(
            "Open and Reset capture the current aliases, parameters and library. Relax "
            "delay is 5 * $qubit.t1. Probe sweep is $resonator.freq +/- 1.5 * "
            "$resonator.kappa. Classification seeds "
            "$qubit.ext.readout_cal.g_center/e_center/radius; missing calibration stays "
            "Invalid. Inline templates read $qubit.freq, $qubit.wiring.L01.ch and "
            "$resonator.freq/wiring.ch/wiring.ro_ch; readout trigger uses "
            "$resonator.ext.time_of_fly + 0.05 us. Missing required sources remain "
            "Invalid; no numeric fallback is supplied."
        ),
        expects_ml=(
            "Probe starts from res_probe. Optional init/reset use "
            "$qubit.module.x180/reset. Readout uses $qubit.module.readout or readout "
            "template. Sweep-overwritten pulse values remain editable and library keys "
            "stay selectable. Run uses this fixed capture."
        ),
        typical_writeback=(
            "No writeback — the population curves are read off the plot by eye."
        ),
        recommended=(
            "Set calibration cfg directly or seed it with 'singleshot/ge'. A frequency "
            "span around the resonator line captures the MIST response; use a large enough "
            "shot count (reps) for clean populations."
        ),
    )

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def run(
        self,
        req: RunRequest,
        cfg: FreqCfg,
        *,
        context: RunContext,
    ) -> MistFreqRunResult:
        del req
        return RunRecord(cfg, FreqDepExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MistFreqRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> MistFreqAnalyzeResult:
        options = FreqAnalyzeOptions(
            confusion_matrix=read_confusion_matrix(req.parameters),
        )
        FreqDepExp().analyze(req.run_result, options, plots=plots)
        return MistFreqAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_mist_freq_{time.strftime('%m%d')}"
