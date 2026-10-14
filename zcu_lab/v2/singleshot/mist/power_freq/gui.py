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
from zcu_lab.v2.singleshot.mist.power_freq.core import (
    FreqPowerAnalyzeOptions,
    FreqPowerCfg,
    FreqPowerExp,
    FreqPowerResult,
)
from zcu_lab.v2.singleshot.mist.power_freq.form import make_form

MistPowerFreqRunResult: TypeAlias = RunRecord[FreqPowerCfg, FreqPowerResult]


@dataclass
class MistPowerFreqAnalyzeResult(AnalyzeResultBase):
    pass


class MistPowerFreqAdapter(
    BaseAdapter[
        FreqPowerCfg,
        MistPowerFreqRunResult,
        MistPowerFreqAnalyzeResult,
        NoAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = FreqPowerExp.persistence_tag()

    @staticmethod
    def make_form(ctx: FormContext) -> Form[FreqPowerCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = FreqPowerExp

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "MIST probe power × frequency landscape: drives a probe pulse while sweeping "
            "its gain and frequency (2D), classifying each shot in-program against the "
            "|g>/|e> single-shot centres, and plotting the ground/excited/other "
            "populations as 2D maps over (gain, freq). Runs on real hardware; the result "
            "is already populations (no per-point fit)."
        ),
        expects_md=(
            "Open and Reset capture the current aliases, parameters and library. Relax "
            "delay is 5 * $qubit.t1. Probe frequency is $resonator.freq +/- 1.5 * "
            "$resonator.kappa; gain sweep is 0 to 1. Classification seeds "
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
            "No writeback — the population landscapes are read off the plot by eye."
        ),
        recommended=(
            "Set calibration cfg directly or seed it with 'singleshot/ge'. Sweep the "
            "probe gain across the MIST onset and the frequency around the "
            "qubit/resonator line of interest."
        ),
    )

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def run(
        self,
        req: RunRequest,
        cfg: FreqPowerCfg,
        *,
        context: RunContext,
    ) -> MistPowerFreqRunResult:
        del req
        return RunRecord(cfg, FreqPowerExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MistPowerFreqRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> MistPowerFreqAnalyzeResult:
        options = FreqPowerAnalyzeOptions(
            confusion_matrix=read_confusion_matrix(req.parameters),
        )
        FreqPowerExp().analyze(req.run_result, options, plots=plots)
        return MistPowerFreqAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_mist_power_freq_{time.strftime('%m%d')}"
