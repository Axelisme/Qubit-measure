from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import (
    read_analysis_float,
    read_confusion_matrix,
)
from zcu_lab.v2.singleshot.mist.power.core import (
    PowerAnalyzeOptions,
    PowerCfg,
    PowerExp,
    PowerResult,
)
from zcu_lab.v2.singleshot.mist.power.form import FORM

MistPowerRunResult: TypeAlias = RunRecord[PowerCfg, PowerResult]


@dataclass
class MistPowerAnalyzeResult(AnalyzeResultBase):
    pass


@dataclass
class MistPowerAnalyzeParams:
    """MIST plot settings; log_scale uses a logarithmic gain/photon-number axis."""

    log_scale: Annotated[bool, ParamMeta(label="Logarithmic x-axis")] = False


class MistPowerAdapter(
    BaseAdapter[
        PowerCfg, MistPowerRunResult, MistPowerAnalyzeResult, MistPowerAnalyzeParams
    ]
):
    experiment_tag: ClassVar[str | None] = PowerExp.persistence_tag()
    form = FORM
    exp_cls = PowerExp
    ExpCfg_cls = PowerCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "MIST probe-power sweep: drives a probe pulse whose gain is swept and "
            "classifies each shot in-program against the |g>/|e> single-shot centres, "
            "plotting the ground/excited/other populations versus probe gain (or photon "
            "number when an AC-Stark coefficient is known). Runs on real hardware; the "
            "result is already populations (no per-point fit)."
        ),
        expects_md=(
            "Open and Reset capture the current aliases, parameters and library. Relax "
            "delay is 5 * $qubit.t1. Probe gain sweep runs from 0 to 1. Classification "
            "seeds $qubit.ext.readout_cal.g_center/e_center/radius; missing calibration "
            "stays Invalid. Inline templates read $qubit.freq, $qubit.wiring.L01.ch and "
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
            "Set calibration cfg directly or seed it with 'singleshot/ge'. Sweep the "
            "probe gain across the MIST onset; provide 'ac_stark_coeff' (from the "
            "AC-Stark experiment) for a photon-number x-axis."
        ),
    )

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> MistPowerRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, PowerExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MistPowerRunResult, MistPowerAnalyzeParams],
        *,
        plots: Plots,
    ) -> MistPowerAnalyzeResult:
        options = PowerAnalyzeOptions(
            ac_coeff=read_analysis_float(req.parameters, "$qubit.ext.ac_stark_coeff"),
            log_scale=req.analyze_params.log_scale,
            confusion_matrix=read_confusion_matrix(req.parameters),
        )
        PowerExp().analyze(req.run_result, options, plots=plots)
        return MistPowerAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_mist_power_{time.strftime('%m%d')}"
