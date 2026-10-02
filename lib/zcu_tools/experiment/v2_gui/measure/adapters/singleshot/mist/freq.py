from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.singleshot.mist import FreqCfg, FreqDepExp, FreqResult
from zcu_tools.experiment.v2.singleshot.mist.freq import FreqAnalyzeOptions
from zcu_tools.experiment.v2_gui.measure.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    custom,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.measure.adapters.base import BaseAdapter
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.cfg import (
    EvalValue,
    ScalarSpec,
)
from zcu_tools.plotting.plots import Plots

from .._shared import readout_probe_freq, readout_probe_freq_range

MistFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class MistFreqAnalyzeResult(AnalyzeResultBase):
    pass


class MistFreqAdapter(
    BaseAdapter[FreqCfg, MistFreqRunResult, MistFreqAnalyzeResult, NoAnalyzeParams]
):
    exp_cls = FreqDepExp
    ExpCfg_cls: ClassVar[Any] = FreqCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "MIST probe-frequency sweep: drives a probe pulse whose frequency "
            "is swept and classifies each shot in-program against the |g>/|e> "
            "single-shot centres, plotting the ground/excited/other populations "
            "versus probe frequency. Runs on real hardware; the result is "
            "already populations (no per-point fit)."
        ),
        expects_md=(
            "Run freezes 'g_center' / 'e_center' / 'ge_radius' from resolved "
            "cfg, not live MetaDict. Enter direct cfg values or optionally seed "
            "defaults with 'singleshot/ge' writeback. The run classifies each "
            "shot using these values; missing or invalid cfg calibration fails "
            "before hardware. Optionally reads 'confusion_matrix' (the GE 3x3 matrix) "
            "to readout-correct the populations at analyze time, and 't1' to "
            "set the relax delay; 'readout_f' or 'r_f' plus 'rf_w' / 'res_ch' "
            "seed the probe drive and frequency sweep."
        ),
        expects_ml=(
            "Needs a probe pulse and a readout module. Optionally references a "
            "calibrated reset and an init pulse — both disabled when no library "
            "entry exists."
        ),
        typical_writeback=(
            "No writeback — the population curves are read off the plot by eye."
        ),
        recommended=(
            "Set calibration cfg directly or seed it with 'singleshot/ge'. A "
            "frequency span around the qubit drive captures the MIST response; "
            "use a large enough shot count (reps) for clean populations."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("init_pulse", role_id="pi_pulse", optional=True)
            .pulse(
                "probe_pulse",
                role_id="res_probe",
                label="Probe Pulse",
                init=ModuleInit.INLINE,
                overrides={
                    "freq": custom(
                        readout_probe_freq,
                        description="readout probe frequency",
                    )
                },
            )
            .readout()
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=30.5))
            .sweep(
                "freq",
                label="Probe freq (MHz)",
                default=custom(
                    lambda ctx: readout_probe_freq_range(ctx, 51),
                    description="readout probe frequency range",
                ),
            )
            .field(
                "g_center",
                spec=ScalarSpec("Ground center", complex),
                default=EvalValue("g_center"),
            )
            .field(
                "e_center",
                spec=ScalarSpec("Excited center", complex),
                default=EvalValue("e_center"),
            )
            .field(
                "radius",
                spec=ScalarSpec("Classification radius", float),
                default=EvalValue("ge_radius"),
            )
            .reps(10000)
            .rounds(1)
            .build()
        )

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def run(
        self,
        req: RunRequest,
        raw_cfg: dict[str, object],
        *,
        context: RunContext,
    ) -> MistFreqRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, FreqDepExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MistFreqRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> MistFreqAnalyzeResult:
        options = FreqAnalyzeOptions(
            confusion_matrix=req.md.get("confusion_matrix"),
        )
        FreqDepExp().analyze(req.run_result, options, plots=plots)
        return MistFreqAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_mist_freq_{time.strftime('%m%d')}"
