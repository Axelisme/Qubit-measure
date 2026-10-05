from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.singleshot.mist import PowerCfg, PowerExp, PowerResult
from zcu_tools.experiment.v2.singleshot.mist.power import PowerAnalyzeOptions
from zcu_tools.experiment.v2_gui.measure.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    ModuleInit,
    custom,
    scaled_md,
)
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import (
    EvalValue,
    ScalarSpec,
    SweepValue,
)
from zcu_tools.plotting.plots import Plots

from .._shared import readout_probe_freq

MistPowerRunResult: TypeAlias = RunRecord[PowerCfg, PowerResult]


@dataclass
class MistPowerAnalyzeResult(AnalyzeResultBase):
    pass


class MistPowerAdapter(
    BaseAdapter[PowerCfg, MistPowerRunResult, MistPowerAnalyzeResult, NoAnalyzeParams]
):
    exp_cls = PowerExp
    ExpCfg_cls: ClassVar[Any] = PowerCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "MIST probe-power sweep: drives a probe pulse whose gain is swept "
            "and classifies each shot in-program against the |g>/|e> "
            "single-shot centres, plotting the ground/excited/other "
            "populations versus probe gain (or photon number when an AC-Stark "
            "coefficient is known). Runs on real hardware; the result is "
            "already populations (no per-point fit)."
        ),
        expects_md=(
            "Run freezes 'g_center' / 'e_center' / 'ge_radius' from resolved "
            "cfg, not live MetaDict. Enter direct cfg values or optionally seed "
            "defaults with 'singleshot/ge' writeback. The run classifies each "
            "shot using these values; missing or invalid cfg calibration fails "
            "before hardware. Optionally reads 'confusion_matrix' (readout correction) "
            "and 'ac_stark_coeff' (rescales the x-axis to photon number) at "
            "analyze time, and 't1' to set the relax delay; 'readout_f' or "
            "'r_f' plus 'res_ch' seed the probe drive."
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
            "Set calibration cfg directly or seed it with 'singleshot/ge'. Sweep "
            "the probe gain across the MIST onset; provide 'ac_stark_coeff' "
            "(from the AC-Stark experiment) for a photon-number x-axis."
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
                "gain",
                label="Probe gain (a.u.)",
                default=SweepValue(start=0.0, stop=1.0, expts=151),
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
            .reps(1000)
            .rounds(100)
            .build()
        )

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def run(
        self,
        req: RunRequest,
        raw_cfg: dict[str, object],
        *,
        context: RunContext,
    ) -> MistPowerRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, PowerExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MistPowerRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> MistPowerAnalyzeResult:
        options = PowerAnalyzeOptions(
            ac_coeff=req.md.get("ac_stark_coeff"),
            log_scale=bool(req.md.get("log_scale", False)),
            confusion_matrix=req.md.get("confusion_matrix"),
        )
        PowerExp().analyze(req.run_result, options, plots=plots)
        return MistPowerAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_mist_power_{time.strftime('%m%d')}"
