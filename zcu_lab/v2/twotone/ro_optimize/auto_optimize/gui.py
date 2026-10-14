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
    ParameterWriteback,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.writeback_helpers import readout_dpm_writeback_items
from zcu_lab.v2.twotone.ro_optimize.auto_optimize.core import (
    AutoOptCfg,
    AutoOptExp,
    AutoOptResult,
)
from zcu_lab.v2.twotone.ro_optimize.auto_optimize.form import FORM

RoOptAutoRunResult: TypeAlias = RunRecord[AutoOptCfg, AutoOptResult]


@dataclass
class RoOptAutoAnalyzeParams:
    # analyze() takes no tuning knobs — it reads off the best of the optimized
    # points. No form fields.
    pass


@dataclass
class RoOptAutoAnalyzeResult(AnalyzeResultBase):
    best_freq: float
    best_gain: float
    best_length: float


class RoOptAutoAdapter(
    BaseAdapter[
        AutoOptCfg,
        RoOptAutoRunResult,
        RoOptAutoAnalyzeResult,
        RoOptAutoAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = AutoOptExp.persistence_tag()
    exp_cls = AutoOptExp
    ExpCfg_cls = AutoOptCfg
    form = FORM

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Automatic readout optimization: a Bayesian optimizer (skopt) "
            "searches readout frequency, gain and length jointly to maximize "
            "the g/e signal-to-noise ratio (SNR), with the qubit toggled "
            "between g and e by a pi pulse. Runs on real hardware. Use it to "
            "find a good readout in one shot instead of tuning freq / power / "
            "length one axis at a time."
        ),
        expects_md=(
            "Search bounds seed 51 frequency points at $resonator.freq +/- 0.2 * "
            "$resonator.kappa, 51 gain points from 0.1 to 0.25, and 51 length points from "
            "5.0 to 10.0 us. These are optimizer bounds, not an acquisition grid. Relax "
            "delay reads 3.0 * $qubit.t1; optimizer points start at 1001. Open and Reset "
            "evaluate starting expressions once into direct input using captured sources. "
            "Inline qubit pulses read $qubit.freq and $qubit.wiring.L01.ch. Inline "
            "readout reads $resonator.freq, $resonator.wiring.ch, $resonator.wiring.ro_ch "
            "and $resonator.ext.time_of_fly + 0.05 us for trigger timing. Missing or "
            "invalid required sources leave the affected fields empty and Invalid; supply "
            "calibration or direct values before Run. No numeric calibration fallback is "
            "supplied."
        ),
        expects_ml=(
            "The qubit pulse uses $qubit.module.x180 as a captured library link, "
            "otherwise a fresh pi_pulse template with a const waveform. "
            "$qubit.module.readout selects a captured library link, otherwise a fresh "
            "inline pulse readout template with a const waveform. Direct readout is "
            "not supported: frequency, gain and length optimization requires a pulse. "
            "Its required parameter "
            "sources must still be present. Optional $qubit.module.reset has no fallback: "
            "an unset/dangling slot or unbound alias leaves it disabled and Valid."
        ),
        typical_writeback=(
            "Proposes the optimizer's best readout frequency, gain and length into "
            "parameter context $qubit.ext.best_ro_freq (MHz), $qubit.ext.best_ro_gain "
            "(a.u.) and $qubit.ext.best_ro_length (us). When a cfg snapshot with pulse "
            "readout is available, also proposes $qubit.readout.readout_dpm."
        ),
        recommended=(
            "The three sweep axes define the optimizer's SEARCH BOUNDS (min / "
            "max), not a grid — keep them reasonably tight (e.g. freq within a "
            "fraction of a linewidth, gain and length in a sensible band) so "
            "the search converges. 'Optimizer points' (~1000) sets how many "
            "evaluations to spend; raise it for wider bounds, lower it for a "
            "quick search. No analysis knobs — it reports the best evaluated "
            "point."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> RoOptAutoRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, AutoOptExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[RoOptAutoRunResult, RoOptAutoAnalyzeParams],
        *,
        plots: Plots,
    ) -> RoOptAutoAnalyzeResult:
        result = AutoOptExp().analyze(req.run_result, None, plots=plots)
        return RoOptAutoAnalyzeResult(
            best_freq=result.best_freq,
            best_gain=result.best_gain,
            best_length=result.best_length,
        )

    def get_writeback_items(
        self, req: WritebackRequest[RoOptAutoRunResult, RoOptAutoAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.best_ro_freq",
                description="Optimal readout frequency (MHz)",
                proposed_value=result.best_freq,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.best_ro_gain",
                description="Optimal readout gain (a.u.)",
                proposed_value=result.best_gain,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.best_ro_length",
                description="Optimal readout length (us)",
                proposed_value=result.best_length,
            ),
        ]
        items.extend(
            readout_dpm_writeback_items(
                req.ctx,
                req.run_result.cfg,
                proposed={
                    "best_ro_freq": result.best_freq,
                    "best_ro_gain": result.best_gain,
                    "best_ro_length": result.best_length,
                },
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ro_opt_auto_{time.strftime('%m%d')}"
