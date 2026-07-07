from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

from matplotlib.figure import Figure

from zcu_tools.experiment.v2.twotone.time_domain.t2ramsey import (
    T2RamseyCfg,
    T2RamseyExp,
    T2RamseyResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    Init,
    build_exp_spec,
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
    md_eval_scaled_or_value,
    proper_relax,
)
from zcu_tools.experiment.v2_gui.adapters.twotone.time_domain._detune_shared import (
    detune_ratio_of,
    resolve_detune,
    strip_detune_ratio,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    FloatSpec,
    MetaDictWriteback,
    ParamMeta,
    RunRequest,
    SweepSpec,
    SweepValue,
    WritebackItem,
    WritebackRequest,
    require_soc_handles,
)

logger = logging.getLogger(__name__)


T2RamseyRunResult: TypeAlias = T2RamseyResult


@dataclass
class T2RamseyAnalyzeParams:
    fit_fringe: Annotated[bool, ParamMeta(label="Fit fringe")] = True


@dataclass
class T2RamseyAnalyzeResult(AnalyzeResultBase):
    t2r: float
    t2r_err: float
    # ``detune`` is the fitted fringe frequency (MHz); meaningful only when the
    # fringe fit ran (``fit_fringe`` True). Decay-only fits report 0.0.
    detune: float
    fit_fringe: bool
    figure: Figure


class T2RamseyAdapter(
    BaseAdapter[
        T2RamseyCfg,
        T2RamseyRunResult,
        T2RamseyAnalyzeResult,
        T2RamseyAnalyzeParams,
    ]
):
    exp_cls = T2RamseyExp
    ExpCfg_cls: ClassVar[Any] = T2RamseyCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "T2 Ramsey: two pi/2 pulses separated by a swept free-evolution "
            "delay; fits the decaying fringe to extract the Ramsey coherence "
            "time T2* and the qubit-drive detuning. Runs on real hardware. Run "
            "after a pi/2 pulse is calibrated; a small deliberate detuning "
            "produces the fringe whose frequency reads out the true qubit "
            "frequency offset."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional, seeding defaults): 't2r' — "
            "prior T2-Ramsey estimate (us); the delay sweep spans up to "
            "1.5*t2r (fallback ~0.4 us). 't1' seeds relax_delay (5*t1, fallback ~100 "
            "us). The pi/2 pulse pulls 'q_f' (~2000–6000 MHz) and 'qub_ch'. "
            "Readout pulls 'r_f' (~4000–8000 MHz), 'res_ch' / 'ro_ch', and "
            "'timeFly' for the readout trigger offset (~0–1 us)."
        ),
        expects_ml=(
            "Needs a qubit pi/2-pulse module — references a calibrated library "
            "entry ('pi2_amp' / 'pi2_len') when present, else a blank inline "
            "pulse — and a readout module "
            "(calibrated 'readout_dpm' / 'readout_rf' / 'readout' / "
            "'res_readout', else a blank pulse-readout referencing "
            "'ro_waveform' when present). Optional reset references a library "
            "reset ('reset_bath' / 'reset_10' / 'reset_120') when present."
        ),
        typical_writeback=(
            "Proposes the fitted T2-Ramsey time into MetaDict 't2r' (us). When "
            "the fringe fit ran (the default), also proposes the corrected "
            "qubit frequency into MetaDict 'q_f' (MHz), computed as the pi/2 "
            "pulse frequency + the realized applied detune − the fitted fringe "
            "detune (mirrors the notebook). A pure decay fit proposes only "
            "'t2r'. No ModuleLibrary writeback."
        ),
        recommended=(
            "Set the 'Detune ratio (fringes/step)' cfg knob to a small "
            "deliberate offset (default 0.05) — it is the number of fringe "
            "periods per delay-sweep step; the absolute detune (MHz) applied to "
            "the second pi/2 pulse phase is detune_ratio / sweep step, so the "
            "Ramsey signal oscillates at that detuning. The analysis 'Fit "
            "fringe' option (on by default, matching the nonzero default ratio) "
            "fits the fringe frequency to read back the true qubit-frequency "
            "offset and write 'q_f'. Set the ratio to 0 and turn 'Fit fringe' "
            "off for a pure decay fit when driving on resonance. A delay sweep "
            "reaching a few T2* captures enough fringe periods and decay."
        ),
    )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        # detune_ratio is a run-only knob (not part of T2RamseyCfg): it is the
        # number of fringe periods per delay-sweep step, converted to the
        # absolute detune (MHz) kwarg of T2RamseyExp.run as detune_ratio / step.
        # It lives in build_exp_spec's `extra` slot and is stripped before
        # ml.make_cfg in build_exp_cfg (mirrors ro_optimize/auto's num_points).
        return build_exp_spec(
            modules={
                "reset": make_reset_module_spec(optional=True),
                "pi2_pulse": make_pulse_module_spec(),
                "readout": make_readout_module_spec(),
            },
            sweep={"length": SweepSpec(label="Delay (us)")},
            extra={
                "detune_ratio": FloatSpec(
                    label="Detune ratio (fringes/step)", decimals=3
                )
            },
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(
                reps=1000, rounds=100, relax_delay=proper_relax(ctx), detune_ratio=0.05
            )
            # optional → None (disabled) when no library reset (ADR-0010)
            .role("modules.reset", "reset", Init.DISABLED)
            .role("modules.pi2_pulse", "pi2_pulse")
            .role("modules.readout", "readout")
            .set_sweep(
                "sweep.length",
                SweepValue(
                    start=0.0,
                    stop=md_eval_scaled_or_value(ctx, "t2r", factor=1.5, fallback=0.4),
                    expts=101,
                ),
            )
            .build()
        )

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> T2RamseyCfg:
        # Strip the run-only detune_ratio knob before lowering to T2RamseyCfg,
        # which would reject the unknown key.
        return req.ml.make_cfg(strip_detune_ratio(raw_cfg), T2RamseyCfg)

    def run(self, req: RunRequest, schema: CfgSchema) -> T2RamseyRunResult:
        soc, soccfg = require_soc_handles(req)
        raw_cfg = schema.to_raw_dict(req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        # detune_ratio (fringes-per-step) → absolute applied detune (MHz) over the
        # lowered length sweep step (SweepCfg guarantees step != 0 for expts > 1).
        detune = resolve_detune(detune_ratio_of(raw_cfg), cfg.sweep.length.step)
        # T2RamseyExp.run returns (result, true_detune); the GUI run contract
        # returns only the Result. Stash true_detune for the q_f writeback and
        # log the realized detune.
        result = self.exp_cls().run(soc, soccfg, cfg, detune=detune)
        logger.info("T2 Ramsey true detune: %.3f MHz", result.true_activate_detune)
        return result

    def analyze(
        self, req: AnalyzeRequest[T2RamseyRunResult, T2RamseyAnalyzeParams]
    ) -> T2RamseyAnalyzeResult:
        params = req.analyze_params
        t2r, t2r_err, detune, _, fig = T2RamseyExp().analyze(
            req.run_result, fit_fringe=params.fit_fringe
        )
        return T2RamseyAnalyzeResult(
            t2r=t2r,
            t2r_err=t2r_err,
            detune=detune,
            fit_fringe=params.fit_fringe,
            figure=fig,
        )

    def get_writeback_items(
        self, req: WritebackRequest[T2RamseyRunResult, T2RamseyAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            MetaDictWriteback(
                target_name="t2r",
                description="T2 Ramsey time (us)",
                proposed_value=result.t2r,
            ),
        ]

        # q_f writeback mirrors the notebook:
        #   q_f = pi2_pulse.freq + true_detune - fitted_fringe_detune.
        # Only emit it when the fringe fit ran (a fitted detune exists), the run
        # captured the cfg_snapshot (pi2 freq source), and the realized
        # true_detune from this run is known — never propose a NaN q_f.
        snapshot = req.run_result.cfg_snapshot
        if (
            result.fit_fringe
            and snapshot is not None
            and req.run_result.true_activate_detune is not None
        ):
            q_f = (
                snapshot.modules.pi2_pulse.freq
                + req.run_result.true_activate_detune
                - result.detune
            )
            items.append(
                MetaDictWriteback(
                    target_name="q_f",
                    description="Qubit frequency (MHz)",
                    proposed_value=q_f,
                )
            )

        return items

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_t2ramsey_{time.strftime('%m%d')}"
