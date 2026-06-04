from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Annotated, Any, ClassVar, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.ro_optimize.power import (
    PowerCfg,
    PowerExp,
    PowerResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_qub_probe_default,
    make_readout_default,
    make_reset_module_spec,
    make_reset_ref_default,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WritebackItem,
    WritebackRequest,
)

RoOptPowerRunResult: TypeAlias = PowerResult


@dataclass
class RoOptPowerAnalyzeParams:
    penalty_ratio: Annotated[float, ParamMeta(label="Power penalty ratio", decimals=2)]


@dataclass
class RoOptPowerAnalyzeResult(AnalyzeResultBase):
    best_gain: float
    figure: Figure


class RoOptPowerAdapter(
    BaseAdapter[
        PowerCfg,
        RoOptPowerRunResult,
        RoOptPowerAnalyzeResult,
        RoOptPowerAnalyzeParams,
    ]
):
    exp_cls = PowerExp
    ExpCfg_cls: ClassVar[Any] = PowerCfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "Readout power optimization: with the qubit toggled between g and "
                "e by a pi pulse, sweeps the readout gain and measures the g/e "
                "signal-to-noise ratio (SNR), to pick the readout power that best "
                "distinguishes the states. Runs on real hardware. A readout-tuning "
                "step, typically after the readout frequency is set."
            ),
            expects_md=(
                "Reads from the MetaDict (all optional): 'r_f' / 'best_ro_freq' — "
                "resonator / chosen readout frequency for the probe (~4000–8000 "
                "MHz); 'res_ch' / 'ro_ch' — drive / ADC channels; 'timeFly' — "
                "trigger-offset cable delay; 'q_f' / 'qub_ch' — qubit frequency / "
                "channel for the g↔e pi pulse."
            ),
            expects_ml=(
                "Needs a qubit-probe pulse module (typically a calibrated pi "
                "pulse, e.g. 'pi_amp') and a pulse-readout module (e.g. "
                "'readout_rf', usually pinned to the chosen readout frequency); "
                "references a ModuleLibrary waveform 'ro_waveform' when present. "
                "Optionally references a reset module."
            ),
            typical_writeback=(
                "Proposes the SNR-maximizing readout gain into MetaDict "
                "'best_ro_gain' (a.u.). No ModuleLibrary writeback — combine the "
                "best readout params into a 'readout_dpm' module afterwards (the "
                "'readout_dpm' role)."
            ),
            recommended=(
                "Analysis applies a 'power penalty ratio' that down-weights high "
                "gains (SNR × exp(-gain × ratio)), biasing the choice toward lower "
                "power to limit measurement-induced effects; ~0.5 is a sensible "
                "default — raise it to push harder toward low power, set 0 to take "
                "the raw SNR maximum. Sweep gain from near zero up to where the "
                "SNR saturates (e.g. ~0.001–0.2)."
            ),
        )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    label="Modules",
                    fields={
                        "reset": make_reset_module_spec(optional=True),
                        "qub_pulse": make_pulse_module_spec(),
                        "readout": make_pulse_readout_module_spec(),
                    },
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "relax_delay": ScalarSpec(
                    label="Relax delay (us)", type=float, decimals=3
                ),
                "sweep": CfgSectionSpec(
                    label="Sweep",
                    fields={"gain": SweepSpec(label="Readout gain (a.u.)")},
                ),
            }
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return CfgSectionValue(
            fields={
                "modules": CfgSectionValue(
                    fields={
                        "qub_pulse": make_qub_probe_default(ctx),
                        "readout": make_readout_default(ctx),
                        "reset": make_reset_ref_default(ctx, optional=True),
                    }
                ),
                "reps": DirectValue(1000),
                "rounds": DirectValue(100),
                "relax_delay": DirectValue(1.0),
                "sweep": CfgSectionValue(
                    fields={"gain": SweepValue(start=0.001, stop=0.2, expts=101)},
                ),
            }
        )

    def get_analyze_params(
        self, result: RoOptPowerRunResult, ctx: ExpContext
    ) -> RoOptPowerAnalyzeParams:
        return RoOptPowerAnalyzeParams(penalty_ratio=0.5)

    def analyze(
        self, req: AnalyzeRequest[RoOptPowerRunResult, RoOptPowerAnalyzeParams]
    ) -> RoOptPowerAnalyzeResult:
        params = req.analyze_params
        best_gain, fig = PowerExp().analyze(
            req.run_result, penalty_ratio=params.penalty_ratio
        )
        return RoOptPowerAnalyzeResult(best_gain=best_gain, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[RoOptPowerRunResult, RoOptPowerAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            MetaDictWriteback(
                target_name="best_ro_gain",
                description="Optimal readout gain (a.u.)",
                proposed_value=result.best_gain,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_ro_opt_gain_{time.strftime('%m%d')}"
