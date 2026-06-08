from __future__ import annotations

import time
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Any, ClassVar, Sequence, TypeAlias

from zcu_tools.experiment.v2.twotone.ro_optimize.length import (
    LengthCfg,
    LengthExp,
    LengthResult,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_reset_module_spec,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    MetaDictWriteback,
    ScalarSpec,
    SweepSpec,
    WritebackItem,
    WritebackRequest,
)

RoOptLengthRunResult: TypeAlias = LengthResult


@dataclass
class RoOptLengthAnalyzeParams:
    # The underlying analyze() takes an optional 't0' length-penalty knob
    # (t0=None → raw SNR max; t0>0 → max snr/sqrt(length+t0), favouring shorter
    # readout). Optional analyze fields are not yet supported by the form
    # framework, so this adapter pins t0=None for now and exposes no field.
    # See memory: project_gui_optional_analyze_param.
    pass


@dataclass
class RoOptLengthAnalyzeResult(AnalyzeResultBase):
    best_length: float
    figure: Figure


class RoOptLengthAdapter(
    BaseAdapter[
        LengthCfg,
        RoOptLengthRunResult,
        RoOptLengthAnalyzeResult,
        RoOptLengthAnalyzeParams,
    ]
):
    exp_cls = LengthExp
    ExpCfg_cls: ClassVar[Any] = LengthCfg

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior=(
                "Readout length optimization: with the qubit toggled between g and "
                "e by a pi pulse, sweeps the readout window length and measures the "
                "g/e signal-to-noise ratio (SNR), to pick the shortest readout that "
                "still resolves the states well. Runs on real hardware. A "
                "readout-tuning step, typically after the readout frequency and "
                "power are set."
            ),
            expects_md=(
                "Reads from the MetaDict (all optional): 'r_f' / 'best_ro_freq' — "
                "resonator / chosen readout frequency (~4000–8000 MHz); 'res_ch' / "
                "'ro_ch' — drive / ADC channels; 'timeFly' — trigger-offset cable "
                "delay; 'q_f' / 'qub_ch' — qubit frequency / channel for the g↔e "
                "pi pulse."
            ),
            expects_ml=(
                "Needs a qubit-probe pulse module (typically a calibrated pi "
                "pulse, e.g. 'pi_amp') and a pulse-readout module (e.g. "
                "'readout_rf', usually pinned to the chosen readout frequency and "
                "gain); references a ModuleLibrary waveform 'ro_waveform' when "
                "present. Optionally references a reset module."
            ),
            typical_writeback=(
                "Proposes the SNR-maximizing readout length into MetaDict "
                "'best_ro_length' (us). No ModuleLibrary writeback — combine the "
                "best readout params into a 'readout_dpm' module afterwards (the "
                "'readout_dpm' role)."
            ),
            recommended=(
                "Analysis currently picks the raw SNR maximum (the underlying "
                "length-penalty knob that would bias toward shorter windows is not "
                "yet exposed in the GUI). Sweep length from short to a few us "
                "(e.g. ~0.01–3.5 us); since SNR usually keeps rising with length, "
                "judge by eye where it plateaus and trade length for SNR."
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
                        # The sweep axis owns the readout acquisition window
                        # (set_param("ro_length") at run), so lock it off the
                        # form. The pulse waveform length is auto-derived from the
                        # sweep max but stays editable (it sets the gaussian
                        # ratio for shaped pulses — see Phase 140 C-table).
                        "readout": make_pulse_readout_module_spec().lock_literal(
                            "ro_cfg.ro_length", 0.0
                        ),
                    },
                ),
                "reps": ScalarSpec(label="Reps", type=int),
                "rounds": ScalarSpec(label="Rounds", type=int),
                "relax_delay": ScalarSpec(
                    label="Relax delay (us)", type=float, decimals=3
                ),
                "sweep": CfgSectionSpec(
                    label="Sweep",
                    fields={"length": SweepSpec(label="Readout length (us)")},
                ),
            }
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(reps=10000, rounds=1, relax_delay=1.0)
            .role("modules.qub_pulse", "qub_probe", prefer_blank=True)
            .role("modules.readout", "readout", prefer_blank=True)
            .role("modules.reset", "reset", optional=True)
            .sweep("sweep.length", 0.01, 3.5, 51)
            .build()
        )

    def get_analyze_params(
        self, result: RoOptLengthRunResult, ctx: ExpContext
    ) -> RoOptLengthAnalyzeParams:
        return RoOptLengthAnalyzeParams()

    def analyze(
        self, req: AnalyzeRequest[RoOptLengthRunResult, RoOptLengthAnalyzeParams]
    ) -> RoOptLengthAnalyzeResult:
        # t0 pinned to None (raw SNR max) until optional analyze fields land.
        best_length, fig = LengthExp().analyze(req.run_result, t0=None)
        return RoOptLengthAnalyzeResult(best_length=best_length, figure=fig)

    def get_writeback_items(
        self, req: WritebackRequest[RoOptLengthRunResult, RoOptLengthAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            MetaDictWriteback(
                target_name="best_ro_length",
                description="Optimal readout length (us)",
                proposed_value=result.best_length,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_ro_opt_length_{time.strftime('%m%d')}"
