from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    MetaDictWriteback,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import (
    EvalValue,
    ReferenceValue,
    SweepValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.resources.entry import RoleSpec

from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_lab.v2._support.measure.seeds import (
    custom,
    module_seed,
    path_seed,
    scaled_path_seed,
)
from zcu_lab.v2._support.measure.spec_helpers import (
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_reset_module_spec,
)
from zcu_lab.v2._support.measure.writeback_helpers import readout_dpm_writeback_items
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import (
    FreqGainAnalyzeOptions,
    FreqGainCfg,
    FreqGainExp,
    FreqGainResult,
)

RoOptFreqGainRunResult: TypeAlias = RunRecord[FreqGainCfg, FreqGainResult]


def _best_ro_freq_range(ctx: SessionEnv) -> SweepValue:
    """Center this experiment's frequency scan on its latest trusted optimum."""
    parameters = ctx.parameters
    if parameters is None:
        return SweepValue(start=6250.0, stop=6750.0, expts=31)
    center_path = "qubit.ext.best_ro_freq"
    center_value = parameters.lookup(center_path).value
    if center_value is None:
        center_path = "resonator.freq"
        center_value = parameters.lookup(center_path).value
    width_value = parameters.lookup("resonator.kappa").value
    center = float(center_value) if isinstance(center_value, (int, float)) else 6500.0
    width = float(width_value) if isinstance(width_value, (int, float)) else 500.0
    if center_value is not None and width_value is not None:
        start: float | EvalValue = EvalValue(
            expr=f"{center_path} - 0.5 * resonator.kappa"
        )
        stop: float | EvalValue = EvalValue(
            expr=f"{center_path} + 0.5 * resonator.kappa"
        )
    else:
        half_span = 0.5 * width if width > 0.0 else 30.0
        start = center - half_span
        stop = center + half_span
    return SweepValue(start=start, stop=stop, expts=31)


def _blank_pi_pulse(ctx: SessionEnv) -> ReferenceValue:
    value = make_default_value(make_pulse_module_spec().allowed[0])
    value.with_field("freq", path_seed("qubit.freq", fallback=4000.0).resolve(ctx))
    value.with_field("ch", path_seed("qubit.wiring.L01.ch", fallback=0).resolve(ctx))
    value.with_field("nqz", 2)
    value.with_field("gain", 0.1)
    value.with_field("waveform.length", 5.1)
    return ReferenceValue(make_custom_reference_key("Pulse"), value)


def _blank_readout(ctx: SessionEnv) -> ReferenceValue:
    value = make_default_value(make_pulse_readout_module_spec().allowed[0])
    frequency = path_seed("resonator.freq", fallback=6000.0).resolve(ctx)
    value.with_field("pulse_cfg.freq", frequency)
    value.with_field(
        "pulse_cfg.ch", path_seed("resonator.wiring.ch", fallback=0).resolve(ctx)
    )
    value.with_field("pulse_cfg.nqz", 2)
    value.with_field("pulse_cfg.gain", 0.1)
    value.with_field("pulse_cfg.waveform.length", 1.0)
    value.with_field("ro_cfg.ro_freq", frequency)
    value.with_field(
        "ro_cfg.ro_ch", path_seed("resonator.wiring.ro_ch", fallback=0).resolve(ctx)
    )
    value.with_field("ro_cfg.ro_length", 0.9)
    value.with_field("ro_cfg.trig_offset", 0.55)
    return ReferenceValue(make_custom_reference_key("Pulse Readout"), value)


@dataclass
class RoOptFreqGainAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0


@dataclass
class RoOptFreqGainAnalyzeResult(AnalyzeResultBase):
    best_freq: float
    best_gain: float


class RoOptFreqGainAdapter(
    BaseAdapter[
        FreqGainCfg,
        RoOptFreqGainRunResult,
        RoOptFreqGainAnalyzeResult,
        RoOptFreqGainAnalyzeParams,
    ]
):
    exp_cls = FreqGainExp
    ExpCfg_cls: ClassVar[Any] = FreqGainCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Readout frequency–power joint optimization: with the qubit "
            "toggled between g and e by a pi pulse, runs a 2D sweep of readout "
            "frequency × readout gain and measures the g/e signal-to-noise "
            "ratio (SNR), picking the (freq, gain) pair that best resolves the "
            "states. Runs on real hardware. Use this to refine freq and power "
            "together once you have a rough readout frequency."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 'r_f' / 'best_ro_freq' — "
            "resonator / chosen readout frequency centring the freq sweep "
            "(~4000–8000 MHz); 'rf_w' — linewidth, setting the freq half-span "
            "(~5–50 MHz); 'res_ch' / 'ro_ch' — drive / ADC channels; 'timeFly' "
            "— trigger-offset cable delay; 'q_f' / 'qub_ch' — qubit frequency "
            "/ channel for the g↔e pi pulse."
        ),
        expects_ml=(
            "Needs a qubit-probe pulse module (typically a calibrated pi "
            "pulse, e.g. 'pi_amp') and a pulse-readout module (e.g. "
            "'readout_rf'); references a ModuleLibrary waveform 'ro_waveform' "
            "when present. Optionally references a reset module."
        ),
        typical_writeback=(
            "Proposes the SNR-maximizing readout frequency and gain into "
            "MetaDict 'best_ro_freq' (MHz) and 'best_ro_gain' (a.u.). When a "
            "cfg snapshot with pulse readout is available and "
            "'best_ro_freq' / 'best_ro_gain' / 'best_ro_length' are known "
            "from this result plus MetaDict, also proposes ModuleLibrary "
            "'readout_dpm'."
        ),
        recommended=(
            "Analysis denoises the 2D SNR map before picking the peak. Wavelet "
            "smoothing is the default; switch to Gaussian only when comparing "
            "against the older sigma-based result. Keep the freq span tight and "
            "the gain range modest so the 2D scan stays affordable."
        ),
    )

    def parameter_roles(self) -> Mapping[str, RoleSpec]:
        """Capture the qubit and its referenced resonator when opening a tab."""
        return {
            "qubit": RoleSpec("qubit/*"),
            "resonator": RoleSpec("resonator", via="qubit.resonator"),
        }

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .field(
                "modules.reset",
                spec=make_reset_module_spec(optional=True),
                default=module_seed("qubit.module.reset", fallback=None),
            )
            .field(
                "modules.qub_pulse",
                spec=make_pulse_module_spec(),
                default=module_seed(
                    "qubit.module.x180",
                    fallback=custom(_blank_pi_pulse, description="inline pi pulse"),
                ),
            )
            .field(
                "modules.readout",
                spec=make_pulse_readout_module_spec()
                .lock_literal("pulse_cfg.freq", 0.0)
                .lock_literal("ro_cfg.ro_freq", 0.0)
                .lock_literal("pulse_cfg.gain", 0.0),
                default=module_seed(
                    "qubit.module.readout",
                    fallback=custom(_blank_readout, description="inline readout"),
                ),
            )
            .relax_delay(scaled_path_seed("qubit.t1", factor=5.0, fallback_value=100.0))
            .sweep(
                "freq",
                label="Readout freq (MHz)",
                default=custom(
                    _best_ro_freq_range,
                    description="best readout frequency range (31 points)",
                ),
            )
            .sweep(
                "gain",
                label="Readout gain (a.u.)",
                default=SweepValue(start=0.0, stop=0.2, expts=31),
            )
            .float("skew_penalty", label="Skew penalty", default=0.0, decimals=3)
            .reps(100)
            .rounds(1000)
            .build()
        )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> RoOptFreqGainRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return RunRecord(cfg, FreqGainExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[RoOptFreqGainRunResult, RoOptFreqGainAnalyzeParams],
        *,
        plots: Plots,
    ) -> RoOptFreqGainAnalyzeResult:
        params = req.analyze_params
        options = FreqGainAnalyzeOptions(
            smooth=params.smooth,
            smooth_method=params.smooth_method,
        )
        result = FreqGainExp().analyze(req.run_result, options, plots=plots)
        return RoOptFreqGainAnalyzeResult(
            best_freq=result.best_freq,
            best_gain=result.best_gain,
        )

    def get_writeback_items(
        self, req: WritebackRequest[RoOptFreqGainRunResult, RoOptFreqGainAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            MetaDictWriteback(
                target_name="best_ro_freq",
                description="Optimal readout frequency (MHz)",
                proposed_value=result.best_freq,
            ),
            MetaDictWriteback(
                target_name="best_ro_gain",
                description="Optimal readout gain (a.u.)",
                proposed_value=result.best_gain,
            ),
        ]
        items.extend(
            readout_dpm_writeback_items(
                req.ctx,
                req.run_result.cfg,
                proposed={
                    "best_ro_freq": result.best_freq,
                    "best_ro_gain": result.best_gain,
                },
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ro_opt_freqgain_{time.strftime('%m%d')}"
