from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    MetaDictWriteback,
    ModuleWriteback,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
    require_soc_handles,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.cfg import (
    CfgSchema,
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.resources.entry import RoleSpec

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_lab.v2._support.measure.seeds import (
    SweepDefault,
    custom,
    module_seed,
    path_seed,
    scaled_path_seed,
)
from zcu_lab.v2._support.measure.spec_helpers import (
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
)
from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiAnalyzeOptions,
    LenRabiCfg,
    LenRabiExp,
    LenRabiResult,
)

LenRabiRunResult: TypeAlias = RunRecord[LenRabiCfg, LenRabiResult]


def _blank_rabi_pulse(ctx: SessionEnv) -> ReferenceValue:
    value = make_default_value(make_pulse_module_spec().allowed[0])
    value.with_field("freq", path_seed("qubit.freq", fallback=4000.0).resolve(ctx))
    value.with_field("ch", path_seed("qubit.wiring.L01.ch", fallback=0).resolve(ctx))
    value.with_field("nqz", 2)
    value.with_field("gain", 0.3)
    value.with_field("waveform.length", 1.0)
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
class LenRabiAnalyzeParams:
    decay: Annotated[bool, ParamMeta(label="Fit decay envelope")] = True
    fit_phase: Annotated[bool, ParamMeta(label="Fit phase offset")] = False


@dataclass
class LenRabiAnalyzeResult(AnalyzeResultBase):
    pi_len: float
    pi_len_err: float
    pi2_len: float
    pi2_len_err: float
    # Rabi oscillation frequency in MHz (1/us), proposed at qubit.ext.rabi_freq.
    rabi_f: float
    rabi_f_err: float
    fit_quality: dict[str, dict[str, object]] | None = None


class LenRabiAdapter(
    BaseAdapter[
        LenRabiCfg,
        LenRabiRunResult,
        LenRabiAnalyzeResult,
        LenRabiAnalyzeParams,
    ]
):
    exp_cls = LenRabiExp
    ExpCfg_cls: ClassVar[Any] = LenRabiCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Length Rabi: drives the qubit at its known frequency and sweeps "
            "the drive-pulse length, fitting the Rabi oscillation to find the "
            "pi and pi/2 pulse lengths. Runs on real hardware. Run once you "
            "know the qubit frequency, to calibrate a pi pulse by duration "
            "(the pulse gain stays fixed)."
        ),
        expects_md=(
            "Uses the tab's captured qubit and resonator roles. Length sweep "
            "stop is 4 * qubit.module.x180.waveform.length (fallback 0.3 us); "
            "relax delay is 5 * qubit.t1 (fallback 30.5 us). Fresh drive uses "
            "qubit.freq (fallback 4000 MHz) and wiring.L01.ch (fallback 0). "
            "Fresh readout uses resonator.freq (fallback 6000 MHz) and "
            "wiring.ch/wiring.ro_ch (fallback 0), with trigger offset 0.55 us."
        ),
        expects_ml=(
            "Mounts captured qubit.module.x180/readout/reset slots. Missing "
            "or dangling drive/readout slots use fresh inline pulses; reset "
            "stays disabled. The length sweep owns waveform.length, locked "
            "at 1 us in the cfg. Inline drive gain is 0.3; inline readout "
            "gain is 0.1, const length 1 us and ADC window 0.9 us. No named "
            "waveform is adopted."
        ),
        typical_writeback=(
            "Proposes qubit.ext.rabi_freq in MHz from the fit. With a source "
            "cfg, copies the drive pulse into captured qubit.pulses.pi_amp "
            "and pi2_amp, replacing waveform.length with fitted pi and pi/2 "
            "lengths. Separately proposes qubit.module.x180/x90 pointing to "
            "those resolved keys, even when already bound to them, plus "
            "qubit.ext.pi_product = source gain * fitted pi length (a.u. * us). "
            "Every item is independently selectable. Without cfg only "
            "rabi_freq remains. No separate length scalars or live defaults."
        ),
        recommended=(
            "Phase defaults to fixed 0/180 degrees; enable Fit phase offset "
            "when pulse shaping shifts the length oscillation. "
            "Analysis defaults to fitting a decay envelope on the oscillation; "
            "keep it on when the Rabi oscillation visibly damps over the "
            "sweep, turn it off for a pure undamped cosine fit. A length sweep "
            "spanning a few pi lengths (up to 4*pi_len when calibrated) captures a full "
            "oscillation; widen it if no full period is visible."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> LenRabiRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = LenRabiExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

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
                spec=make_pulse_module_spec(label="Rabi Pulse").lock_literal(
                    "waveform.length", 1.0
                ),
                default=module_seed(
                    "qubit.module.x180",
                    fallback=custom(_blank_rabi_pulse, description="inline rabi pulse"),
                ),
            )
            .field(
                "modules.readout",
                spec=make_readout_module_spec(),
                default=module_seed(
                    "qubit.module.readout",
                    fallback=custom(_blank_readout, description="inline readout"),
                ),
            )
            .relax_delay(
                scaled_path_seed("qubit.t1", factor=5.0, fallback_value=30.5),
            )
            .sweep(
                "length",
                label="Length (us)",
                default=SweepDefault(
                    start=0.03,
                    stop=scaled_path_seed(
                        "qubit.module.x180.waveform.length",
                        factor=4.0,
                        fallback_value=0.3,
                    ),
                    expts=101,
                ),
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    def validate_run_request(
        self, req: RunRequest, raw_cfg: Mapping[str, ExperimentCfgValue]
    ) -> None:
        cfg = self.build_exp_cfg(raw_cfg, req)
        _, soccfg = require_soc_handles(req)
        sweep2array(
            cfg.sweep.length,
            "time",
            {"soccfg": soccfg, "gen_ch": cfg.modules.qub_pulse.ch},
        )

    def analyze(
        self,
        req: AnalyzeRequest[LenRabiRunResult, LenRabiAnalyzeParams],
        *,
        plots: Plots,
    ) -> LenRabiAnalyzeResult:
        params = req.analyze_params
        analysis = LenRabiExp().analyze(
            req.run_result,
            LenRabiAnalyzeOptions(decay=params.decay, fit_phase=params.fit_phase),
            plots=plots,
        )
        return LenRabiAnalyzeResult(
            pi_len=analysis.pi_len,
            pi_len_err=analysis.pi_len_err,
            pi2_len=analysis.pi2_len,
            pi2_len_err=analysis.pi2_len_err,
            rabi_f=analysis.rabi_f,
            rabi_f_err=analysis.rabi_f_err,
            fit_quality=fit_quality_summary(analysis.fit_quality),
        )

    def get_writeback_items(
        self, req: WritebackRequest[LenRabiRunResult, LenRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose independent calibrations from the source and captured roles.

        req pairs the source RunRecord, fitted lengths/frequency and captured
        ctx.parameters. Always propose qubit.ext.rabi_freq in MHz. With cfg,
        add qubit.pulses.pi_amp/pi2_amp pulse proposals with fitted lengths,
        independent x180/x90 slot proposals containing their resolved keys,
        and qubit.ext.pi_product = source gain * pi_len (a.u. * us).
        Same-key slots are still proposed. Without cfg return only rabi_freq.
        Only waveform lengths change; source cfg is not mutated. Missing
        capture, role resolution, non-scalar source gain and unsupported
        waveform-length conversion fail explicitly. No hardware, live
        calibration reads or writes occur.
        """
        parameters = req.ctx.parameters
        if parameters is None:
            raise ValueError("Length Rabi writeback requires captured parameters")
        result = req.analyze_result
        items: list[WritebackItem] = [
            MetaDictWriteback(
                target_name="qubit.ext.rabi_freq",
                description="Rabi oscillation frequency (MHz)",
                proposed_value=result.rabi_f,
            ),
        ]
        snapshot = req.run_result.cfg
        if snapshot is None:
            return items
        pulse = snapshot.modules.qub_pulse
        gain = pulse.gain
        if not isinstance(gain, (int, float)):
            raise ValueError("Length Rabi pi_product requires a scalar source gain")
        for target, slot, length, description in (
            ("qubit.pulses.pi_amp", "qubit.module.x180", result.pi_len, "Pi pulse"),
            ("qubit.pulses.pi2_amp", "qubit.module.x90", result.pi2_len, "Pi/2 pulse"),
        ):
            key = parameters.library_key(target)
            spec, value = module_cfg_to_value(pulse)
            value.with_field("waveform.length", length)
            items.append(
                ModuleWriteback(
                    target_name=target,
                    description=description,
                    edit_schema=CfgSchema(spec=spec, value=value),
                )
            )
            items.append(
                MetaDictWriteback(
                    target_name=slot,
                    description=f"{description} slot",
                    proposed_value=key,
                )
            )
        items.append(
            MetaDictWriteback(
                target_name="qubit.ext.pi_product",
                description="Pi pulse gain times length (a.u. * us)",
                proposed_value=gain * result.pi_len,
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_len_rabi_{time.strftime('%m%d')}"
