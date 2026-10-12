from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    AnalyzeResultBase,
    ModuleWriteback,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.app.measure.seeds import (
    SweepDefault,
    custom,
    module_seed,
    path_seed,
    scaled_path_seed,
)
from zcu_tools.gui.app.measure.spec_helpers import (
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
)
from zcu_tools.gui.cfg import (
    CfgSchema,
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2.twotone.rabi.amp_rabi.core import (
    AmpRabiAnalyzeOptions,
    AmpRabiCfg,
    AmpRabiExp,
    AmpRabiResult,
)

AmpRabiRunResult: TypeAlias = RunRecord[AmpRabiCfg, AmpRabiResult]


@dataclass
class AmpRabiAnalyzeParams:
    skip: Annotated[int, ParamMeta(label="Skip points")] = 0


@dataclass
class AmpRabiAnalyzeResult(AnalyzeResultBase):
    # Fitted gains stay in this analysis result and the proposed pulse modules.
    pi_gain: float
    pi_gain_err: float
    pi2_gain: float
    pi2_gain_err: float
    fit_quality: dict[str, dict[str, object]] | None = None


def _blank_rabi_pulse(ctx: SessionEnv) -> ReferenceValue:
    """Seed a fresh drive pulse from captured qubit paths.

    Use qubit.freq (fallback 4000 MHz), wiring.L01.ch (fallback 0), nqz 2,
    gain 0 and a const waveform whose length is 1.01 times the captured
    module.x180.waveform.length (fallback 1.1 us). Missing/null calibration
    uses the fallback; role, capture and materialization faults propagate.
    """
    spec = make_pulse_module_spec().allowed[0]
    value = make_default_value(spec)
    return (
        ReferenceValue(make_custom_reference_key(spec.label), value)
        .with_field("freq", path_seed("$qubit.freq", fallback=4000.0).resolve(ctx))
        .with_field("ch", path_seed("$qubit.wiring.L01.ch", fallback=0).resolve(ctx))
        .with_field("nqz", 2)
        .with_field("gain", 0.0)
        .with_field(
            "waveform.length",
            scaled_path_seed(
                "$qubit.module.x180.waveform.length", factor=1.01, fallback_value=1.1
            ).resolve(ctx),
        )
    )


def _blank_readout(ctx: SessionEnv) -> ReferenceValue:
    """Seed a fresh pulse readout from captured resonator paths.

    Use resonator.freq for generator and ADC (fallback 6000 MHz), wiring.ch
    and wiring.ro_ch (fallback 0), nqz 2, gain 0.1, const pulse length 1 us
    and ADC window 0.9 us. Trigger offset is ext.time_of_fly plus 0.05 us
    (fallback 0.55 us). Missing/null values use these fallbacks; role,
    capture and materialization faults propagate. Do not use named waveforms.
    """
    spec = make_pulse_readout_module_spec().allowed[0]
    value = make_default_value(spec)
    frequency = path_seed("$resonator.freq", fallback=6000.0).resolve(ctx)
    return (
        ReferenceValue(make_custom_reference_key(spec.label), value)
        .with_field("pulse_cfg.freq", frequency)
        .with_field(
            "pulse_cfg.ch", path_seed("$resonator.wiring.ch", fallback=0).resolve(ctx)
        )
        .with_field("pulse_cfg.nqz", 2)
        .with_field("pulse_cfg.gain", 0.1)
        .with_field("pulse_cfg.waveform.length", 1.0)
        .with_field("ro_cfg.ro_freq", frequency)
        .with_field(
            "ro_cfg.ro_ch",
            path_seed("$resonator.wiring.ro_ch", fallback=0).resolve(ctx),
        )
        .with_field("ro_cfg.ro_length", 0.9)
        .with_field(
            "ro_cfg.trig_offset",
            path_seed(
                "$resonator.ext.time_of_fly",
                fallback=0.55,
                expr="$resonator.ext.time_of_fly + 0.05",
            ).resolve(ctx),
        )
    )


class AmpRabiAdapter(
    BaseAdapter[
        AmpRabiCfg,
        AmpRabiRunResult,
        AmpRabiAnalyzeResult,
        AmpRabiAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = AmpRabiExp.persistence_tag()
    exp_cls = AmpRabiExp
    ExpCfg_cls = AmpRabiCfg

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Amplitude Rabi: drives the qubit at its known frequency and "
            "sweeps the drive-pulse gain, fitting the resulting Rabi "
            "oscillation to find the pi and pi/2 pulse amplitudes. Runs on "
            "real hardware. Run once you know the qubit frequency, to "
            "calibrate a pi pulse by amplitude (the pulse length stays fixed)."
        ),
        expects_md=(
            "Uses the tab's captured qubit and resonator roles. Gain sweep stop "
            "is 1.2 * qubit.module.x180.gain (fallback 0.6); relax delay is "
            "5 * qubit.t1 (fallback 30.5 us). Fresh drive defaults read "
            "qubit.freq and qubit.wiring.L01.ch, with length "
            "1.01 * qubit.module.x180.waveform.length (fallback 1.1 us). "
            "Fresh readout reads resonator.freq, wiring.ch, wiring.ro_ch "
            "and ext.time_of_fly + 0.05 us (fallback 0.55 us)."
        ),
        expects_ml=(
            "Mounts captured qubit.module.x180, qubit.module.readout and "
            "qubit.module.reset slots. Missing/null or dangling drive and "
            "readout slots use fresh inline pulses; reset stays disabled. "
            "The gain sweep owns drive gain, locked at zero in the cfg. "
            "Inline readout uses a const waveform, not a named waveform."
        ),
        typical_writeback=(
            "Copies the source drive pulse into captured qubit.pulses.pi_amp "
            "and pi2_amp, replacing gain with the fitted pi and pi/2 gains. "
            "Separately proposes qubit.module.x180 and x90 pointing to those "
            "resolved library keys, even when already bound to the same keys. "
            "Also proposes qubit.ext.pi_product as fitted pi gain times the "
            "source pulse length (a.u. * us). Every item is independently "
            "selectable; none is proposed without a source cfg. Does not "
            "write separate gain scalars or read current calibration."
        ),
        recommended=(
            "Analysis takes a 'Skip points' count (default 0) to drop leading "
            "low-gain sweep points before the cosine fit; raise it only if the "
            "first few points are distorted. A gain sweep spanning roughly two "
            "pi amplitudes captures at least a full oscillation; widen it if "
            "no full period is visible."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> AmpRabiRunResult:
        """Build typed cfg from resolved GUI values and run amplitude Rabi.

        req provides the captured cfg context and device snapshot; context owns
        acquisition and cancellation. Return the typed cfg alongside the result.
        Config, device and acquisition failures propagate without live defaults.
        """
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = AmpRabiExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        """Return the role-based amplitude-Rabi template, without live reads.

        Instantiation mounts captured reset/x180/readout slots and resolves
        gain/length/t1 seeds on that capture. Missing/null calibration uses
        documented inline fallbacks; source and role errors propagate.
        """
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .field(
                "modules.reset",
                spec=make_reset_module_spec(optional=True),
                default=module_seed("$qubit.module.reset", fallback=None),
            )
            .field(
                "modules.qub_pulse",
                spec=make_pulse_module_spec(label="Rabi Pulse").lock_literal(
                    "gain", 0.0
                ),
                default=module_seed(
                    "$qubit.module.x180",
                    fallback=custom(_blank_rabi_pulse, description="inline Rabi pulse"),
                ),
            )
            .field(
                "modules.readout",
                spec=make_readout_module_spec(),
                default=module_seed(
                    "$qubit.module.readout",
                    fallback=custom(_blank_readout, description="inline readout"),
                ),
            )
            .relax_delay(
                scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=30.5),
            )
            .sweep(
                "gain",
                label="Gain (a.u.)",
                default=SweepDefault(
                    start=-0.3,
                    stop=scaled_path_seed(
                        "$qubit.module.x180.gain", factor=1.2, fallback_value=0.6
                    ),
                    expts=51,
                ),
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    def analyze(
        self,
        req: AnalyzeRequest[AmpRabiRunResult, AmpRabiAnalyzeParams],
        *,
        plots: Plots,
    ) -> AmpRabiAnalyzeResult:
        params = req.analyze_params
        analysis = AmpRabiExp().analyze(
            req.run_result,
            AmpRabiAnalyzeOptions(skip=params.skip),
            plots=plots,
        )
        return AmpRabiAnalyzeResult(
            pi_gain=analysis.pi_amp,
            pi_gain_err=analysis.pi_amp_err,
            pi2_gain=analysis.pi2_amp,
            pi2_gain_err=analysis.pi2_amp_err,
            fit_quality=fit_quality_summary(analysis.fit_quality),
        )

    def get_writeback_items(
        self, req: WritebackRequest[AmpRabiRunResult, AmpRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose five independent calibrations from the captured source.

        req pairs the source RunRecord, fitted gains and captured ctx.parameters.
        With a cfg, return pulse-module proposals at qubit.pulses.pi_amp/pi2_amp,
        separate x180/x90 slot proposals containing their resolved library keys,
        and qubit.ext.pi_product = pi_gain * source waveform.length (a.u. * us).
        Same-key slots are still proposed. Without cfg return no items.
        Only module gains change; source cfg is not mutated. Missing capture,
        role resolution, non-scalar source length and conversion/asset faults
        fail explicitly. No hardware, live calibration reads or writes occur.
        """
        snapshot = req.run_result.cfg
        if snapshot is None:
            return []
        parameters = req.ctx.parameters
        if parameters is None:
            raise ValueError("Amplitude Rabi writeback requires captured parameters")
        pulse = snapshot.modules.qub_pulse
        length = pulse.waveform.length
        if not isinstance(length, (int, float)):
            raise ValueError(
                "Amplitude Rabi pi_product requires a scalar source length"
            )
        result = req.analyze_result
        items: list[WritebackItem] = []
        for target, slot, gain, description in (
            ("$qubit.pulses.pi_amp", "$qubit.module.x180", result.pi_gain, "Pi pulse"),
            (
                "$qubit.pulses.pi2_amp",
                "$qubit.module.x90",
                result.pi2_gain,
                "Pi/2 pulse",
            ),
        ):
            key = parameters.library_key(target)
            spec, value = module_cfg_to_value(pulse)
            value.with_field("gain", gain)
            items.append(
                ModuleWriteback(
                    target_name=target,
                    description=description,
                    edit_schema=CfgSchema(spec=spec, value=value),
                )
            )
            items.append(
                ParameterWriteback(
                    target_name=slot,
                    description=f"{description} slot",
                    proposed_value=key,
                )
            )
        items.append(
            ParameterWriteback(
                target_name="$qubit.ext.pi_product",
                description="Pi pulse gain times length (a.u. * us)",
                proposed_value=result.pi_gain * length,
            )
        )
        return items

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_amp_rabi_{time.strftime('%m%d')}"
